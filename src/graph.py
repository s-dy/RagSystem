import asyncio
from typing import List, Optional, Dict

from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from langgraph.graph import MessagesState, StateGraph, START, END
from langgraph.store.postgres.aio import AsyncPostgresStore

from config import RagSystemConfig, get_postgresql_url, PostgreSQLConfig
from src.core.adapter import CommonTaskAdapterHandler
from src.core.memory_manager import MemoryManager
from src.core.tools_pool import ToolsPool
from src.node.generate import GenerateNodeMixin
from src.node.retrieval import RetrievalNodeMixin
from src.node.route import RouteNodeMixin
from src.observability.langfuse_monitor import langfuse_handler
from src.observability.logger import get_logger, set_request_id
from src.services.cross_encoder_ranker import CrossEncoderRanker
from src.services.grade_model import DocumentGrader
from src.services.llm.models import get_qwen_model, get_embedding_model
from src.services.task_analyzer import TaskCharacteristics

logger = get_logger(__name__)


class State(MessagesState):
    original_query: str  # 原始查询
    task_characteristics: Optional[TaskCharacteristics]  # 任务分析特征
    need_retrieval: bool  # 是否需要增强检索
    router_index: Optional[Dict[str, List[str]]]  # 查询路由
    search_content: Optional[str]  # 检索内容整合（带来源编号）
    retrieved_documents: List[str]  # 检索到的文档列表，用于评估
    retrieval_scores: List[float]  # 重排序分数，用于置信度计算
    run_count: int  # 运行次数
    grade_retry_count: int  # 当前子问题的评分重试计数（防止 grade 循环）
    answer: str

    sub_questions: List[str]  # 待解决的子问题队列（顺序执行）
    current_sub_question: Optional[str]  # 当前正在处理的子问题
    reasoning_context: str  # 已解决的上下文（用于注入后续步骤）
    reasoning_steps: List[dict]  # 推理过程记录（子问题分解、中间答案等）

    conversation_summary: str  # 渐进式摘要：跨轮对话的累积摘要


class Graph(RouteNodeMixin, RetrievalNodeMixin, GenerateNodeMixin):
    def __init__(self, config: RagSystemConfig = None):
        if not config:
            config = RagSystemConfig()
        self.config = config
        self.llm = get_qwen_model()
        self.embedding = get_embedding_model("qwen")
        self.task_adapter_handlers = [CommonTaskAdapterHandler()]
        self._cross_encoder_ranker = None
        self._document_grader = None
        self._evaluator = None
        self.tools_pool = ToolsPool()
        self.memory_manager = MemoryManager()
        self.graph = None
        # 持久化连接上下文，避免每次请求重新编译
        self._conn_ctx = None
        self._store_ctx = None
        self._compile_lock = asyncio.Lock()

    @property
    def cross_encoder_ranker(self) -> CrossEncoderRanker:
        """懒加载交叉编码器，首次访问时才加载模型"""
        if self._cross_encoder_ranker is None:
            self._cross_encoder_ranker = CrossEncoderRanker()
        return self._cross_encoder_ranker

    @property
    def document_grader(self) -> DocumentGrader:
        """懒加载文档评分器，首次访问时才加载模型"""
        if self._document_grader is None:
            self._document_grader = DocumentGrader(
                threshold=self.config.grader_threshold
            )
        return self._document_grader

    async def _compile_graph(self):
        """编译 graph，使用 PostgreSQL checkpointer + store 持久化。

        首次调用时建立连接并编译，后续调用直接复用已编译的 graph，
        避免每次请求都重建连接池和重新编译 LangGraph。
        """
        # 已编译则直接返回，无需加锁（快路径）
        if self.graph is not None:
            return self.graph

        # 加锁防止并发初始化时重复编译
        async with self._compile_lock:
            # 双重检查：加锁后再判断一次
            if self.graph is not None:
                return self.graph

            workflow = await self._init_graph()

            from src.services.storage.postgres_connector import (
                ensure_postgres_database_exists,
            )

            ensure_postgres_database_exists(PostgreSQLConfig())

            self._conn_ctx = AsyncPostgresSaver.from_conn_string(get_postgresql_url())
            self._store_ctx = AsyncPostgresStore.from_conn_string(get_postgresql_url())
            checkpointer = await self._conn_ctx.__aenter__()
            store = await self._store_ctx.__aenter__()
            await store.setup()
            await checkpointer.setup()
            self.memory_manager = MemoryManager(store)
            self.graph = workflow.compile(
                checkpointer=checkpointer,
                store=store,
            ).with_config(callbacks=[langfuse_handler])
            logger.info("[Graph] LangGraph 编译完成，连接已建立")
            return self.graph

    async def close(self):
        """释放 PostgreSQL 连接，应在应用关闭时调用"""
        if self._store_ctx is not None:
            await self._store_ctx.__aexit__(None, None, None)
            self._store_ctx = None
        if self._conn_ctx is not None:
            await self._conn_ctx.__aexit__(None, None, None)
            self._conn_ctx = None
        self.graph = None
        logger.info("[Graph] 连接已关闭")

    async def start(self, input_data: dict, config: RunnableConfig = None):
        graph = await self._compile_graph()
        # 设置请求ID用于日志追踪
        thread_id = (
            config.get("configurable", {}).get("thread_id", "unknown")
            if config
            else "unknown"
        )
        set_request_id(thread_id)
        try:
            return await graph.ainvoke(input_data, config)
        except Exception as e:
            logger.error(f"[Graph] 推理失败: {e}")
            raise

    async def start_stream(self, input_data: dict, config: RunnableConfig = None):
        """流式输出入口：逐 token 返回 LLM 生成内容和节点状态更新

        使用方式：
            async for event in graph_instance.start_stream(
                {"messages": [HumanMessage(content="你的问题")]},
                {"configurable": {"thread_id": "1", "user_id": "user1"}}
            ):
                print(event)  # {"type": "token", "content": "..."} 或 {"type": "node", ...}
        """
        graph = await self._compile_graph()

        # 设置请求ID用于日志追踪
        thread_id = (
            config.get("configurable", {}).get("thread_id", "unknown")
            if config
            else "unknown"
        )
        set_request_id(thread_id)

        # 是否为多跳场景（用于控制 token 展示策略）
        is_multi_hop = False

        try:
            # 使用 astream 并设置 stream_mode 实现逐 token 流式
            async for event_chunk in graph.astream(
                input_data, config, stream_mode=["messages", "updates", "custom"]
            ):
                # 检查当前任务是否已被取消
                current_task = asyncio.current_task()
                if current_task and current_task.cancelled():
                    logger.warning("[Graph] 流式推理被中断: 前端主动取消")
                    break

                # 处理不同类型的流式输出
                if isinstance(event_chunk, tuple) and len(event_chunk) == 2:
                    mode, chunk = event_chunk

                    if mode == "messages":
                        # 逐 token 流式输出
                        msg_chunk, metadata = chunk
                        node_name = metadata.get("langgraph_node", "")

                        # 只处理 LLM 流式 token（AIMessageChunk），过滤掉节点返回的完整 AIMessage
                        from langchain_core.messages import AIMessageChunk

                        if not isinstance(msg_chunk, AIMessageChunk):
                            continue

                        # Token 展示策略：通过 tags 精确区分核心生成 vs 辅助调用
                        # 核心生成函数的 chain 带有 tags=["stream_to_user"]，辅助调用不带
                        # 多跳场景下，子问题的 token 不展示（由 is_multi_hop 控制）
                        chunk_tags = metadata.get("tags", [])
                        should_stream = "stream_to_user" in chunk_tags
                        if (
                            should_stream
                            and node_name == "generate_current_answer"
                            and is_multi_hop
                        ):
                            should_stream = False

                        if should_stream and msg_chunk.content:
                            yield {
                                "type": "token",
                                "content": msg_chunk.content,
                                "node": node_name,
                            }

                    elif mode == "custom":
                        # 自定义流事件（检索进度）
                        if isinstance(chunk, dict) and chunk.get("type"):
                            yield chunk

                    elif mode == "updates":
                        # 节点更新事件
                        for node_name, node_output in chunk.items():
                            # 子问题分解可视化
                            if node_name == "prepare_next_step":
                                reasoning_steps = node_output.get("reasoning_steps", [])
                                if reasoning_steps and reasoning_steps[0].get(
                                    "sub_questions"
                                ):
                                    is_multi_hop = True
                                    yield {
                                        "type": "decomposition",
                                        "sub_questions": reasoning_steps[0][
                                            "sub_questions"
                                        ],
                                    }

                            # 子问题中间答案 / 单跳最终答案
                            elif node_name == "generate_current_answer":
                                reasoning_steps = node_output.get("reasoning_steps", [])
                                if reasoning_steps:
                                    latest_step = reasoning_steps[-1]
                                    if is_multi_hop and not latest_step.get("is_final"):
                                        yield {
                                            "type": "sub_answer",
                                            "sub_question": latest_step.get(
                                                "sub_question", ""
                                            ),
                                            "answer": latest_step.get("answer", ""),
                                        }

            # 流结束标记
            yield {"type": "done"}

        except asyncio.CancelledError:
            logger.warning("[Graph] 流式推理被取消: 正在清理资源")
            raise

    async def _init_graph(self):
        graph = StateGraph(State)

        # 统一节点（使用修饰后的名称访问 Mixin 中的双下划线方法）
        graph.add_node("retrieve_or_respond", self._RouteNodeMixin__retrieve_or_respond)
        graph.add_node("prepare_next_step", self._RouteNodeMixin__prepare_next_step)
        graph.add_node(
            "enhance_and_route_current", self._RouteNodeMixin__enhance_and_route_current
        )
        graph.add_node("fusion_retrieve", self._RetrievalNodeMixin__fusion_retrieve)
        graph.add_node(
            "generate_current_answer", self._GenerateNodeMixin__generate_current_answer
        )
        graph.add_node("synthesize", self._GenerateNodeMixin__synthesize)
        graph.add_node("final", self._GenerateNodeMixin__final)

        # 图结构
        graph.add_edge(START, "retrieve_or_respond")

        # 路由：是否需要检索
        graph.add_conditional_edges(
            "retrieve_or_respond",
            lambda s: "final" if not s.get("need_retrieval") else "prepare_next_step",
            {"final": "final", "prepare_next_step": "prepare_next_step"},
        )

        # 统一流程
        graph.add_edge("prepare_next_step", "enhance_and_route_current")
        graph.add_edge("enhance_and_route_current", "fusion_retrieve")
        graph.add_conditional_edges(
            "fusion_retrieve",
            self._RetrievalNodeMixin__grade_documents,
            {"good": "generate_current_answer", "bad": "enhance_and_route_current"},
        )
        # 条件循环：检查是否还有子问题
        graph.add_conditional_edges(
            "generate_current_answer",
            lambda s: "continue" if s.get("sub_questions") else "done",
            {"continue": "prepare_next_step", "done": "synthesize"},
        )
        # 注意多跳问题需要走完sub_questions后在走一次generate_current_answer生成最终原始问题的答案
        graph.add_edge("synthesize", "final")
        graph.add_edge("final", END)

        return graph
