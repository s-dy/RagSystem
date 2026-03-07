import asyncio
from typing import List, Optional, Dict

from langchain_core.runnables import RunnableConfig
from langgraph.store.postgres.aio import AsyncPostgresStore
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from langgraph.graph import MessagesState, StateGraph, START, END

from config import RagSystemConfig, POSTGRESQL_URL
from src.core.adapter import CommonTaskAdapterHandler
from src.core.memory_manager import MemoryManager
from src.core.tools_pool import ToolsPool
from src.observability.langfuse_monitor import langfuse_handler
from src.observability.logger import monitor_task_status
from src.services.cross_encoder_ranker import CrossEncoderRanker
from src.services.grade_model import DocumentGrader
from src.services.llm.models import get_qwen_model, get_embedding_model
from src.services.task_analyzer import TaskCharacteristics
from src.node.route import RouteNodeMixin
from src.node.retrieval import RetrievalNodeMixin
from src.node.generate import GenerateNodeMixin


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
        self.embedding = get_embedding_model('qwen')
        self.task_adapter_handlers = [CommonTaskAdapterHandler()]
        self._cross_encoder_ranker = None
        self._document_grader = None
        self._evaluator = None
        self.tools_pool = ToolsPool()
        self.memory_manager = MemoryManager()
        self.graph = None

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
            self._document_grader = DocumentGrader(threshold=self.config.grader_threshold)
        return self._document_grader

    async def _compile_graph(self):
        """编译 graph，使用 PostgreSQL checkpointer + store 持久化

        Returns:
            (compiled_graph, (conn_ctx, store_ctx))
        """
        # 构建graph
        workflow = await self._init_graph()
        # 初始化 PostgreSQL 存储
        conn_ctx = AsyncPostgresSaver.from_conn_string(POSTGRESQL_URL)
        store_ctx = AsyncPostgresStore.from_conn_string(POSTGRESQL_URL)
        checkpointer = await conn_ctx.__aenter__()
        store = await store_ctx.__aenter__()
        await store.setup()
        await checkpointer.setup()
        self.memory_manager = MemoryManager(store)
        graph = workflow.compile(
            checkpointer=checkpointer,
            store=store
        ).with_config(callbacks=[langfuse_handler])
        self.graph = graph
        return graph, (conn_ctx, store_ctx)

    async def start(self, input_data: dict, config: RunnableConfig = None):
        graph, ctx = await self._compile_graph()
        try:
            return await graph.ainvoke(input_data, config)
        except Exception as e:
            monitor_task_status(f"推理失败：{e}")
            raise
        finally:
            # 清理数据库连接上下文
            if ctx:
                conn_ctx, store_ctx = ctx
                await store_ctx.__aexit__(None, None, None)
                await conn_ctx.__aexit__(None, None, None)


    async def start_stream(self, input_data: dict, config: RunnableConfig = None):
        """流式输出入口：逐 token 返回 LLM 生成内容和节点状态更新

        使用方式：
            async for event in graph_instance.start_stream(
                {"messages": [HumanMessage(content="你的问题")]},
                {"configurable": {"thread_id": "1", "user_id": "user1"}}
            ):
                print(event)  # {"type": "token", "content": "..."} 或 {"type": "node", ...}
        """
        graph, ctx = await self._compile_graph()

        try:
            async for event in graph.astream_events(input_data, config, version="v2"):
                # 检查当前任务是否已被取消（前端中断时 server 会 cancel 此任务）
                current_task = asyncio.current_task()
                if current_task and current_task.cancelled():
                    monitor_task_status("流式推理被前端中断，停止生成", level="WARNING")
                    break

                event_kind = event.get("event", "")

                # LLM 逐 token 流式输出
                # 只允许 synthesize（最终合成答案）和 final 节点的 token 流式展示
                # generate_current_answer 的 token 不流式展示，因为：
                #   - 子问题答案通过 sub_answer 事件展示
                #   - 最终答案通过 final_answer 事件展示
                #   - 多跳时该节点被多次调用，token 会混在一起
                if event_kind == "on_chat_model_stream":
                    node_name = event.get("metadata", {}).get("langgraph_node", "")
                    if node_name not in ("synthesize", "final"):
                        continue
                    chunk = event.get("data", {}).get("chunk")
                    if chunk and hasattr(chunk, "content") and chunk.content:
                        yield {
                            "type": "token",
                            "content": chunk.content,
                            "node": node_name,
                        }

                # 节点执行完成事件
                elif event_kind == "on_chain_end" and event.get("name") in (
                    "retrieve_or_respond", "prepare_next_step", "generate_current_answer", "synthesize", "final"
                ):
                    output = event.get("data", {}).get("output", {})
                    node_name = event.get("name", "")

                    # 直接回复（不需要检索时，retrieve_or_respond 直接返回答案）
                    if node_name == "retrieve_or_respond":
                        # astream_events v2 中 output 可能是 {node_name: {...}} 或直接 {...}
                        node_output = output.get("retrieve_or_respond", output)
                        if not node_output.get("need_retrieval") and node_output.get("answer"):
                            yield {
                                "type": "final_answer",
                                "answer": node_output["answer"],
                            }

                    # 子问题分解可视化
                    if node_name == "prepare_next_step" and output.get("reasoning_steps"):
                        yield {
                            "type": "decomposition",
                            "sub_questions": output["reasoning_steps"][0].get("sub_questions", []),
                        }

                    # 子问题中间答案
                    if node_name == "generate_current_answer" and output.get("reasoning_steps"):
                        latest_step = output["reasoning_steps"][-1]
                        if not latest_step.get("is_final"):
                            yield {
                                "type": "sub_answer",
                                "sub_question": latest_step.get("sub_question", ""),
                                "answer": latest_step.get("answer", ""),
                            }

                    # 最终答案
                    if output.get("answer") and node_name in ("generate_current_answer", "synthesize"):
                        reasoning_steps = output.get("reasoning_steps", [])
                        if not reasoning_steps or reasoning_steps[-1].get("is_final"):
                            yield {
                                "type": "final_answer",
                                "answer": output["answer"],
                            }

            # 流结束标记
            yield {"type": "done"}

        except asyncio.CancelledError:
            monitor_task_status("流式推理任务被取消，正在清理资源", level="WARNING")
            raise
        finally:
            # 清理数据库连接上下文
            if ctx:
                conn_ctx, store_ctx = ctx
                await store_ctx.__aexit__(None, None, None)
                await conn_ctx.__aexit__(None, None, None)

    async def _init_graph(self):
        graph = StateGraph(State)

        # 统一节点（使用修饰后的名称访问 Mixin 中的双下划线方法）
        graph.add_node('retrieve_or_respond', self._RouteNodeMixin__retrieve_or_respond)
        graph.add_node('prepare_next_step', self._RouteNodeMixin__prepare_next_step)
        graph.add_node('enhance_and_route_current', self._RouteNodeMixin__enhance_and_route_current)
        graph.add_node('fusion_retrieve', self._RetrievalNodeMixin__fusion_retrieve)
        graph.add_node('generate_current_answer', self._GenerateNodeMixin__generate_current_answer)
        graph.add_node('synthesize', self._GenerateNodeMixin__synthesize)
        graph.add_node('final', self._GenerateNodeMixin__final)

        # 图结构
        graph.add_edge(START, 'retrieve_or_respond')

        # 路由：是否需要检索
        graph.add_conditional_edges(
            'retrieve_or_respond',
            lambda s: 'final' if not s.get('need_retrieval') else 'prepare_next_step',
            {'final': 'final', 'prepare_next_step': 'prepare_next_step'}
        )

        # 统一流程
        graph.add_edge('prepare_next_step', 'enhance_and_route_current')
        graph.add_edge('enhance_and_route_current', 'fusion_retrieve')
        graph.add_conditional_edges(
            'fusion_retrieve',
            self._RetrievalNodeMixin__grade_documents,
            {'good':'generate_current_answer','bad':'enhance_and_route_current'}
        )
        # 条件循环：检查是否还有子问题
        graph.add_conditional_edges(
            'generate_current_answer',
            lambda s: 'continue' if s.get('sub_questions') else 'done',
            {'continue': 'prepare_next_step', 'done': 'synthesize'}
        )
        # 注意多跳问题需要走完sub_questions后在走一次generate_current_answer生成最终原始问题的答案
        graph.add_edge('synthesize','final')
        graph.add_edge('final', END)

        return graph
