import os
import re
import time
from typing import List, Optional, Dict

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableConfig
from langgraph.store.base import BaseStore
from langgraph.store.postgres.aio import AsyncPostgresStore
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from langgraph.graph import MessagesState, StateGraph, START, END

from config.Config import RagSystemConfig, POSTGRESQL_URL, QueryEnhancementConfig
from src.node.route import QueryRouter
from src.core.adapter import CommonTaskAdapterHandler
from src.node.expansion.query_enhancer import QueryEnhancer
from src.node.generate import retrieve_answer_or_retrieve, generate_answer_for_query,synthesize_final_subs
from src.node.retrieval.fusion_retrieve import FusionRetrieve
from src.services.task_analyzer import TaskCharacteristics, TaskAnalyzer, TaskType
from src.monitoring.langfuse_monitor import langfuse_handler
from src.monitoring.logger import monitor_task_status
from src.services.CrossEncoderRanker import CrossEncoderRanker
from src.services.GradeModel import DocumentGrader
from src.services.llm.models import get_qwen_model, get_embedding_model
from src.services.relation_db import MySQLConnector
from src.services.tools.ToolsPool import ToolsPool
from utils.message_util import get_last_user_msg, get_conversation_context


class State(MessagesState):
    original_query: str  # 原始查询
    task_characteristics: Optional[TaskCharacteristics]  # 任务分析特征
    need_retrieval: bool  # 是否需要增强检索
    router_index: Optional[Dict[str, List[str]]]  # 查询路由
    search_content: Optional[str]  # 检索内容整合
    run_count: int  # 运行次数
    answer: str

    sub_questions: List[str]  # 待解决的子问题队列（顺序执行）
    current_sub_question: Optional[str]  # 当前正在处理的子问题
    reasoning_context: str  # 已解决的上下文（用于注入后续步骤）


class Graph:
    def __init__(self, config: RagSystemConfig = None):
        if not config:
            config = RagSystemConfig()
        self.config = config
        self.llm = get_qwen_model()
        self.embedding = get_embedding_model('qwen')
        self.task_adapter_handlers = [CommonTaskAdapterHandler()]
        self.cross_encoder_ranker = CrossEncoderRanker()
        self.document_grader = DocumentGrader(threshold=0.5)
        self.tools_pool = ToolsPool()

    async def start(self, *args, **kwargs):
        workflow = await self._init_graph()
        if not self.tools_pool.init_instance:
            await self.tools_pool.initialize()

        if os.getenv('IS_LANGSMITH') == 'True':
            graph = workflow.compile().with_config(callbacks=[langfuse_handler])
            return await graph.ainvoke(*args, **kwargs)
        else:
            async with (
                AsyncPostgresSaver.from_conn_string(POSTGRESQL_URL) as checkpointer,
                AsyncPostgresStore.from_conn_string(POSTGRESQL_URL) as store
            ):
                await store.setup()
                await checkpointer.setup()
                graph = workflow.compile(
                    checkpointer=checkpointer,
                    store=store
                ).with_config(callbacks=[langfuse_handler])
                return await graph.ainvoke(*args, **kwargs)

    async def _init_graph(self):
        graph = StateGraph(State)

        # 统一节点
        graph.add_node('retrieve_or_respond', self._retrieve_or_respond)
        graph.add_node('prepare_next_step', self._prepare_next_step)
        graph.add_node('enhance_and_route_current', self._enhance_and_route_current)
        graph.add_node('fusion_retrieve', self._fusion_retrieve)
        graph.add_node('generate_current_answer', self._generate_current_answer)
        graph.add_node('synthesize', self._synthesize)
        graph.add_node('final', self._final)

        # 图结构
        graph.add_edge(START, 'retrieve_or_respond')

        # 主路由：是否需要检索
        graph.add_conditional_edges(
            'retrieve_or_respond',
            lambda s: 'end' if not s.get('need_retrieval') else 'prepare_next_step',
            {'end': END, 'prepare_next_step': 'prepare_next_step'}
        )

        # 统一流程
        graph.add_edge('prepare_next_step', 'enhance_and_route_current')
        graph.add_edge('enhance_and_route_current', 'fusion_retrieve')
        graph.add_conditional_edges(
            'fusion_retrieve',
            self._grade_documents,
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

    async def _retrieve_or_respond(self, state: State, config: RunnableConfig, store: BaseStore) -> dict:
        """统一入口：决定是否需要检索，并初始化多跳结构"""
        monitor_task_status("---GENERATE QUERY OR RESPOND---")
        query = get_last_user_msg(state['messages'])
        conversation_context = get_conversation_context(state['messages'], num_messages=5)

        task_analyzer = TaskAnalyzer()
        task_char = task_analyzer.analyze_task(query)
        monitor_task_status("Task Analysis Result", repr(task_char))

        # 检查是否需要检索
        response = await retrieve_answer_or_retrieve(self.llm, query, conversation_context)
        is_need_retrieval = response.strip().find("NEED_RETRIEVAL") > -1 or task_char.task_type == TaskType.FACT_RETRIEVAL

        if not is_need_retrieval:
            return {'need_retrieval': False}

        if task_char.is_multi_hop:
            # 多跳：需要分解（空列表表示待分解）
            sub_questions = []
        else:
            # 单跳：直接作为一步多跳
            sub_questions = [query]

        thread_id = config['configurable'].get('thread_id', 'default')
        user_id = config['configurable'].get('user_id', 'default')
        if thread_id and user_id:
            await store.aput((user_id, thread_id,), key=f'need_retrieval_{int(time.time() * 1000)}',
                             value={'query': query, "result": True})

        return {
            'original_query': query,
            'need_retrieval': True,
            'run_count': 0,
            'task_characteristics': task_char,
            'sub_questions': sub_questions,
            'reasoning_context': ""
        }

    async def _prepare_next_step(self, state: State, config: RunnableConfig, store: BaseStore) -> dict:
        """统一预处理：处理子问题分解或准备当前子问题"""

        # 情况1：需要分解多跳问题（sub_questions 为空列表）
        if not state["sub_questions"]:
            monitor_task_status("---DECOMPOSING MULTI-HOP QUESTION---")
            enhancer_config = QueryEnhancementConfig(decompose_to_subquestions=True)
            enhancer = QueryEnhancer(self.llm, enhancer_config)
            query = state["original_query"]
            conversation_context = get_conversation_context(state["messages"], num_messages=5)
            enhanced_result = await enhancer.enhance(query, conversation_context=conversation_context)

            if enhanced_result:
                sub_questions = [item["query"] for item in enhanced_result]
            else:
                sub_questions = [query]  # 分解失败回退到单跳

            return {
                "sub_questions": sub_questions,
                "current_sub_question":sub_questions[0],
            }

        # 情况2：已有子问题，准备当前子问题
        if state["sub_questions"]:
            current = state["sub_questions"][0]

            return {
                "current_sub_question": current,
            }

        # 情况3：无子问题（安全兜底）
        return {"current_sub_question": None}

    async def _enhance_and_route_current(self, state: State, config: RunnableConfig, store: BaseStore) -> dict:
        """增强当前查询"""
        # 获取当前查询和上下文
        if state.get("current_sub_question"):
            # 多跳子问题
            query = state["current_sub_question"]
            conversation_context = state.get("reasoning_context", "")
            task_char = TaskCharacteristics(task_type=TaskType.FACT_RETRIEVAL)
            run_count = 0
        else:
            # 单跳主问题
            query = state["original_query"]
            conversation_context = get_conversation_context(state["messages"], num_messages=5)
            task_char = state["task_characteristics"]
            run_count = state.get("run_count", 0)

        # 查询增强
        enhancer_config = self._get_enhancer_config_by_task(task_char, run_count)
        enhancer = QueryEnhancer(self.llm, enhancer_config)
        enhanced_result = await enhancer.enhance(query, conversation_context=conversation_context)

        if not enhanced_result:
            enhanced_result = enhancer.parse_query_time([query])

        thread_id = config['configurable'].get('thread_id', 'default')
        user_id = config['configurable'].get('user_id', 'default')
        if thread_id and user_id:
            await store.aput((user_id, thread_id,), key=f'query_enhancer_{int(time.time() * 1000)}',
                             value={
                                 "enhanced_queries": enhanced_result,
                                 "enhancer_config": enhancer_config.__dict__,
                             })

        # 查询路由
        monitor_task_status("---QUERY ROUTING---")
        queries = [_['query'] for _ in enhanced_result]
        query_route = QueryRouter(self.llm)
        knowledge_bases = MySQLConnector().get_all_collections()
        route_result = await query_route.multi_all_queries_index_router(queries, knowledge_bases)
        internal_routes = {}
        for route in route_result:
            idx = route["index"]
            internal_routes[idx] = queries

        result = {'router_index': internal_routes}
        if not state.get("current_sub_question"):  # 只有主问题才更新 run_count
            result['run_count'] = run_count + 1

        return result

    def _get_enhancer_config_by_task(self, task_char: TaskCharacteristics, run_count: int) -> QueryEnhancementConfig:
        """根据任务特征生成增强配置"""
        config = {
            'paraphrase': False,
            'formalize': False,
            'expand': False,
            'enable_query_decomposition': False,
            'hyde_predict': False
        }

        task_type = task_char.task_type
        if run_count > 1:
            if task_type == TaskType.ANALYTICAL_COMPARISON:
                config['enable_query_decomposition'] = True
            elif task_type == TaskType.REAL_TIME_INTERACTION:
                config['hyde_predict'] = True
            return QueryEnhancementConfig(**config)

        if task_type == TaskType.ANALYTICAL_COMPARISON:
            config['enable_query_decomposition'] = True
            config['expand'] = True
        elif task_type == TaskType.PROCEDURAL_QUERY:
            config['expand'] = True
            config['formalize'] = True
        elif task_type == TaskType.FACT_RETRIEVAL:
            config['paraphrase'] = True
        elif task_type == TaskType.COMPLEX_PLANNING:
            config['enable_query_decomposition'] = True
            config['expand'] = True
            config['formalize'] = True
        elif task_type == TaskType.MULTI_STEP_EXECUTION:
            config['enable_query_decomposition'] = True
            config['expand'] = True
        elif task_type == TaskType.REAL_TIME_INTERACTION:
            config['hyde_predict'] = True
            config['paraphrase'] = True
        elif task_type == TaskType.VALIDATION_VERIFICATION:
            config['paraphrase'] = True
            config['formalize'] = True
        elif task_type == TaskType.CREATIVE_GENERATION:
            config['expand'] = True
            config['paraphrase'] = True

        if not any(config.values()):
            config['paraphrase'] = True

        return QueryEnhancementConfig(**config)

    async def _retrieve_internal(self, router_index):
        """处理内部检索"""
        doc_result = []
        if not router_index:
            return doc_result
        monitor_task_status('开始处理内部检索')
        search_model = FusionRetrieve()
        for collection_index, queries in router_index.items():
            try:
                result = await search_model.search_queries(queries, collection_name=collection_index)
                if not result:
                    continue
                for doc_list in result:
                    if doc_list:
                        doc_result.extend(doc_list)
            except Exception as e:
                monitor_task_status(f"检索出错: {str(e)}")
        monitor_task_status('内部检索完成')
        return doc_result

    async def _retrieve_external(self, query: str) -> list:
        """处理外部检索"""
        external_docs = []
        if not query:
            return external_docs
        monitor_task_status('开始调用外部工具')

        prompt = ChatPromptTemplate.from_template("""
        你是一个优化在搜索引擎中搜索查询的助手，需要重写用户问题，便于使用搜索引擎搜索。

        以下是需要重写的用户问题：
        【{question}】

        要求：
        1. 只返回最终的答案，不用返回其他无关内容。
        """)
        chain = prompt | self.llm | StrOutputParser()
        search_query = await chain.ainvoke({'question': query})
        monitor_task_status('rewrite search query', search_query)

        search_result = self.tools_pool.get_response(
            await self.tools_pool.call_tool('bing_search', {'query': search_query})
        )
        uids, uid_map = [], {}
        if len(search_result):
            results = search_result[0].get('results')
        else:
            results = []
        for item in results:
            uids.append(item['uuid'])
            uid_map[item['uuid']] = item['url']

        pages_results = self.tools_pool.get_response(
            await self.tools_pool.call_tool('crawl_webpage', {'uuids': uids, 'url_map': uid_map})
        )
        if pages_results:
            pages_results = pages_results[0]
        for item in pages_results:
            if item.get('content'):
                external_docs.append(item['content'])

        monitor_task_status('外部工具调用完成', external_docs)
        return external_docs

    async def _fusion_retrieve(self, state: State, config: RunnableConfig, store: BaseStore) -> dict:
        """融合检索节点"""
        monitor_task_status("---FUSION RETRIEVAL---")
        task_characteristics = state.get('task_characteristics') or TaskCharacteristics(
            task_type=TaskType.FACT_RETRIEVAL, requires_external_tools=False
        )

        # 内部检索
        internal_docs = await self._retrieve_internal(state['router_index'])

        # 外部检索
        external_docs = []
        query = state.get("current_sub_question") or state["original_query"]
        if not internal_docs or task_characteristics.requires_external_tools:
            external_docs = await self._retrieve_external(query)

        # 融合 + 去重 + 重排序
        all_docs = internal_docs + external_docs
        seen = set()
        unique_docs = []
        for doc in all_docs:
            normalized = re.sub(r'[^\w\s\u4e00-\u9fa5]', '', doc.lower().strip())
            if normalized and normalized not in seen:
                seen.add(normalized)
                unique_docs.append(doc)

        if unique_docs:
            rerank_query = state.get("current_sub_question") or state["original_query"]
            unique_docs = self.cross_encoder_ranker.reranker(rerank_query, unique_docs)
            content = "\n\n".join([doc for doc, score in unique_docs])
        else:
            content = ""

        thread_id = config['configurable'].get('thread_id', 'default')
        user_id = config['configurable'].get('user_id', 'default')
        if thread_id and user_id:
            await store.aput((user_id, thread_id,), key=f"final_retrieval_{int(time.time() * 1000)}", value={
                "unique_docs_count": len(unique_docs),
                "docs": [f"score:{str(score)}\n" + doc for doc, score in unique_docs]
            })

        return {'search_content': content}

    def _grade_documents(self, state: State, config: RunnableConfig, store: BaseStore):
        """检索结果评分节点"""
        question = state.get("current_sub_question") or state["original_query"]
        docs = state.get('search_content','')
        if not docs or state['run_count'] > 2:
            return 'good'
        grade = self.document_grader.grade(question, docs.split('\n\n'))
        if grade:
            return 'good'
        else:
            return 'bad'

    async def _generate_current_answer(self, state: State, config: RunnableConfig, store: BaseStore) -> dict:
        """生成当前查询的答案"""
        current_q = state.get("current_sub_question") or state["original_query"]
        conversation_context = get_conversation_context(state['messages'],num_messages=6)
        docs = state["search_content"] or "未找到相关信息"

        # 判断是否为最终答案：普通问题只剩一个子问题、多跳问题需要处理完所有子问题
        if state['task_characteristics'].is_multi_hop:
            is_final = len(state["sub_questions"]) == 0
        else:
            is_final = len(state["sub_questions"]) == 1

        answer = await generate_answer_for_query(
            self.llm,
            query=current_q,
            docs_content=docs,
            conversation_context=conversation_context,
            is_final=is_final
        )
        monitor_task_status('current answer', answer)

        # 更新状态
        remaining_subs = state["sub_questions"][1:] if state["sub_questions"] else []
        new_reasoning_context = state.get("reasoning_context", "") + f"问题：{current_q}\n答案：{answer}\n\n"

        thread_id = config["configurable"].get("thread_id", "default")
        user_id = config["configurable"].get("user_id", "default")
        if thread_id and user_id:
            if is_final:
                await store.aput((user_id, thread_id,), key=f"final_response_{int(time.time() * 1000)}", value={
                    "question": current_q,
                    "response": answer,
                    "context_used": docs,
                    "is_final": True
                })
            else:
                await store.aput(
                    (user_id, thread_id),
                    key=f"sub_answer_{int(time.time() * 1000)}",
                    value={"question": current_q, "answer": answer}
                )

        # 如果是最终答案，直接设置 answer 字段
        if is_final:
            return {"answer": answer, "sub_questions": []}

        return {
            "sub_questions": remaining_subs,
            "messages": [AIMessage(content=f"[中间] {current_q} → {answer}")],
            "reasoning_context": new_reasoning_context
        }

    async def _synthesize(self,state: State, config: RunnableConfig, store: BaseStore) -> dict:
        """合并答案"""
        if not state['task_characteristics'].is_multi_hop:
            return {}
        question = state["original_query"]
        reasoning_ctx = state.get("reasoning_context", "")
        answer = await synthesize_final_subs(
            self.llm,
            query=question,
            reasoning_context=reasoning_ctx,
        )
        monitor_task_status('synthesize answer', answer)

        thread_id = config["configurable"].get("thread_id", "default")
        user_id = config["configurable"].get("user_id", "default")
        if thread_id and user_id:
            await store.aput((user_id, thread_id,), key=f"synthesize_response_{int(time.time() * 1000)}", value={
                "question": question,
                "response": answer,
            })

        return {"answer": answer}

    def _final(self, state: State):
        """最终输出节点"""
        return {'messages': [AIMessage(content=state['answer'])]}
