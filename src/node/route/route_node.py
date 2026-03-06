"""入口/路由节点 Mixin

包含：
- _retrieve_or_respond: 统一入口，决定是否需要检索
- _prepare_next_step: 子问题分解或准备当前子问题
- _enhance_and_route_current: 查询增强 + 路由
- _get_enhancer_config_by_task: 根据任务特征生成增强配置
"""

import asyncio

from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableConfig
from langgraph.store.base import BaseStore

from .query_router import QueryRouter
from .query_enhancer import QueryEnhancer
from .retrieve_or_response import retrieve_answer_or_retrieve

from config import QueryEnhancementConfig
from src.services.task_analyzer import TaskCharacteristics, TaskAnalyzer, TaskType
from src.services.storage import PostgreSQLConnector
from src.observability.logger import monitor_task_status

from utils.message_util import get_last_user_msg, get_conversation_context


class RouteNodeMixin:
    """入口/路由节点方法集合，通过 Mixin 注入到 Graph 类"""

    async def __retrieve_or_respond(self, state, config: RunnableConfig, store: BaseStore) -> dict:
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
            # LLM 已直接生成回复内容，存入 answer
            return {
                'original_query': query,
                'need_retrieval': False,
                'answer': response,
            }

        if task_char.is_multi_hop:
            # 多跳：需要分解（空列表表示待分解）
            sub_questions = []
        else:
            # 单跳：直接作为一步多跳
            sub_questions = [query]

        monitor_task_status("need_retrieval", {"query": query, "result": True})

        return {
            'original_query': query,
            'need_retrieval': True,
            'run_count': 0,
            'task_characteristics': task_char,
            'sub_questions': sub_questions,
            'reasoning_context': ""
        }

    async def __prepare_next_step(self, state, config: RunnableConfig, store: BaseStore) -> dict:
        """统一预处理：处理子问题分解或准备当前子问题"""

        # 情况1：需要分解多跳问题（sub_questions 为空列表）
        if not state["sub_questions"]:
            monitor_task_status("---DECOMPOSING MULTI-HOP QUESTION---")
            # 只开启子问题分解
            enhancer_config = QueryEnhancementConfig(enable_query_decomposition=True)
            enhancer = QueryEnhancer(self.llm, enhancer_config)
            query = state["original_query"]
            conversation_context = get_conversation_context(state["messages"], num_messages=5)
            enhanced_result = await enhancer.enhance(query, conversation_context=conversation_context)

            if enhanced_result:
                sub_questions = [item["query"] for item in enhanced_result]
            else:
                sub_questions = [query]  # 分解失败回退到单跳

            # 子问题分解可视化：向用户展示分解结果
            decomposition_text = (
                f"🔍 为了更好地回答您的问题，我将其分解为以下子问题：\n"
                + "\n".join(f"  {i + 1}. {q}" for i, q in enumerate(sub_questions))
                + "\n\n正在逐一检索和回答..."
            )
            decomposition_step = {
                "type": "decomposition",
                "original_query": query,
                "sub_questions": sub_questions,
            }

            return {
                "sub_questions": sub_questions,
                "current_sub_question": sub_questions[0],
                "messages": [AIMessage(content=decomposition_text)],
                "reasoning_steps": [decomposition_step],
            }

        # 情况2：已有子问题，准备当前子问题
        if state["sub_questions"]:
            current = state["sub_questions"][0]

            return {
                "current_sub_question": current,
                "grade_retry_count": 0,
            }

        # 情况3：无子问题（安全兜底）
        return {"current_sub_question": None}

    async def __enhance_and_route_current(self, state, config: RunnableConfig, store: BaseStore) -> dict:
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
        enhancer_config = self.__get_enhancer_config_by_task(task_char, run_count)
        enhancer = QueryEnhancer(self.llm, enhancer_config)
        enhanced_result = await enhancer.enhance(query, conversation_context=conversation_context)

        if not enhanced_result:
            enhanced_result = enhancer.parse_query_time([query])

        monitor_task_status("query_enhancer", {
            "enhanced_queries": enhanced_result,
            "enhancer_config": enhancer_config.__dict__,
        })

        # 查询路由
        monitor_task_status("---QUERY ROUTING---")
        queries = [_['query'] for _ in enhanced_result]
        query_route = QueryRouter(self.llm)
        knowledge_bases = await asyncio.to_thread(PostgreSQLConnector().get_all_collections)
        route_result = await query_route.multi_all_queries_index_router(queries, knowledge_bases)
        internal_routes = {}
        for route in route_result:
            idx = route["index"]
            internal_routes[idx] = queries

        result = {'router_index': internal_routes}
        if not state.get("current_sub_question"):  # 只有主问题才更新 run_count
            result['run_count'] = run_count + 1

        # 递增评分重试计数（grade 返回 bad 时会重新进入本节点）
        result['grade_retry_count'] = state.get('grade_retry_count', 0) + 1

        return result

    def __get_enhancer_config_by_task(self, task_char: TaskCharacteristics, run_count: int) -> QueryEnhancementConfig:
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
