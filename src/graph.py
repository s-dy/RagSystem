import re
from typing import List, Optional, Literal, Dict, Any
from pydantic import BaseModel, Field

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import AIMessage, HumanMessage,AnyMessage
from langchain_core.runnables import RunnableConfig
from langgraph.store.redis import AsyncRedisStore,BaseStore
from langgraph.checkpoint.redis import AsyncRedisSaver
from langgraph.graph import MessagesState, StateGraph, START, END

from config.Config import QueryEnhancementConfig,REDIS_URI
from src.core.router.query_router import QueryRouter
from src.core.shared.adapter import CommonTaskAdapterHandler
from src.core.shared.query_enhancer import QueryEnhancer
from src.core.shared.fusion_retrieve import FusionRetrieve
from src.core.shared.task_analyzer import TaskCharacteristics, TaskAnalyzer, TaskType
from src.monitoring.langfuse_monitor import langfuse_handler
from src.monitoring.logger import monitor_task_status
from src.services.llm.models import get_qwen_model, get_embedding_model
from src.services.relation_db import MySQLConnector
from src.services.tools.ToolsPool import ToolsPool
from src.services.tools.agent import ToolsAgent
from utils.async_task import async_run


class State(MessagesState):
    original_query: str  # 原始查询
    task_characteristics: Optional[TaskCharacteristics] # 任务分析特征
    need_retrieval: bool  # 是否需要增强检索
    enhanced_queries: Optional[List[Dict[str,Any]]]  # 增强查询
    router_index: Optional[Dict[str, List[str]]] # 查询路由
    search_content: Optional[str]  # 检索内容整合
    run_count: int  # 运行次数


class GradeDocuments(BaseModel):
    """Binary score for relevance check on retrieved documents."""
    binary_score: str = Field(
        description="检索到的文档与问题相关性，'yes' or 'no'"
    )
    reasoning: Optional[str] = Field(
        description="简要推理",
        default=None
    )


def get_last_user_msg(messages: List[AnyMessage]):
    user_messages = [msg for msg in messages if isinstance(msg, HumanMessage)]
    if not user_messages:
        return None
    last_msg = user_messages[-1].content
    monitor_task_status('last human message', last_msg)
    return last_msg


class Graph:
    def __init__(self):
        self.llm = get_qwen_model()
        self.embedding = get_embedding_model('qwen')
        self.workflow = None

        # 任务适配器
        self.task_adapter_handlers = [CommonTaskAdapterHandler()]

    async def _init_graph(self):
        graph = StateGraph(State)

        graph.add_node('generate_query_or_respond', self._generate_query_or_respond)
        graph.add_node('query_enhancer', self._query_enhancer)
        graph.add_node('query_router', self._query_router)
        graph.add_node('fusion_retrieve', self._fusion_retrieve)
        graph.add_node('generate_response', self._generate_response)
        # graph.add_node('grade_answer_quality',self._grade_answer_quality)
        # graph.add_node('try_different_approach',self._try_different_approach)

        # 构建图结构
        graph.add_edge(START, 'generate_query_or_respond')
        graph.add_conditional_edges('generate_query_or_respond', self._retrieve_condition,
                                    {'retrieve': 'query_enhancer', 'end': END})
        graph.add_edge('query_enhancer', 'query_router')
        graph.add_edge('query_router', 'fusion_retrieve')
        graph.add_conditional_edges('fusion_retrieve', self._grade_documents,
                                    {'rewrite': 'query_enhancer', 'generate': 'generate_response'})
        graph.add_edge('generate_response', END)
        # graph.add_edge('generate_response', 'grade_answer_quality')
        # graph.add_conditional_edges('grade_answer_quality', self._grade_answer_quality_conditional,{'good':END,'bad':'try_different_approach'})
        # graph.add_conditional_edges('try_different_approach',self._try_different_approach_conditional,{'yes':'query_enhancer','no':END})


        async with (AsyncRedisSaver.from_conn_string(REDIS_URI) as redis_checkpointer,AsyncRedisStore.from_conn_string(REDIS_URI) as redis_store):
            self.workflow = graph.compile(checkpointer=redis_checkpointer,store=redis_store)

    async def _generate_query_or_respond(self, state: State) -> dict:
        """决定是使用检索工具搜索信息，还是直接回复用户"""
        monitor_task_status("---GENERATE QUERY OR RESPOND---")
        query = get_last_user_msg(state['messages'])

        # 分析任务特征
        task_analyzer = TaskAnalyzer()
        task_char = task_analyzer.analyze_task(query)
        monitor_task_status("Task Analysis Result", repr(task_char))


        prompt = ChatPromptTemplate.from_template("""
        你需要判断用户的查询是否需要检索外部信息来回答。

        规则：
        1. 如果查询是简单的问候、闲聊或不需要外部知识就能回答的问题，直接回复。
        2. 如果查询需要具体的事实、数据、知识或最新信息，则需要检索。
        3. 常见的需要检索的情况包括：事实查询、技术问题、新闻事件、专业知识等。
        4. 常见的可以直接回答的情况包括：问候、简单聊天、无需外部知识的推理问题等。

        你的响应必须是以下格式之一：
        - 如果直接回答，输出完整的回复内容
        - 如果需要检索，只输出：NEED_RETRIEVAL
        
        以下是用户的查询：
        {query}
        """)
        chain = prompt | self.llm | StrOutputParser()
        response = await chain.ainvoke({'query': query})
        # 检查是否需要检索
        if response.strip() == "NEED_RETRIEVAL":
            return {
                'messages': [AIMessage(content="我将为您检索相关信息。")],
                'original_query':query,
                'need_retrieval':True,
                "run_count":0,
                'task_characteristics': task_char
            }
        else:
            # 如果可以直接回答，返回响应
            return {'messages': [response], 'need_retrieval':False}

    async def _retrieve_condition(self, state: State) -> Literal['end', 'retrieve']:
        """判断是否需要检索"""
        monitor_task_status("---CHECK RETRIEVAL CONDITION---")
        return 'retrieve' if state.get('need_retrieval') else 'end'

    def _get_enhancer_config_by_task(self, task_char: TaskCharacteristics, run_count: int) -> QueryEnhancementConfig:
        """根据任务特征生成增强配置"""
        # 默认基础配置
        config = {
            'paraphrase': False,  # 同义改写
            'formalize': False,  # 专业术语规范化
            'expand': False,  # 上下文扩展
            'enable_query_decomposition': False,  # 查询分解
            'hyde_predict': False  # 假设文档生成
        }

        # 重试时的降级策略（避免过度增强导致噪声）
        if run_count > 1:
            # 仅保留最核心的增强方式
            if task_char.task_type == TaskType.ANALYTICAL_COMPARISON:
                config['enable_query_decomposition'] = True  # 分解仍是关键
            elif task_char.requires_real_time_data:
                config['hyde_predict'] = True  # 实时数据仍需假设
            return QueryEnhancementConfig(**config)

        # 根据任务类型和特征动态配置
        task_type = task_char.task_type

        if task_type == TaskType.ANALYTICAL_COMPARISON:
            # 分析对比：必须分解 + 扩展维度
            config['enable_query_decomposition'] = True
            config['expand'] = True
            # 如果比较对象多，加强同义改写
            if task_char.comparison_count >= 2:
                config['paraphrase'] = True

        elif task_type == TaskType.PROCEDURAL_QUERY:
            # 流程查询：扩展步骤细节
            config['expand'] = True
            config['formalize'] = True  # 使用标准操作术语

        elif task_type == TaskType.FACT_RETRIEVAL:
            # 事实检索：同义改写提高召回率
            config['paraphrase'] = True
            # 如果有数值或实体，加强扩展
            if task_char.numeric_values or task_char.entities:
                config['expand'] = True

        elif task_type == TaskType.COMPLEX_PLANNING:
            # 复杂规划：需要分解 + 扩展
            config['enable_query_decomposition'] = True
            config['expand'] = True
            config['formalize'] = True

        elif task_type == TaskType.MULTI_STEP_EXECUTION:
            # 多步骤执行：必须分解
            config['enable_query_decomposition'] = True
            if task_char.steps_required > 3:
                config['expand'] = True  # 步骤多时扩展细节

        elif task_type == TaskType.REAL_TIME_INTERACTION:
            # 实时交互：HyDE 预测 + 同义改写
            config['hyde_predict'] = True

        elif task_type == TaskType.VALIDATION_VERIFICATION:
            # 验证核查：精确匹配优先，少用扩展
            config['paraphrase'] = True  # 保持语义不变的改写
            config['formalize'] = True  # 使用标准验证术语

        elif task_type == TaskType.CREATIVE_GENERATION:
            # 创造性生成：通常不需要检索增强
            # 但若进入此分支，可能是混合任务，启用扩展
            config['expand'] = True
            config['paraphrase'] = True

        # 兜底策略：如果以上都没触发，至少启用 paraphrase
        if not any(config.values()):
            config['paraphrase'] = True

        return QueryEnhancementConfig(**config)

    async def _query_enhancer(self, state: State) -> dict:
        """查询增强节点"""
        monitor_task_status("---ENHANCE QUERY---")
        # 查询增强
        task_char = state.get('task_characteristics')
        # 动态配置 QueryEnhancer
        config = self._get_enhancer_config_by_task(task_char, state['run_count'])

        enhancer = QueryEnhancer(self.llm,config)
        # 获取原始用户查询
        original_query = state['original_query']
        if not original_query:
            return {'enhanced_queries': []}

        enhanced_queries = await enhancer.enhance(original_query)
        # 返回增强后的查询列表
        if not enhanced_queries:
            enhanced_queries = enhancer.parse_query_time([original_query])  # 回退到原始查询
        return {'enhanced_queries': enhanced_queries,'run_count':state.get('run_count', 0) + 1}

    async def _query_router(self, state: State) -> dict:
        """查询路由节点"""
        monitor_task_status("---QUERY ROUTING---")
        query_route = QueryRouter(self.llm)
        # 获取知识库配置
        knowledge_bases = MySQLConnector().get_all_collections()
        enhanced_queries = state.get('enhanced_queries', [])

        queries = [_['query'] for _ in enhanced_queries]
        route_result = await query_route.multi_all_queries_index_router(queries, knowledge_bases)
        internal_routes = {}
        # 一次性路由全部查询时的处理方式
        for route in route_result:
            idx = route["index"]
            internal_routes[idx] = queries
        # 一次路由一个查询的处理方式
        # for query, routes in route_result:
        #     for route in routes:
        #         idx = route["index"]
        #         if idx not in internal_routes:
        #             internal_routes[idx] = []
        #         internal_routes[idx].append(query)
        return {'router_index': internal_routes}

    async def _retrieve_internal(self,router_index):
        """处理内部检索"""
        doc_result = []
        if not router_index:
            return doc_result
        monitor_task_status('开始处理内部检索')
        search_model = FusionRetrieve()
        # 提取检索结果
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

    async def _retrieve_external(self,query:str) -> list:
        """处理外部检索"""
        external_docs = []
        if not query:
            return external_docs
        monitor_task_status('开始调用外部工具')
        external_docs = []

        prompt = ChatPromptTemplate.from_template("""
        你是一个优化在搜索引擎中搜索查询的助手，需要重写用户问题，便于使用搜索引擎搜索。
        
        用户问题：
        {question}
        
        要求：
        1. 只返回最终的答案，不用返回其他无关内容。
        """)
        chain = prompt | self.llm | StrOutputParser()
        search_query = await chain.ainvoke({'question': query})
        monitor_task_status('rewrite search query',search_query)
        search_tool = ToolsPool().get_tool('bing_search')
        search_result = await search_tool.ainvoke(search_query)
        external_docs.append(search_result)
        monitor_task_status('外部工具调用完成',external_docs)
        return external_docs

    async def _fusion_retrieve(self, state: State) -> dict:
        """融合检索节点"""
        monitor_task_status("---FUSION RETRIEVAL---")
        task_characteristics = state.get('task_characteristics')
        # 处理内部检索
        internal_docs = await self._retrieve_internal(state['router_index'])
        # 调用外部工具的条件 ，待补充
        external_docs = []
        if not internal_docs or task_characteristics.requires_external_tools:
            # 处理外部检索
            external_docs = await self._retrieve_external(state['original_query'])
        # 合并结果
        all_docs = internal_docs + external_docs

        # 去重结果
        seen = set()
        unique_docs = []
        for doc in all_docs:
            normalized = doc.lower().strip()
            # 移除标点符号
            normalized = re.sub(r'[^\w\s\u4e00-\u9fa5]', '', normalized)
            if normalized and normalized not in seen:
                seen.add(normalized)
                unique_docs.append(doc)

        monitor_task_status("fusion_unique_docs", unique_docs)
        if unique_docs:
            content = "\n\n".join(unique_docs)  # 限制内容长度
            return {
                'search_content': content
            }

        return {
            'search_content': ""
        }

    async def _grade_documents(self, state: State) -> Literal['rewrite', 'generate']:
        print("---CHECK DOCUMENT RELEVANCE TO QUESTION---")
        question = state['original_query']
        docs = state['search_content']
        monitor_task_status("run_count",state['run_count'])
        if not question or not docs or state['run_count'] >= 2:   # 运行次数上限 2
            monitor_task_status('not enough question or documents or achieve max retries')
            return 'generate'

        prompt = ChatPromptTemplate.from_template("""
            您是一名评分者，正在评估检索到的文档与用户问题的相关性。

            请严格按照以下 JSON 格式输出您的评分结果：
            {{
                "binary_score": "yes" or "no",
            }}

            检索到的文档：
            {docs}

            用户提问：
            {question}
            
            1. 如果文档包含与用户问题相关的关键字或语义，请将 binary_score 设置为 "yes"，否则设置为 "no"。
            """)
        # 首次检索：用 LLM 评分
        if state['run_count'] == 1:
            chain = prompt | self.llm.with_structured_output(GradeDocuments)
            score = await chain.ainvoke({'question': question, 'docs': docs})
            is_relevant = (score.binary_score == "yes")
        else:
            # 重试时：用关键词匹配
            is_relevant = self._keyword_relevance(question, docs)
        monitor_task_status('grade_documents relevant', is_relevant)
        if is_relevant:
            return "generate"
        else:
            return "rewrite"

    def _keyword_relevance(self, question: str, docs: str) -> bool:
        """基于关键词的快速相关性判断"""
        q_words = set(re.findall(r'\w+', question.lower()))
        d_words = set(re.findall(r'\w+', docs.lower()))
        common = q_words & d_words
        return len(common) >= max(1, len(q_words) // 3)  # 至少匹配1/3关键词

    async def _generate_response(self, state: State) -> dict:
        """生成最终响应"""
        monitor_task_status("---GENERATE FINAL RESPONSE---")

        # 获取原始用户查询
        question = state['original_query']
        if not question:
            return {"messages": [AIMessage(content="未找到用户查询。")]}

        # 获取检索到的文档内容
        docs_content = state.get('search_content') or "未找到相关信息"

        SYSTEM_PROMPT = """
        你是一个专业、严谨的智能助手，请严格根据以下提供的【上下文】回答用户问题。

        规则：
        1. 如果【上下文】包含问题的答案，请用简洁、准确的语言回答。
        2. 如果【上下文】与问题无关或信息不足，请回答："根据现有资料，无法回答该问题。"
        3. 严格按照【上下文】中的内容回答，不要编造信息，不要提及"上下文"或"文档"，直接给出答案。
        4. 回答尽量简短。
        5. 如果答案来自多个来源，进行总结和整合。

        【上下文】
        {content}
        """
        system_msg = SYSTEM_PROMPT
        if self.task_adapter_handlers:
            monitor_task_status('task adapter start')
            for adapter_handler in self.task_adapter_handlers:
                for route_index in state['router_index']:
                    if not adapter_handler.support(route_index):
                        continue
                    system_msg = adapter_handler.dispatch(system_msg)
            monitor_task_status(f'task adapter end')

        prompt = ChatPromptTemplate.from_messages([
            ("system", system_msg),
            ("human", "{query}")
        ])

        chain = prompt | self.llm | StrOutputParser()

        # 调用模型生成响应
        response = await chain.ainvoke({'query': question, 'content': docs_content})

        return {"messages": [AIMessage(content=response)]}

    @property
    async def graph(self):
        if self.workflow is None:
            await self._init_graph()
        return self.workflow.with_config(callbacks=[langfuse_handler])


if __name__ == '__main__':
    async def test_main():
        graph = await Graph().graph

        # 测试用例1：需要检索的问题
        inputs = {
            "messages": [{"role": "user", "content": "搜索transformer的一些讲解文章，并进行总结"}],
        }
        user_id = "1"
        config:RunnableConfig = {'configurable':{'thread_id':'1','user_id':user_id}}
        print("=" * 50)
        async for output in graph.astream(inputs,config=config):
            for key, value in output.items():
                print(f"Node '{key}':")
                if "messages" in value:
                    print(f"内容: {value['messages'][-1]}")
                elif "enhanced_queries" in value:
                    print(f"增强查询: {value['enhanced_queries']}")
                elif "search_content" in value:
                    print(f"检索内容（摘要）: {value['search_content']}...")
                print("-" * 50)

    async_run(test_main())