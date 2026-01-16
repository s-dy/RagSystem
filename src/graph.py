import os
import re
import time
from typing import List, Optional, Literal, Dict

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableConfig
from langgraph.store.redis import AsyncRedisStore,BaseStore
from langgraph.store.postgres.aio import AsyncPostgresStore
from langgraph.checkpoint.redis import AsyncRedisSaver
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from langgraph.graph import MessagesState, StateGraph, START, END

from config.Config import QueryEnhancementConfig,REDIS_URI,RagSystemConfig,POSTGRESQL_URL
from src.core.router.query_router import QueryRouter
from src.core.shared.adapter import CommonTaskAdapterHandler
from src.core.shared.query_enhancer import QueryEnhancer
from src.core.shared.fusion_retrieve import FusionRetrieve
from src.core.shared.task_analyzer import TaskCharacteristics, TaskAnalyzer, TaskType
from src.core.shared.memory_manager import get_memory_manager
from src.monitoring.langfuse_monitor import langfuse_handler
from src.monitoring.logger import monitor_task_status
from src.services.CrossEncoderRanker import CrossEncoderRanker
from src.services.GradeModel import DocumentGrader
from src.services.llm.models import get_qwen_model, get_embedding_model, get_ollama_deepseek_model
from src.services.relation_db import MySQLConnector
from src.services.tools.ToolsPool import ToolsPool
from src.services.tools.agent import ToolsAgent
from utils.message_util import get_last_user_msg, get_conversation_context


class State(MessagesState):
    original_query: str  # 原始查询
    task_characteristics: Optional[TaskCharacteristics] # 任务分析特征
    need_retrieval: bool  # 是否需要增强检索
    router_index: Optional[Dict[str, List[str]]] # 查询路由
    search_content: Optional[str]  # 检索内容整合
    run_count: int  # 运行次数
    answer: str
    answer_quality: str


class Graph:
    def __init__(self,config: RagSystemConfig = None):
        if not config:
            config = RagSystemConfig()
        self.config = config
        self.llm = get_qwen_model()
        self.embedding = get_embedding_model('qwen')
        # self.workflow:CompiledStateGraph = None

        # 任务适配器
        self.task_adapter_handlers = [CommonTaskAdapterHandler()]
        # 交叉编码器重排序
        self.cross_encoder_ranker = CrossEncoderRanker()
        # 工具池
        self.tools_pool = ToolsPool()

    async def start(self,*args,**kwargs):
        workflow = await self._init_graph()
        # 初始化工具
        if not self.tools_pool.init_instance:
            await self.tools_pool.initialize()

        if os.getenv('IS_LANGSMITH') == 'True':
            # 使用LangSmith时
            graph = workflow.compile().with_config(callbacks=[langfuse_handler])
            response = await graph.ainvoke(*args, **kwargs)
            return response
        else:
            # 普通使用时
            # async with (AsyncRedisSaver.from_conn_string(REDIS_URI) as redis_checkpointer,AsyncRedisStore.from_conn_string(REDIS_URI) as redis_store):
            #     graph = workflow.compile(checkpointer=redis_checkpointer,store=redis_store)
            async with (AsyncPostgresSaver.from_conn_string(POSTGRESQL_URL) as post_checkpointer, AsyncPostgresStore.from_conn_string(POSTGRESQL_URL) as post_store):
                await post_store.setup()
                await post_checkpointer.setup()
                graph = workflow.compile(checkpointer=post_checkpointer, store=post_store).with_config(
                    callbacks=[langfuse_handler])
                response = await graph.ainvoke(*args,**kwargs)
                return response

    async def _init_graph(self):
        graph = StateGraph(State)

        graph.add_node('retrieve_or_respond', self._retrieve_or_respond)
        graph.add_node('query_enhancer_and_route', self._query_enhancer_and_route)
        graph.add_node('fusion_retrieve', self._fusion_retrieve)
        graph.add_node('generate_response', self._generate_response)
        graph.add_node('grade_answer_quality',self._grade_answer_quality)
        graph.add_node('final',self._final)

        # 构建图结构
        graph.add_edge(START, 'retrieve_or_respond')
        graph.add_conditional_edges('retrieve_or_respond', self._retrieve_condition,
                                    {'retrieve': 'query_enhancer_and_route', 'end': END})
        graph.add_edge('query_enhancer_and_route', 'fusion_retrieve')
        graph.add_conditional_edges('fusion_retrieve', self._grade_documents,
                                    {'rewrite': 'query_enhancer_and_route', 'generate': 'generate_response'})
        # graph.add_edge('generate_response', END)
        graph.add_edge('generate_response', 'grade_answer_quality')
        graph.add_conditional_edges('grade_answer_quality', self._grade_answer_quality_conditional,{'good':'final','bad':'query_enhancer_and_route'})
        # self.workflow = graph
        return graph

    async def _retrieve_or_respond(self, state: State,config:RunnableConfig,store:BaseStore) -> dict:
        """决定是使用检索工具搜索信息，还是直接回复用户"""
        monitor_task_status("---GENERATE QUERY OR RESPOND---")
        query = get_last_user_msg(state['messages'])
        conversation_context = get_conversation_context(state['messages'], num_messages=5)  # 获取更多的上下文信息

        # 分析任务特征
        task_analyzer = TaskAnalyzer()
        task_char = task_analyzer.analyze_task(query)
        monitor_task_status("Task Analysis Result", repr(task_char))
        thread_id = config['configurable'].get('thread_id','default')
        user_id = config['configurable'].get('user_id','default')
        if thread_id and user_id:
            await store.aput((user_id,thread_id,), key=f"task_analyzer_{int(time.time()*1000)}", value={"query":query,'result':repr(task_char)})

        # 使用上下文来更好地判断是否需要检索
        prompt = ChatPromptTemplate.from_template("""
        你需要根据当前查询和对话历史来判断是否需要检索外部信息来回答。

        对话历史：
        {conversation_context}

        当前查询：
        {query}

        规则：
        1. 如果查询是简单的问候、闲聊或不需要外部知识就能回答的问题，直接回复。
        2. 如果查询需要具体的事实、数据、知识或最新信息，则需要检索。
        3. 如果当前问题是基于之前对话的追问，且涉及具体信息，也需要检索。
        4. 常见的需要检索的情况包括：事实查询、技术问题、新闻事件、专业知识等。
        5. 常见的可以直接回答的情况包括：问候、简单聊天、无需外部知识的推理问题等。

        你的响应必须是以下格式之一：
        - 如果直接回答，输出完整的回复内容
        - 如果需要检索，只输出：NEED_RETRIEVAL
        """)
        chain = prompt | self.llm | StrOutputParser()
        response = await chain.ainvoke({'query': query, 'conversation_context': conversation_context})
        # 检查是否需要检索
        if response.strip().find("NEED_RETRIEVAL") > -1 or task_char.task_type == TaskType.FACT_RETRIEVAL:
            if thread_id and user_id:
                await store.aput((user_id, thread_id,), key=f'need_retrieval_{int(time.time()*1000)}', value={'query':query,"result": True})
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

        # 根据任务类型和特征动态配置
        task_type = task_char.task_type
        # 重试时的降级策略（避免过度增强导致噪声）
        if run_count > 1:
            # 仅保留最核心的增强方式
            if task_type == TaskType.ANALYTICAL_COMPARISON:
                config['enable_query_decomposition'] = True
            elif task_type == TaskType.REAL_TIME_INTERACTION:
                config['hyde_predict'] = True  # 实时数据仍需假设
            return QueryEnhancementConfig(**config)

        if task_type == TaskType.ANALYTICAL_COMPARISON:
            # 分析对比：必须分解 + 扩展维度
            config['enable_query_decomposition'] = True
            config['expand'] = True

        elif task_type == TaskType.PROCEDURAL_QUERY:
            # 流程查询：扩展步骤细节
            config['expand'] = True
            config['formalize'] = True  # 使用标准操作术语

        elif task_type == TaskType.FACT_RETRIEVAL:
            # 事实检索：同义改写提高召回率
            config['paraphrase'] = True

        elif task_type == TaskType.COMPLEX_PLANNING:
            # 复杂规划：需要分解 + 扩展
            config['enable_query_decomposition'] = True
            config['expand'] = True
            config['formalize'] = True

        elif task_type == TaskType.MULTI_STEP_EXECUTION:
            # 多步骤执行：必须分解
            config['enable_query_decomposition'] = True
            config['expand'] = True

        elif task_type == TaskType.REAL_TIME_INTERACTION:
            # 实时交互：HyDE 预测 + 同义改写
            config['hyde_predict'] = True
            config['paraphrase'] = True

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

    async def _query_enhancer_and_route(self, state: State,config:RunnableConfig,store:BaseStore) -> dict:
        """查询增强与查询路由节点"""
        monitor_task_status("---ENHANCE QUERY---")
        # 查询增强
        task_char = state.get('task_characteristics')
        # 动态配置 QueryEnhancer
        enhancer_config = self._get_enhancer_config_by_task(task_char, state['run_count'])

        enhancer = QueryEnhancer(self.llm,enhancer_config)
        # 获取原始用户查询
        original_query = state['original_query']
        conversation_context = get_conversation_context(state['messages'], num_messages=5)  # 获取对话上下文
        
        if not original_query:
            enhanced_queries = []
        else:
            enhanced_queries = await enhancer.enhance(original_query,conversation_context=conversation_context)
        # 增强后的查询列表
        if not enhanced_queries:
            enhanced_queries = enhancer.parse_query_time([original_query])  # 回退到原始查询

        thread_id = config['configurable'].get('thread_id', 'default')
        user_id = config['configurable'].get('user_id', 'default')
        if thread_id and user_id:
            await store.aput((user_id, thread_id,), key=f'query_enhancer_{int(time.time()*1000)}',
                value={
                    "enhanced_queries": enhanced_queries,
                    "enhancer_config": enhancer_config.__dict__
                },
            )

        # 查询路由节点
        monitor_task_status("---QUERY ROUTING---")
        query_route = QueryRouter(self.llm)
        # 获取知识库配置
        knowledge_bases = MySQLConnector().get_all_collections()
        queries = [_['query'] for _ in enhanced_queries]
        route_result = await query_route.multi_all_queries_index_router(queries, knowledge_bases)
        internal_routes = {}
        # 一次性路由全部查询时的处理方式
        for route in route_result:
            idx = route["index"]
            internal_routes[idx] = queries
        return {'router_index': internal_routes,'run_count':state.get('run_count', 0) + 1}

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
        
        以下是需要重写的用户问题：
        【{question}】
        
        要求：
        1. 只返回最终的答案，不用返回其他无关内容。
        """)
        chain = prompt | self.llm | StrOutputParser()
        search_query = await chain.ainvoke({'question': query})
        monitor_task_status('rewrite search query',search_query)
        search_result = self.tools_pool.get_response(await self.tools_pool.call_tool('bing_search',{'query':search_query}))
        uids,uid_map = [],{}
        if len(search_result):
            results = search_result[0].get('results')
        else:
            results = []
        for item in results:
            uids.append(item['uuid'])
            uid_map[item['uuid']] = item['url']
        pages_results =  self.tools_pool.get_response(await self.tools_pool.call_tool('crawl_webpage',{'uuids':uids,'url_map':uid_map}))
        if pages_results:
            pages_results = pages_results[0]
        for item in pages_results:
            if item.get('content'):
                external_docs.append(item['content'])

        monitor_task_status('外部工具调用完成',external_docs)
        return external_docs

    async def _fusion_retrieve(self, state: State,config:RunnableConfig,store:BaseStore) -> dict:
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

        unique_docs = self.cross_encoder_ranker.reranker(state['original_query'], unique_docs)

        monitor_task_status("fusion_unique_docs", unique_docs)
        if unique_docs:
            content = "\n\n".join([doc for doc,score in unique_docs])
            thread_id = config['configurable'].get('thread_id', 'default')
            user_id = config['configurable'].get('user_id', 'default')
            if thread_id and user_id:
                await store.aput((user_id, thread_id,),key=f"final_retrieval_{int(time.time()*1000)}",value={
                    "unique_docs_count": len(unique_docs),
                    "docs": [f"score:{str(score)}\n" + doc for doc,score in unique_docs]
                })
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

        is_relevant = DocumentGrader(threshold=0.5).grade(question, [docs])
        if is_relevant:
            return "generate"
        else:
            return "rewrite"

    async def _generate_response(self, state: State,config:RunnableConfig,store:BaseStore) -> dict:
        """生成最终响应"""
        monitor_task_status("---GENERATE FINAL RESPONSE---")

        # 获取原始用户查询
        question = state['original_query']
        if not question:
            return {"messages": [AIMessage(content="未找到用户查询。")], "search_content": ""}

        # 获取检索到的文档内容
        docs_content = state.get('search_content') or "未找到相关信息"
        
        # 获取对话上下文
        conversation_context = get_conversation_context(state['messages'], num_messages=5)

        SYSTEM_PROMPT = """
        你是一个专业、严谨的智能助手，请根据以下【检索到的信息】和【对话历史】来回答用户问题。

        检索到的信息：
        {content}
        
        对话历史：
        {conversation_context}

        规则：
        1. 如果【检索到的信息】包含问题的答案，请结合对话历史，用简洁、准确的语言回答。
        2. 如果【检索到的信息】与问题无关或信息不足，但对话历史中有相关信息，请参考已有对话进行回答。
        3. 如果【检索到的信息】与对话历史都无相关信息，请只回答："根据现有资料，无法回答该问题。"
        4. 严格按照【检索到的信息】和【对话历史】中的内容回答，不要编造信息。
        5. 回答要连贯，符合对话上下文。
        6. 如果答案来自多个来源，进行总结和整合。
        7. 回答要简短。
        8. 不要提及【根据检索到的信息】。
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
        response = await chain.ainvoke({'query': question, 'content': docs_content, 'conversation_context': conversation_context})
        monitor_task_status('generating response',response)
        thread_id = config['configurable'].get('thread_id', 'default')
        user_id = config['configurable'].get('user_id', 'default')
        if thread_id and user_id:
            await store.aput((user_id, thread_id,), key=f"final_response_{int(time.time() * 1000)}", value={
                "question": question,
                "response": response,
                "context_used": docs_content,
                "conversation_context": conversation_context
            })

        # return {"messages": [AIMessage(content=response)]}
        return {"answer": response}

    async def _grade_answer_quality(self,state: State,config:RunnableConfig,store:BaseStore):
        """回答评分节点"""
        if state['run_count'] >= 2:
            return {'answer_quality':'good'}
        question = state['original_query']
        answer = state['answer']
        prompt = ChatPromptTemplate.from_template(
            """你是一个评分者，需要根据用户问题来评估回答的准确性。
            
            问题：{question}
            
            回答：{answer}
            
            1. 如果回答准确，返回【good】，如果回答不准确，返回 bad
            2. 如果回答包含 “无法回答该问题”，直接返回 good
            
            只返回 good 和 bad。不需要返回其他任何内容。
            """
        )
        chain = prompt | self.llm | StrOutputParser()
        response = await chain.ainvoke({'question': question, 'answer': answer})
        monitor_task_status('answer quality grade',response)
        thread_id = config['configurable'].get('thread_id', 'default')
        user_id = config['configurable'].get('user_id', 'default')
        if thread_id and user_id:
            await store.aput((user_id, thread_id,), key=f"grade_answer_quality_{int(time.time() * 1000)}", value={'query':question,'answer':answer,'response': response})

        return {'answer_quality': response}

    def _grade_answer_quality_conditional(self,state:State) -> Literal['good', 'bad']:
        answer_quality = state['answer_quality']
        if answer_quality == 'bad':
            return 'bad'
        else:
            return 'good'

    def _final(self,state:State,config:RunnableConfig,store:BaseStore):
        return {'messages':[AIMessage(content=state['answer'])]}
