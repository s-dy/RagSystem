import os
import re
import time
from typing import List, Dict, Optional

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from langgraph.store.base import BaseStore
from langgraph.store.postgres import AsyncPostgresStore
from langgraph.graph import MessagesState, START, END, StateGraph

from config.Config import POSTGRESQL_URL, QueryEnhancementConfig
from src.node.retrieval.fusion_retrieve import FusionRetrieve
from src.node.expansion.query_enhancer import QueryEnhancer
from src.monitoring.langfuse_monitor import langfuse_handler
from src.monitoring.logger import monitor_task_status
from src.services.llm.models import get_qwen_model
from utils.async_task import async_run
from utils.message_util import get_last_user_msg


class State(MessagesState):
    original_query: str
    sub_questions: List[str]  # 待解决的子问题队列（顺序执行）
    resolved_sub_answers: Dict[str, str]  # 已解决的子问题及其答案
    current_sub_question: Optional[str]  # 当前正在处理的子问题
    reasoning_context: str  # 已解决的上下文（用于注入后续步骤）
    search_content: str  # 当前检索内容


class MultiStepGraph:
    def __init__(self, llm: BaseChatModel):
        self.llm = llm

    def _init_graph(self):
        graph = StateGraph(State)

        # 节点定义
        graph.add_node("decompose_question", self._decompose_question)
        graph.add_node("prepare_current_sub", self._prepare_current_sub)
        graph.add_node("retrieve", self._retrieve)
        graph.add_node("generate_response", self._generate_response)
        graph.add_node("synthesize_final", self._synthesize_final_answer)

        # 边定义
        graph.add_edge(START, "decompose_question")
        graph.add_edge("decompose_question", "prepare_current_sub")
        graph.add_edge("prepare_current_sub", "retrieve")
        graph.add_edge("retrieve", "generate_response")
        graph.add_conditional_edges(
            "generate_response",
            self._check_more_subs,
            {"continue": "prepare_current_sub", "done": "synthesize_final"}
        )
        graph.add_edge("synthesize_final", END)

        return graph

    async def start(self, *args, **kwargs):
        workflow = self._init_graph()
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

    async def _decompose_question(self, state: State, config: RunnableConfig, store: BaseStore) -> dict:
        monitor_task_status("---DECOMPOSING QUESTION---")
        original_query = get_last_user_msg(state["messages"])
        try:
            sub_questions = await QueryEnhancer(self.llm,QueryEnhancementConfig(decompose_to_subquestions=True)).enhance(original_query)
            if not isinstance(sub_questions, list):
                raise ValueError("Not a list")
        except Exception as e:
            monitor_task_status(f"分解失败，回退到单问题: {e}")
            sub_questions = [original_query]

        thread_id = config["configurable"].get("thread_id", "default")
        user_id = config["configurable"].get("user_id", "default")
        if thread_id and user_id:
            await store.aput(
                (user_id, thread_id),
                key=f"decompose_{int(time.time() * 1000)}",
                value={"original": original_query, "sub_questions": sub_questions}
            )

        return {
            "original_query": original_query,
            "sub_questions": sub_questions,
            "resolved_sub_answers": {},
        }

    def _prepare_current_sub(self, state: State) -> dict:
        if not state["sub_questions"]:
            return {"current_sub_question": None}

        current = state["sub_questions"][0]
        # 构建推理上下文
        ctx_lines = []
        for q, a in state["resolved_sub_answers"].items():
            ctx_lines.append(f"问题：{q}\n答案：{a}")
        reasoning_context = "\n\n".join(ctx_lines) if ctx_lines else ""
        monitor_task_status(f'current_sub_question ==> {current},reasoning_context ==> {reasoning_context}')

        return {
            "current_sub_question": current,
            "reasoning_context": reasoning_context
        }

    async def _retrieve_internal(self, queries: List[str], collection_name: str = "cmrc_dataset"):
        """简化内部检索，接受 query 列表"""
        doc_result = []
        monitor_task_status(f"开始内部检索: {queries}")
        search_model = FusionRetrieve()
        try:
            results = await search_model.search_queries(queries, collection_name=collection_name)
            for doc_list in results or []:
                if doc_list:
                    doc_result.extend(doc_list)
        except Exception as e:
            monitor_task_status(f"检索出错: {str(e)}")
        return doc_result

    async def _retrieve(self, state: State, config: RunnableConfig, store: BaseStore) -> dict:
        current_q = state["current_sub_question"]
        reasoning_ctx = state.get("reasoning_context", "")

        # 用上下文增强当前子问题
        if reasoning_ctx.strip():
            enhance_prompt = ChatPromptTemplate.from_template("""
            基于以下已知信息，重写问题使其更明确、可检索：

            已知上下文：
            {context}

            问题：{question}

            重写后的问题（只返回问题，不要解释）：
            """)
            chain = enhance_prompt | self.llm | StrOutputParser()
            enhanced_query = await chain.ainvoke({"context": reasoning_ctx, "question": current_q})
        else:
            enhanced_query = current_q

        # 执行检索
        internal_docs = await self._retrieve_internal([enhanced_query])
        external_docs = []  # 可扩展

        # 去重
        seen = set()
        unique_docs = []
        for doc in internal_docs + external_docs:
            normalized = re.sub(r'[^\w\s\u4e00-\u9fa5]', '', doc.lower().strip())
            if normalized and normalized not in seen:
                seen.add(normalized)
                unique_docs.append(doc)

        content = "\n\n".join(unique_docs) if unique_docs else ""

        thread_id = config["configurable"].get("thread_id", "default")
        user_id = config["configurable"].get("user_id", "default")
        if thread_id and user_id:
            await store.aput(
                (user_id, thread_id),
                key=f"retrieve_{int(time.time() * 1000)}",
                value={"query": current_q, "enhanced": enhanced_query, "docs_count": len(unique_docs)}
            )

        return {
            "search_content": content,
        }

    async def _generate_response(self, state: State, config: RunnableConfig, store: BaseStore) -> dict:
        current_q = state["current_sub_question"]
        docs = state["search_content"] or "未找到相关信息"
        reasoning_ctx = state.get("reasoning_context", "")

        prompt = ChatPromptTemplate.from_messages([
            ("system", """
            你是一个精准的问答助手。请基于以下信息回答问题。

            已知上下文（之前已解答的问题）：
            {reasoning_context}

            检索到的信息：
            {content}

            规则：
            - 只回答当前问题，不要解释过程。
            - 若信息不足，回答“未知”。
            - 不要编造信息。
            """),
            ("human", "{query}")
        ])
        chain = prompt | self.llm | StrOutputParser()
        answer = await chain.ainvoke({
            "query": current_q,
            "content": docs,
            "reasoning_context": reasoning_ctx
        })
        monitor_task_status('generating response',answer)

        # 更新状态
        new_resolved = state["resolved_sub_answers"].copy()
        new_resolved[current_q] = answer

        remaining_subs = state["sub_questions"][1:]  # 移除已处理的第一个

        thread_id = config["configurable"].get("thread_id", "default")
        user_id = config["configurable"].get("user_id", "default")
        if thread_id and user_id:
            await store.aput(
                (user_id, thread_id),
                key=f"sub_answer_{int(time.time() * 1000)}",
                value={"question": current_q, "answer": answer}
            )

        return {
            "resolved_sub_answers": new_resolved,
            "sub_questions": remaining_subs,
            "messages": [AIMessage(content=f"[中间] {current_q} → {answer}")]
        }

    def _check_more_subs(self, state: State) -> str:
        return "continue" if state["sub_questions"] else "done"

    async def _synthesize_final_answer(self, state: State, config: RunnableConfig, store: BaseStore) -> dict:
        original = state["original_query"]
        all_answers = "\n".join([
            f"问题：{q}\n答案：{a}\n" for q, a in state["resolved_sub_answers"].items()
        ])

        prompt = ChatPromptTemplate.from_messages([
            ("system", """
            基于以下子问题解答，回答原始问题。请整合信息，给出简洁、准确、完整的最终答案。

            子问题解答记录：
            {answers}
            """),
            ("human", "{original_query}")
        ])
        chain = prompt | self.llm | StrOutputParser()
        final_ans = await chain.ainvoke({
            "original_query": original,
            "answers": all_answers
        })
        monitor_task_status('final_answer', final_ans)

        thread_id = config["configurable"].get("thread_id", "default")
        user_id = config["configurable"].get("user_id", "default")
        if thread_id and user_id:
            await store.aput(
                (user_id, thread_id),
                key=f"final_answer_{int(time.time() * 1000)}",
                value={"original": original, "final_answer": final_ans}
            )

        return {"messages": [AIMessage(content=final_ans)]}

if __name__ == '__main__':
    graph = MultiStepGraph(get_qwen_model())
    config: RunnableConfig = {'configurable': {'thread_id': '11111111', 'user_id': '11111111'},'recursion_limit': 15}
    inputs = {"messages": [{"role": "user", "content": "《东京战争》是由谁发行的？他可以供玩家选择的战场有哪几个？"}]}
    response = async_run(graph.start(inputs,config=config))
    msg = response['messages'][-1]
    if isinstance(msg, AIMessage):
        llm_answer = response['messages'][-1].content
    elif isinstance(msg, dict):
        llm_answer = msg['messages'][-1]['content']
    else:
        raise NotImplementedError
    print(llm_answer)