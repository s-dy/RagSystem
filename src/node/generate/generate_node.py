"""生成节点 Mixin

包含：
- _generate_current_answer: 生成当前查询的答案
- _synthesize: 合并多跳子问题答案
- _final: 最终输出节点
- _run_eval: RAG 评估
"""

from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableConfig
from langgraph.store.base import BaseStore

from .generate import (
    generate_answer_for_query,
    generate_direct_chat_answer,
    synthesize_final_subs,
    compress_reasoning_context,
)
from src.observability.logger import monitor_task_status
from utils.message_util import get_conversation_context_adaptive


class GenerateNodeMixin:
    """生成节点方法集合，通过 Mixin 注入到 Graph 类"""

    async def __generate_current_answer(self, state, config: RunnableConfig, store: BaseStore) -> dict:
        """生成当前查询的答案（支持流式输出和置信度）"""
        current_q = state.get("current_sub_question") or state["original_query"]
        conversation_context = await get_conversation_context_adaptive(
            state['messages'], self.llm, max_context_tokens=self.config.max_context_tokens,
        )
        docs = state["search_content"] or "未找到相关信息"

        # 判断是否为最终答案
        if state['task_characteristics'].is_multi_hop:
            is_final = len(state["sub_questions"]) == 0
        else:
            is_final = len(state["sub_questions"]) == 1

        # 生成答案
        answer = await generate_answer_for_query(
            self.llm,
            query=current_q,
            docs_content=docs,
            conversation_context=conversation_context,
            reasoning_context=state.get("reasoning_context", ""),
            is_final=is_final,
        )

        # 最终答案附加置信度信息
        if is_final:
            scores = state.get("retrieval_scores", [])
            if scores:
                avg_score = sum(scores) / len(scores)
                source_count = len(scores)
                if avg_score >= 0.8:
                    confidence_level = "高"
                elif avg_score >= 0.5:
                    confidence_level = "中"
                else:
                    confidence_level = "低"
                answer += f"\n\n📊 置信度：{confidence_level}（{avg_score:.2f}）| 参考来源：{source_count} 篇文档"

        monitor_task_status('current answer', answer)

        # 更新状态
        remaining_subs = state["sub_questions"][1:] if state["sub_questions"] else []
        new_reasoning_context = state.get("reasoning_context", "") + f"问题：{current_q}\n答案：{answer}\n\n"

        # 压缩过长的 reasoning_context
        new_reasoning_context = await compress_reasoning_context(self.llm, new_reasoning_context)

        # 记录推理步骤
        existing_steps = state.get("reasoning_steps", [])
        sub_answer_step = {
            "type": "sub_answer",
            "sub_question": current_q,
            "answer": answer,
            "is_final": is_final,
        }

        thread_id = config["configurable"].get("thread_id", "default")
        user_id = config["configurable"].get("user_id", "default")
        # 如果不是最终答案，记录子问题答案
        if not is_final and self.memory_manager and user_id:
            await self.memory_manager.save_conversation_memory(
                user_id=user_id,
                thread_id=thread_id,
                memory_type="qa_pair",
                content={"question": current_q, "answer": answer},
            )

        # 如果是最终答案，直接设置 answer 字段
        if is_final:
            return {"answer": answer, "sub_questions": [], "reasoning_steps": existing_steps + [sub_answer_step]}

        return {
            "sub_questions": remaining_subs,
            "messages": [AIMessage(content=f"📝 子问题 {current_q}\n💡 {answer}")],
            "reasoning_context": new_reasoning_context,
            "reasoning_steps": existing_steps + [sub_answer_step],
        }

    async def __synthesize(self, state, config: RunnableConfig, store: BaseStore) -> dict:
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

        return {"answer": answer}

    async def __final(self, state, config: RunnableConfig):
        """最终输出节点：根据 need_retrieval 选择不同策略生成回复

        - need_retrieval=False: 调用 LLM 用对话 prompt 直接回复（支持流式 token 输出）
        - need_retrieval=True: 使用已有的 answer（来自 generate_current_answer/synthesize）
        """
        thread_id = config["configurable"].get("thread_id", "default")
        user_id = config["configurable"].get("user_id", "default")
        question = state.get("original_query") or ""

        if not state.get("need_retrieval"):
            # 不需要检索：调用 LLM 直接生成对话回复（token 会被 messages 模式捕获并流式展示）
            conversation_context = await get_conversation_context_adaptive(
                state['messages'], self.llm,  max_context_tokens=self.config.max_context_tokens,
            )
            answer = await generate_direct_chat_answer(
                self.llm,
                query=question,
                conversation_context=conversation_context,
            )
        else:
            # 需要检索：使用已有的 answer
            answer = state.get("answer") or ""

        # 统一存储最终问答对
        if self.memory_manager and user_id and question and answer:
            await self.memory_manager.save_conversation_memory(
                user_id=user_id,
                thread_id=thread_id,
                memory_type="qa_pair",
                content={"question": question, "answer": answer},
            )

        if self.config.enable_eval and state.get('need_retrieval'):
            await self.__run_eval(state)
        return {'messages': [AIMessage(content=answer)]}

    async def __run_eval(self, state):
        """运行 RAG 评估并记录结果"""
        try:
            from src.eval.ragas_eval import RagEvaluator, EvalSample

            if self._evaluator is None:
                self._evaluator = RagEvaluator()

            sample = EvalSample(
                user_input=state.get("original_query", ""),
                response=state.get("answer", ""),
                retrieved_contexts=state.get("retrieved_documents", []),
            )
            scores = await self._evaluator.evaluate_sample(sample)
            monitor_task_status("RAG 评估结果", {
                "faithfulness": scores.faithfulness,
                "answer_relevancy": scores.answer_relevancy,
                "context_relevance": scores.context_relevance,
                "context_recall": scores.context_recall,
            })
        except Exception as eval_error:
            monitor_task_status(f"RAG 评估执行失败: {eval_error}", level="WARNING")
