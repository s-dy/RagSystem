"""生成节点 Mixin

包含：
- _generate_current_answer: 生成当前查询的答案
- _synthesize: 合并多跳子问题答案
- _final: 最终输出节点
- _run_eval: RAG 评估
"""

import re
from typing import List

from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableConfig
from langgraph.store.base import BaseStore

from src.observability.logger import get_logger
from utils.message_util import get_conversation_context_adaptive
from .generate import (
    generate_answer_for_query,
    generate_direct_chat_answer,
    synthesize_final_subs,
    compress_reasoning_context,
)

logger = get_logger(__name__)


class GenerateNodeMixin:
    """生成节点方法集合，通过 Mixin 注入到 Graph 类"""

    @staticmethod
    def _extract_cited_source_ids(search_content: str) -> List[int]:
        """从 search_content 中提取所有来源编号。

        search_content 格式约定为：每段来源以 "[编号]" 开头，如 "[1] 内容..."。
        提取所有出现的编号，返回去重后的有序列表。

        Args:
            search_content: 检索内容整合字符串（带来源编号）

        Returns:
            来源编号列表，如 [1, 2, 3]
        """
        if not search_content:
            return []
        source_ids = [int(match) for match in re.findall(r"\[(\d+)\]", search_content)]
        return sorted(set(source_ids))

    @staticmethod
    def _build_citation_summary(reasoning_steps: List[dict]) -> str:
        """根据所有子问题的推理步骤，构建引用溯源摘要字符串。

        遍历每个 sub_answer 步骤，汇总其 cited_sources，生成如下格式的摘要：
            📎 **引用溯源**：
            - 子问题1 → 来源 [1][3]
            - 子问题2 → 来源 [2]

        Args:
            reasoning_steps: State 中的 reasoning_steps 列表

        Returns:
            格式化的引用溯源字符串，若无引用则返回空字符串
        """
        citation_lines = []
        for step in reasoning_steps:
            if step.get("type") != "sub_answer":
                continue
            cited = step.get("cited_sources", [])
            if not cited:
                continue
            sub_q = step.get("sub_question", "")
            display_q = sub_q[:40] + "..." if len(sub_q) > 40 else sub_q
            source_refs = "".join(f"[{sid}]" for sid in cited)
            citation_lines.append(f"- {display_q} → 来源 {source_refs}")

        if not citation_lines:
            return ""
        return "📎 **引用溯源**：\n" + "\n".join(citation_lines)

    async def __generate_current_answer(
        self, state, config: RunnableConfig, store: BaseStore
    ) -> dict:
        """生成当前查询的答案（支持流式输出和置信度）"""
        current_q = state.get("current_sub_question") or state["original_query"]
        logger.info(f"[GenerateNode] 开始生成答案: query={current_q[:50]}...")
        conversation_context = await get_conversation_context_adaptive(
            state["messages"],
            self.llm,
            max_context_tokens=self.config.max_context_tokens,
        )
        docs = state["search_content"] or "未找到相关信息"

        # 判断是否为最终答案
        if state["task_characteristics"].is_multi_hop:
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
                logger.info(
                    f"[GenerateNode] 最终答案生成完成: confidence={confidence_level}, avg_score={avg_score:.2f}, sources={source_count}"
                )

        logger.debug(
            f"[GenerateNode] 答案生成完成: is_final={is_final}, answer_length={len(answer)}"
        )

        # 更新状态
        remaining_subs = state["sub_questions"][1:] if state["sub_questions"] else []
        new_reasoning_context = (
            state.get("reasoning_context", "")
            + f"问题：{current_q}\n答案：{answer}\n\n"
        )

        # 压缩过长的 reasoning_context
        new_reasoning_context = await compress_reasoning_context(
            self.llm, new_reasoning_context
        )

        # 记录推理步骤，同时追踪当前子问题引用的来源文档编号
        existing_steps = state.get("reasoning_steps", [])
        cited_sources = self._extract_cited_source_ids(state.get("search_content", ""))
        sub_answer_step = {
            "type": "sub_answer",
            "sub_question": current_q,
            "answer": answer,
            "is_final": is_final,
            "cited_sources": cited_sources,  # 本步骤引用的来源编号列表，如 [1, 2, 3]
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
            return {
                "answer": answer,
                "sub_questions": [],
                "reasoning_steps": existing_steps + [sub_answer_step],
            }

        return {
            "sub_questions": remaining_subs,
            "messages": [AIMessage(content=f"📝 子问题 {current_q}\n💡 {answer}")],
            "reasoning_context": new_reasoning_context,
            "reasoning_steps": existing_steps + [sub_answer_step],
        }

    async def __synthesize(
        self, state, config: RunnableConfig, store: BaseStore
    ) -> dict:
        """合并多跳子问题答案，并在最终答案中附加完整的引用溯源信息"""
        if not state["task_characteristics"].is_multi_hop:
            return {}
        logger.info("[GenerateNode] 开始综合多跳答案")
        question = state["original_query"]
        reasoning_ctx = state.get("reasoning_context", "")
        answer = await synthesize_final_subs(
            self.llm,
            query=question,
            reasoning_context=reasoning_ctx,
        )

        # 汇总所有子问题的引用来源，生成溯源摘要附加到答案末尾
        citation_summary = self._build_citation_summary(
            state.get("reasoning_steps", [])
        )
        if citation_summary:
            answer = answer + "\n\n" + citation_summary

        logger.info(f"[GenerateNode] 多跳答案综合完成: answer_length={len(answer)}")
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
            logger.info(f"[GenerateNode] 直接对话模式: query={question[:50]}...")
            conversation_context = await get_conversation_context_adaptive(
                state["messages"],
                self.llm,
                max_context_tokens=self.config.max_context_tokens,
            )
            answer = await generate_direct_chat_answer(
                self.llm,
                query=question,
                conversation_context=conversation_context,
            )
        else:
            # 需要检索：使用已有的 answer
            answer = state.get("answer") or ""
            logger.info(f"[GenerateNode] 检索模式最终输出: answer_length={len(answer)}")

        # 统一存储最终问答对
        if self.memory_manager and user_id and question and answer:
            await self.memory_manager.save_conversation_memory(
                user_id=user_id,
                thread_id=thread_id,
                memory_type="qa_pair",
                content={"question": question, "answer": answer},
            )

        if self.config.enable_eval and state.get("need_retrieval"):
            await self.__run_eval(state)
        return {"messages": [AIMessage(content=answer)]}

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
            logger.info(
                f"[GenerateNode] RAG评估完成: faithfulness={scores.faithfulness:.3f}, answer_relevancy={scores.answer_relevancy:.3f}, context_relevance={scores.context_relevance:.3f}"
            )
        except Exception as eval_error:
            logger.warning(f"[GenerateNode] RAG评估执行失败: {eval_error}")
