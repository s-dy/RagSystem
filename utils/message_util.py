from typing import List

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import (
    AnyMessage,
    HumanMessage,
    AIMessage,
    SystemMessage,
    RemoveMessage,
)
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

from src.observability.logger import get_logger

logger = get_logger(__name__)

# ============================================================
# Prompt 模板
# ============================================================

conversation_compress_prompt = """你是一个对话摘要专家。请将以下对话历史压缩为简洁的摘要。

对话历史：
{conversation_history}

要求：
1. 保留所有关键事实、用户确认的约束条件、重要结论
2. 去除寒暄、重复、冗余表述
3. 保持时间顺序
4. 使用简洁的陈述句
5. 摘要长度控制在原文的 30% 以内"""

incremental_summary_prompt = """已有对话摘要：
{existing_summary}

新增对话内容：
{new_messages}

⚠️ 以下关键事实必须在摘要中完整保留，不可省略或改写：
{anchor_facts}

请将新增内容整合到已有摘要中，生成更新后的完整摘要。
保留所有关键信息，去除冗余。确保上述关键事实在摘要中原样保留。"""

context_summary_prompt = (
    "请用 2-3 句话概括以下对话的要点，只保留关键事实和结论：\n\n{history}"
)


# ============================================================
# Token 估算
# ============================================================


def estimate_token_count(text: str) -> int:
    """估算文本的 token 数（混合中英文场景，字符数 * 0.6 ≈ token 数）"""
    if not text:
        return 0
    return int(len(text) * 0.6)


def estimate_messages_tokens(messages: List[AnyMessage]) -> int:
    """估算消息列表的总 token 数"""
    total = 0
    for msg in messages:
        if hasattr(msg, "content") and isinstance(msg.content, str):
            total += estimate_token_count(msg.content)
    return total


def get_last_user_msg(messages: List[AnyMessage]):
    user_messages = [msg for msg in messages if isinstance(msg, HumanMessage)]
    if not user_messages:
        return None
    last_msg = user_messages[-1].content
    msg_preview = last_msg[:100] + "..." if len(last_msg) > 100 else last_msg
    logger.debug(f"[MessageUtil] 获取最后用户消息: {msg_preview}")
    return last_msg


def get_conversation_context(messages: List[AnyMessage], num_messages: int = 3) -> str:
    """获取对话上下文，用于多轮对话理解"""
    if not messages:
        return ""

    # 获取最近的几条消息作为上下文
    recent_messages = (
        messages[-num_messages:] if len(messages) >= num_messages else messages
    )
    context_parts = []

    for msg in recent_messages:
        if isinstance(msg, HumanMessage):
            role = "用户: "
        elif isinstance(msg, AIMessage):
            role = "助手: "
        else:
            role = "系统: "
        context_parts.append(f"{role}{msg.content}")

    context = "\n".join(context_parts)
    context_preview = context[:200] + "..." if len(context) > 200 else context
    logger.debug(f"[MessageUtil] 对话上下文: {context_preview}")
    return context


# ============================================================
# 策略 1：跨轮对话历史压缩
# ============================================================


async def compress_conversation_history(
    llm: BaseChatModel,
    messages: List[AnyMessage],
    keep_recent: int = 3,
    max_compress_tokens: int = 6000,
) -> List[AnyMessage]:
    """跨轮对话历史压缩：将旧消息压缩为 SystemMessage 摘要，保留最近 K 轮原始消息

    Args:
        llm: 语言模型实例
        messages: 完整的消息列表
        keep_recent: 保留最近的轮数（1 轮 = 1 条 Human + 1 条 AI）
        max_compress_tokens: 单次压缩的最大 token 数，超过则分批压缩

    Returns:
        压缩后的消息列表：[SystemMessage(摘要)] + 最近 K 轮原始消息
    """
    if not messages:
        return messages

    # 保留最近 keep_recent 轮（每轮 2 条消息）
    keep_count = keep_recent * 2
    if len(messages) <= keep_count:
        return messages

    recent_messages = messages[-keep_count:]
    old_messages = messages[:-keep_count]

    # 将旧消息拼接为文本
    old_texts = [
        f"{'用户' if isinstance(m, HumanMessage) else '助手'}: {m.content}"
        for m in old_messages
        if isinstance(m, (HumanMessage, AIMessage))
    ]

    if not old_texts:
        return messages

    history_text = "\n".join(old_texts)

    # 防护：如果旧消息总量超出单次压缩的安全上限，分批压缩
    if estimate_token_count(history_text) > max_compress_tokens:
        summary = await _batch_compress(llm, old_texts, max_compress_tokens)
    else:
        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", conversation_compress_prompt),
                ("human", "请压缩上述对话历史。"),
            ]
        )
        chain = prompt | llm | StrOutputParser()
        summary = await chain.ainvoke({"conversation_history": history_text})

    summary_message = SystemMessage(content=f"[对话历史摘要]\n{summary}")
    logger.info(
        f"[MessageUtil] 对话历史压缩完成: original_count={len(old_messages)}, summary_length={len(summary)}"
    )
    return [summary_message] + recent_messages


async def _batch_compress(
    llm: BaseChatModel,
    texts: List[str],
    max_tokens_per_batch: int,
) -> str:
    """分批压缩：将文本列表按 token 预算分段，逐段压缩后合并为最终摘要

    流程：
    1. 将 texts 按 max_tokens_per_batch 分成若干批次
    2. 每批次独立调用 LLM 压缩为段摘要
    3. 将所有段摘要合并，再做一次最终压缩
    """
    batches = []
    current_batch = []
    current_tokens = 0
    for text in texts:
        text_tokens = estimate_token_count(text)
        if current_tokens + text_tokens > max_tokens_per_batch and current_batch:
            batches.append("\n".join(current_batch))
            current_batch = [text]
            current_tokens = text_tokens
        else:
            current_batch.append(text)
            current_tokens += text_tokens
    if current_batch:
        batches.append("\n".join(current_batch))

    prompt = ChatPromptTemplate.from_messages(
        [("system", conversation_compress_prompt), ("human", "请压缩上述对话历史。")]
    )
    chain = prompt | llm | StrOutputParser()

    batch_summaries = []
    for batch_text in batches:
        batch_summary = await chain.ainvoke({"conversation_history": batch_text})
        batch_summaries.append(batch_summary)

    if len(batch_summaries) == 1:
        return batch_summaries[0]

    # 多个批次：合并段摘要，做最终压缩
    merged_text = "\n---\n".join(batch_summaries)
    final_summary = await chain.ainvoke({"conversation_history": merged_text})
    return final_summary


# ============================================================
# 策略 2：对话上下文窗口自适应
# ============================================================


async def get_conversation_context_adaptive(
    messages: List[AnyMessage],
    llm: BaseChatModel,
    max_context_tokens: int = 2000,
    min_messages: int = 2,
    max_messages: int = 10,
) -> str:
    """自适应对话上下文提取：预算内保留原文，超出部分摘要补充

    两层策略：
    1. 从最新消息向前遍历，在 token 预算内尽可能多地保留原始消息
    2. 对超出预算的更早消息，使用 LLM 生成摘要作为补充上下文
    """
    if not messages:
        return ""

    char_budget = int(max_context_tokens / 0.6)

    # 第一层：从最新消息向前，在预算内保留原文
    retained_parts = []
    total_chars = 0
    cutoff_index = len(messages)

    for i, msg in enumerate(reversed(messages)):
        if not isinstance(msg, (HumanMessage, AIMessage)):
            continue
        role = "用户: " if isinstance(msg, HumanMessage) else "助手: "
        text = f"{role}{msg.content}"

        if (
            total_chars + len(text) > char_budget
            and len(retained_parts) >= min_messages
        ):
            cutoff_index = len(messages) - i
            break
        if len(retained_parts) >= max_messages:
            cutoff_index = len(messages) - i
            break

        retained_parts.append(text)
        total_chars += len(text)

    retained_parts.reverse()

    # 第二层：对预算外的更早消息，生成摘要补充
    earlier_messages = [
        msg
        for msg in messages[:cutoff_index]
        if isinstance(msg, (HumanMessage, AIMessage))
    ]

    summary_prefix = ""
    if earlier_messages:
        # 检查是否有已压缩的 SystemMessage 摘要（策略 1 生成的），直接复用
        existing_summaries = [
            msg.content
            for msg in messages[:cutoff_index]
            if isinstance(msg, SystemMessage) and "[对话历史摘要]" in msg.content
        ]
        if existing_summaries:
            summary_prefix = existing_summaries[-1] + "\n\n"
        else:
            earlier_text = "\n".join(
                f"{'用户' if isinstance(m, HumanMessage) else '助手'}: {m.content}"
                for m in earlier_messages
            )
            # 防护：如果早期消息本身也很长，截取最近的部分做摘要
            max_summary_chars = int(4000 / 0.6)
            if len(earlier_text) > max_summary_chars:
                earlier_text = earlier_text[-max_summary_chars:]

            prompt = ChatPromptTemplate.from_template(context_summary_prompt)
            chain = prompt | llm | StrOutputParser()
            summary = await chain.ainvoke({"history": earlier_text})
            summary_prefix = f"[早期对话摘要] {summary}\n\n"

    context = summary_prefix + "\n".join(retained_parts)
    logger.info(
        f"[MessageUtil] 自适应上下文提取完成: retained_count={len(retained_parts)}, has_summary={bool(summary_prefix)}"
    )
    return context


# ============================================================
# 策略 6：渐进式摘要（Incremental Summarization）
# ============================================================


async def incremental_summarize_with_anchors(
    llm: BaseChatModel,
    existing_summary: str,
    new_messages: List[AnyMessage],
    anchor_facts: List[str] = None,
) -> str:
    """带锚点防护的增量摘要，防止关键信息在多次迭代中漂移丢失

    Args:
        llm: 语言模型实例
        existing_summary: 已有的对话摘要
        new_messages: 新增的消息列表
        anchor_facts: 不可省略的关键事实列表（来自 HIGH 重要性消息）

    Returns:
        更新后的完整摘要
    """
    new_text = "\n".join(
        f"{'用户' if isinstance(m, HumanMessage) else '助手'}: {m.content}"
        for m in new_messages
        if isinstance(m, (HumanMessage, AIMessage))
    )

    if not new_text.strip():
        return existing_summary or ""

    anchors_text = (
        "\n".join(f"- {fact}" for fact in anchor_facts) if anchor_facts else "（无）"
    )

    prompt = ChatPromptTemplate.from_template(incremental_summary_prompt)
    chain = prompt | llm | StrOutputParser()
    updated_summary = await chain.ainvoke(
        {
            "existing_summary": existing_summary or "（无）",
            "new_messages": new_text,
            "anchor_facts": anchors_text,
        }
    )

    logger.info(
        f"[MessageUtil] 增量摘要完成: new_messages_count={len(new_messages)}, anchor_facts_count={len(anchor_facts) if anchor_facts else 0}, summary_length={len(updated_summary)}"
    )
    return updated_summary


def should_trigger_incremental_summary(
    messages: List[AnyMessage], interval: int = 5
) -> bool:
    """判断是否应触发渐进式摘要（每隔 interval 轮触发一次）"""
    human_count = sum(1 for m in messages if isinstance(m, HumanMessage))
    return human_count > 0 and human_count % interval == 0


def build_remove_and_replace_messages(
    original_messages: List[AnyMessage],
    compressed_messages: List[AnyMessage],
) -> List:
    """构建 RemoveMessage 删除指令 + 压缩后的新消息列表

    LangGraph 的 MessagesState 默认使用 add_messages（追加语义），
    不能直接返回 {"messages": compressed}，否则会追加而非替换。
    需要使用 RemoveMessage 机制：
    1. 对所有旧消息生成 RemoveMessage(id=msg.id) 删除指令
    2. 将压缩后的消息作为新消息追加
    """
    remove_ops = [
        RemoveMessage(id=msg.id)
        for msg in original_messages
        if hasattr(msg, "id") and msg.id
    ]
    return remove_ops + compressed_messages
