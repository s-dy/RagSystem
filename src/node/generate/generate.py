from typing import AsyncGenerator, List, Optional

from langchain_core.language_models import BaseChatModel
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

from config import RagSystemConfig

final_system_prompt = """
你是一个专业、严谨的智能助手，请根据以下检索到的信息和对话历史来回答用户问题。

检索到的信息（每条信息附带来源编号）：
{content}

对话历史：
{conversation_context}

规则：
1. 基于检索到的信息回答问题，用简洁、准确的语言表述。
2. 在引用具体信息时，以自然的方式在句末标注来源编号，如 [1]、[2]。
3. 如果检索到的信息与问题无关或信息不足，但对话历史中有相关信息，请参考已有对话进行回答。
4. 如果检索到的信息与对话历史都无相关信息，请只回答："根据现有资料，无法回答该问题。"
5. 严格基于检索内容和对话历史回答，不要编造信息。
6. 回答要连贯，符合对话上下文。
7. 如果答案来自多个来源，整合信息，给出简洁、准确、完整的最终答案。
8. 在答案末尾另起一行，以"📎 参考来源："开头列出引用的来源编号及其标题。
"""

sub_system_prompt = """
你是一个精准的问答助手。请基于以下信息回答问题。

已知上下文（之前已解答的问题）：
{reasoning_context}

检索到的信息：
{content}

规则：
- 只回答当前问题，不要解释过程。
- 若信息不足，回答"未知"。
- 不要编造信息。
"""

synthesize_system_prompt = """
基于以下子问题解答，回答原始问题。请整合信息，给出简洁、准确、完整的最终答案。

子问题解答记录：
{reasoning_context}
"""

compress_system_prompt = """
请将以下子问题解答记录压缩为简洁的摘要，保留所有关键事实和结论。

{reasoning_context}

输出要求：
- 保留每个子问题的核心答案
- 去除冗余描述
- 保持信息的准确性和完整性
"""

direct_chat_system_prompt = """
你是一个友好、专业的智能助手。请根据对话历史和用户的当前问题，直接给出回复。

对话历史：
{conversation_context}

规则：
1. 对于问候、闲聊等日常对话，给出自然、友好的回复。
2. 对于不需要外部知识就能回答的问题（如常识、推理、建议等），直接回答。
3. 回答要连贯，符合对话上下文。
4. 不要编造具体的事实或数据。
"""


def _build_answer_chain(llm: BaseChatModel, is_final: bool):
    """构建答案生成的 chain，供流式和非流式共用"""
    system_prompt = final_system_prompt if is_final else sub_system_prompt
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{query}")
    ])
    return prompt | llm.with_config(tags=["stream_to_user"]) | StrOutputParser()


def _build_answer_params(
    query: str,
    docs_content: str,
    conversation_context: str,
    reasoning_context: str,
    is_final: bool,
) -> dict:
    """构建答案生成的参数字典"""
    return {
        "query": query,
        "content": docs_content,
        "conversation_context": conversation_context,
        "reasoning_context": conversation_context if is_final else reasoning_context,
    }


async def generate_answer_for_query(
        llm: BaseChatModel,
        query: str,
        docs_content: str,
        conversation_context: str = "",
        reasoning_context: str = "",
        is_final: bool = False,
) -> str:
    """生成答案（非流式）"""
    chain = _build_answer_chain(llm, is_final)
    params = _build_answer_params(query, docs_content, conversation_context, reasoning_context, is_final)
    return await chain.ainvoke(params)


async def generate_answer_for_query_stream(
        llm: BaseChatModel,
        query: str,
        docs_content: str,
        conversation_context: str = "",
        reasoning_context: str = "",
        is_final: bool = False,
) -> AsyncGenerator[str, None]:
    """流式生成答案，逐 token 返回（供 graph 外层 astream_events 使用）"""
    chain = _build_answer_chain(llm, is_final)
    params = _build_answer_params(query, docs_content, conversation_context, reasoning_context, is_final)
    async for chunk in chain.astream(params):
        yield chunk


async def synthesize_final_subs(llm, query, reasoning_context):
    """多跳问题最终综合答案"""
    prompt = ChatPromptTemplate.from_messages([
        ("system", synthesize_system_prompt),
        ("human", "{query}")
    ])
    chain = prompt | llm.with_config(tags=["stream_to_user"]) | StrOutputParser()
    return await chain.ainvoke({
        "reasoning_context": reasoning_context,
        "query": query,
    })


async def generate_direct_chat_answer(llm: BaseChatModel, query: str, conversation_context: str = "") -> str:
    """直接对话回复（不需要检索的场景），供 final 节点调用"""
    prompt = ChatPromptTemplate.from_messages([
        ("system", direct_chat_system_prompt),
        ("human", "{query}")
    ])
    chain = prompt | llm.with_config(tags=["stream_to_user"]) | StrOutputParser()
    return await chain.ainvoke({
        "query": query,
        "conversation_context": conversation_context,
    })


# ── 多模态生成 ──────────────────────────────────────────────────────────────

multimodal_system_prompt = """
你是一个专业、严谨的多模态智能助手，请根据以下检索到的文字信息和相关图片来回答用户问题。

检索到的文字信息（每条信息附带来源编号）：
{content}

对话历史：
{conversation_context}

规则：
1. 综合文字信息和图片内容回答问题，用简洁、准确的语言表述。
2. 在引用具体文字信息时，以自然的方式在句末标注来源编号，如 [1]、[2]。
3. 如果图片中有与问题相关的内容（图表、示意图、表格截图等），请明确描述图片中的关键信息。
4. 如果文字和图片信息都不足，请回答："根据现有资料，无法回答该问题。"
5. 严格基于检索内容回答，不要编造信息。
6. 在答案末尾另起一行，以"📎 参考来源："开头列出引用的来源编号及其标题。
"""


def _build_multimodal_message_content(
    query: str,
    docs_content: str,
    conversation_context: str,
    image_base64_list: List[str],
) -> list:
    """构建多模态消息的 content 列表（文字 + 图片交织）。

    将系统 prompt 中的变量填充后，与图片 base64 一起组装为
    OpenAI 多模态消息格式（content 为列表）。

    Args:
        query: 用户查询
        docs_content: 检索到的文字内容
        conversation_context: 对话历史
        image_base64_list: 图片 base64 编码列表

    Returns:
        OpenAI 多模态 content 列表
    """
    filled_system = multimodal_system_prompt.format(
        content=docs_content,
        conversation_context=conversation_context,
    )
    content_parts = [{"type": "text", "text": filled_system + f"\n\n用户问题：{query}"}]

    for image_b64 in image_base64_list:
        # 尝试从 base64 头部推断图片格式，默认 jpeg
        image_format = "jpeg"
        if image_b64.startswith("/9j/"):
            image_format = "jpeg"
        elif image_b64.startswith("iVBOR"):
            image_format = "png"
        elif image_b64.startswith("R0lGOD"):
            image_format = "gif"

        content_parts.append(
            {
                "type": "image_url",
                "image_url": {"url": f"data:image/{image_format};base64,{image_b64}"},
            }
        )

    return content_parts


async def generate_multimodal_answer(
    llm: BaseChatModel,
    query: str,
    docs_content: str,
    conversation_context: str = "",
    image_base64_list: Optional[List[str]] = None,
) -> str:
    """多模态答案生成：融合文字检索结果和图片内容生成答案。

    当检索结果中包含相关图片时，使用支持视觉的 LLM（VLM）生成答案，
    将图片以 base64 内嵌到消息中。若无图片，退化为纯文字生成。

    Args:
        llm: 语言模型（需支持多模态，如 qwen-vl-plus）
        query: 用户查询
        docs_content: 检索到的文字内容（带来源编号）
        conversation_context: 对话历史
        image_base64_list: 图片 base64 编码列表，为空则退化为纯文字生成

    Returns:
        生成的答案字符串
    """
    if not image_base64_list:
        # 无图片时退化为普通文字生成
        return await generate_answer_for_query(
            llm=llm,
            query=query,
            docs_content=docs_content,
            conversation_context=conversation_context,
            is_final=True,
        )

    # 构建多模态消息
    content_parts = _build_multimodal_message_content(
        query=query,
        docs_content=docs_content,
        conversation_context=conversation_context,
        image_base64_list=image_base64_list,
    )

    try:
        from langchain_core.messages import HumanMessage

        multimodal_message = HumanMessage(content=content_parts)
        response = await llm.with_config(tags=["stream_to_user"]).ainvoke(
            [multimodal_message]
        )
        return response.content if hasattr(response, "content") else str(response)
    except Exception as exc:
        # VLM 调用失败时退化为纯文字生成，保证系统可用性
        from src.observability.logger import get_logger

        get_logger(__name__).warning(
            f"[Generate] 多模态生成失败，退化为纯文字: {exc}"
        )
        return await generate_answer_for_query(
            llm=llm,
            query=query,
            docs_content=docs_content,
            conversation_context=conversation_context,
            is_final=True,
        )


async def compress_reasoning_context(llm: BaseChatModel, reasoning_context: str) -> str:
    """当 reasoning_context 过长时，使用 LLM 压缩为摘要"""
    if len(reasoning_context) <= RagSystemConfig.max_reasoning_chars:
        return reasoning_context

    prompt = ChatPromptTemplate.from_messages([
        ("system", compress_system_prompt),
        ("human", "请压缩上述记录。")
    ])
    chain = prompt | llm | StrOutputParser()
    return await chain.ainvoke({"reasoning_context": reasoning_context})
