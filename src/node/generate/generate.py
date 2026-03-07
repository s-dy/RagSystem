from typing import AsyncGenerator

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
