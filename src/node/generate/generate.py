from langchain_core.language_models import BaseChatModel
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

final_system_prompt = """
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
6. 如果答案来自多个来源，整合信息，给出简洁、准确、完整的最终答案。
7. 回答要简短。
8. 不要提及【根据检索到的信息】。
"""

# 子问题生成
sub_system_prompt = """
你是一个精准的问答助手。请基于以下信息回答问题。

已知上下文（之前已解答的问题）：
{reasoning_context}

检索到的信息：
{content}

规则：
- 只回答当前问题，不要解释过程。
- 若信息不足，回答“未知”。
- 不要编造信息。
"""

# 总结多条问题模版
synthesize_system_prompt = """
基于以下子问题解答，回答原始问题。请整合信息，给出简洁、准确、完整的最终答案。

子问题解答记录：
{reasoning_context}
"""

async def generate_answer_for_query(
        llm:BaseChatModel,
        query: str,
        docs_content: str,
        conversation_context: str = "",
        reasoning_context: str = "",  # 用于多跳
        is_final: bool = False  # 是否最终答案
) -> str:
    if is_final:
        system_prompt = final_system_prompt
        full_context = conversation_context
    else:
        system_prompt = sub_system_prompt
        full_context = reasoning_context

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{query}")
    ])
    chain = prompt | llm | StrOutputParser()
    response = await chain.ainvoke({
        "query": query,
        "content": docs_content,
        "conversation_context": conversation_context,
        "reasoning_context": full_context
    })
    return response


async def synthesize_final_subs(llm,query,reasoning_context):
    prompt = ChatPromptTemplate.from_messages([
        ("system", synthesize_system_prompt),
        ("human", "{query}")
    ])
    chain = prompt | llm | StrOutputParser()
    final_ans = await chain.ainvoke({
        "reasoning_context": reasoning_context,
        "query": query
    })
    return final_ans