from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate


async def retrieve_answer_or_retrieve(llm,query,conversation_context):
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
    chain = prompt | llm | StrOutputParser()
    response = await chain.ainvoke({'query': query, 'conversation_context': conversation_context})
    return response
