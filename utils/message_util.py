from typing import List
from langchain_core.messages import AnyMessage, HumanMessage, AIMessage

from src.monitoring.logger import monitor_task_status


def get_last_user_msg(messages: List[AnyMessage]):
    user_messages = [msg for msg in messages if isinstance(msg, HumanMessage)]
    if not user_messages:
        return None
    last_msg = user_messages[-1].content
    monitor_task_status('last human message', last_msg)
    return last_msg


def get_conversation_context(messages: List[AnyMessage], num_messages: int = 3) -> str:
    """获取对话上下文，用于多轮对话理解"""
    if not messages:
        return ""

    # 获取最近的几条消息作为上下文
    recent_messages = messages[-num_messages:] if len(messages) >= num_messages else messages
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
    monitor_task_status('conversation context', context)
    return context
