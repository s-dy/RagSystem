import logging
import time
from typing import Dict, List, Any, Optional, Union
from langgraph.store.postgres.aio import AsyncPostgresStore

logger = logging.getLogger(__name__)


class MemoryManager:
    """内存管理器，用于处理用户个性化记忆、对话历史和上下文记忆"""
    
    def __init__(self, store: AsyncPostgresStore = None):
        self.store = store
        
    async def save_user_preference(self, user_id: str, preference_key: str, preference_value: Any, ttl: int = 86400 * 30):
        """保存用户偏好设置"""
        if not self.store:
            return
            
        key = f"user:{user_id}:preference:{preference_key}"
        value = {
            "value": preference_value,
            "timestamp": time.time(),
            "ttl": ttl
        }
        await self.store.aput(("memory",), key=key, value=value)
        
    async def get_user_preference(self, user_id: str, preference_key: str, default_value: Any = None) -> Any:
        """获取用户偏好设置"""
        if not self.store:
            return default_value
            
        key = f"user:{user_id}:preference:{preference_key}"
        try:
            result = await self.store.aget(("memory",), key)
            if result and isinstance(result, dict):
                return result.get("value", default_value)
            return default_value
        except Exception as exc:
            logger.warning("Failed to get user preference [%s:%s]: %s", user_id, preference_key, exc)
            return default_value
    
    async def save_conversation_memory(self, user_id: str, thread_id: str, memory_type: str, content: Any, ttl: int = 86400 * 7):
        """保存对话记忆"""
        if not self.store:
            return
            
        timestamp = time.time()
        key = f"conversation:{user_id}:{thread_id}:{memory_type}:{int(timestamp)}"
        value = {
            "content": content,
            "timestamp": timestamp,
            "type": memory_type,
            "ttl": ttl
        }
        await self.store.aput(("memory",), key=key, value=value)
        
    async def get_recent_conversation_memories(self, user_id: str, thread_id: str, memory_type: str, limit: int = 10) -> List[Dict]:
        """获取最近的对话记忆"""
        if not self.store:
            return []
            
        prefix = f"conversation:{user_id}:{thread_id}:{memory_type}:"
        items = await self.store.asearch(("memory",), filter={"prefix": prefix}, limit=100)
        results = [
            item.value for item in items
            if item.value and isinstance(item.value, dict)
        ]
        results.sort(key=lambda x: x.get("timestamp", 0), reverse=True)
        return results[:limit]
    
    async def save_contextual_memory(self, user_id: str, context_key: str, context_value: Any, ttl: int = 86400 * 7):
        """保存上下文相关的记忆"""
        if not self.store:
            return
            
        key = f"context:{user_id}:{context_key}"
        value = {
            "value": context_value,
            "timestamp": time.time(),
            "ttl": ttl
        }
        await self.store.aput(("memory",), key=key, value=value)
        
    async def get_contextual_memory(self, user_id: str, context_key: str, default_value: Any = None) -> Any:
        """获取上下文相关的记忆"""
        if not self.store:
            return default_value
            
        key = f"context:{user_id}:{context_key}"
        try:
            result = await self.store.aget(("memory",), key)
            if result and isinstance(result, dict):
                return result.get("value", default_value)
            return default_value
        except Exception as exc:
            logger.warning("Failed to get contextual memory [%s:%s]: %s", user_id, context_key, exc)
            return default_value
    
    async def search_related_memories(self, user_id: str, query: str, limit: int = 5) -> List[Dict]:
        """基于分词关键词匹配搜索相关记忆"""
        if not self.store:
            return []

        import jieba
        keywords = [word for word in jieba.cut(query) if len(word) > 1]
        if not keywords:
            return []

        prefixes = [f"conversation:{user_id}:", f"context:{user_id}:"]
        results = []
        for prefix in prefixes:
            items = await self.store.asearch(("memory",), filter={"prefix": prefix}, limit=100)
            for item in items:
                content = str(item.value.get("content", item.value.get("value", "")))
                if any(keyword in content for keyword in keywords):
                    results.append(item.value)
                if len(results) >= limit:
                    break
            if len(results) >= limit:
                break
        return results


# 全局记忆管理器实例
memory_manager = MemoryManager()


def get_memory_manager(store: AsyncPostgresStore = None) -> MemoryManager:
    """获取记忆管理器实例"""
    if store:
        return MemoryManager(store)
    return memory_manager