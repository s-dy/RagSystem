"""内存管理器 - 用于管理用户个性化记忆、对话历史和上下文记忆"""

import json
import time
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Union
from langgraph.store.base import BaseStore
from langgraph.store.redis import AsyncRedisStore


class MemoryManager:
    """内存管理器，用于处理用户个性化记忆、对话历史和上下文记忆"""
    
    def __init__(self, store: BaseStore = None):
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
        except:
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
            
        # 注意：由于RedisStore的限制，这里简化实现
        # 在实际应用中，可能需要更复杂的键管理和查询逻辑
        return []
    
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
        except:
            return default_value
    
    async def save_entity_memory(self, user_id: str, entity_type: str, entity_name: str, entity_info: Any, ttl: int = 86400 * 30):
        """保存实体相关的记忆（如用户提到的人、地点、概念等）"""
        if not self.store:
            return
            
        key = f"entity:{user_id}:{entity_type}:{entity_name}"
        value = {
            "info": entity_info,
            "timestamp": time.time(),
            "ttl": ttl
        }
        await self.store.aput(("memory",), key=key, value=value)
        
    async def get_entity_memory(self, user_id: str, entity_type: str, entity_name: str, default_value: Any = None) -> Any:
        """获取实体相关的记忆"""
        if not self.store:
            return default_value
            
        key = f"entity:{user_id}:{entity_type}:{entity_name}"
        try:
            result = await self.store.aget(("memory",), key)
            if result and isinstance(result, dict):
                return result.get("info", default_value)
            return default_value
        except:
            return default_value
    
    async def search_related_memories(self, user_id: str, query: str, limit: int = 5) -> List[Dict]:
        """搜索与查询相关的记忆（简化版实现）"""
        # 由于RedisStore的限制，这里返回空列表
        # 在实际应用中，需要实现更复杂的搜索逻辑
        return []


# 全局记忆管理器实例
memory_manager = MemoryManager()


def get_memory_manager(store: BaseStore = None) -> MemoryManager:
    """获取记忆管理器实例"""
    if store:
        return MemoryManager(store)
    return memory_manager