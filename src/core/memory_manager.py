import time
from typing import Dict, List, Any

from langgraph.store.postgres.aio import AsyncPostgresStore

from src.observability.logger import get_logger

logger = get_logger(__name__)


class MemoryManager:
    """内存管理器，用于处理用户个性化记忆、对话历史和上下文记忆"""

    def __init__(self, store: AsyncPostgresStore = None):
        self.store = store

    @staticmethod
    def _is_expired(record: dict) -> bool:
        """检查记录是否已超过 TTL 过期时间。

        Args:
            record: 存储的记录字典，需包含 timestamp 和 ttl 字段

        Returns:
            True 表示已过期，False 表示未过期或无 TTL 限制
        """
        ttl = record.get("ttl")
        timestamp = record.get("timestamp")
        if ttl is None or timestamp is None:
            return False
        return (time.time() - timestamp) > ttl

    async def save_user_preference(
        self,
        user_id: str,
        preference_key: str,
        preference_value: Any,
        ttl: int = 86400 * 30,
    ):
        """保存用户偏好设置"""
        if not self.store:
            return

        key = f"user:{user_id}:preference:{preference_key}"
        value = {"value": preference_value, "timestamp": time.time(), "ttl": ttl}
        await self.store.aput(("memory",), key=key, value=value)
        logger.debug(
            f"[MemoryManager] 保存用户偏好: user={user_id}, key={preference_key}"
        )

    async def get_user_preference(
        self, user_id: str, preference_key: str, default_value: Any = None
    ) -> Any:
        """获取用户偏好设置，自动检查 TTL 是否过期"""
        if not self.store:
            return default_value

        key = f"user:{user_id}:preference:{preference_key}"
        try:
            result = await self.store.aget(("memory",), key)
            if result and isinstance(result, dict):
                if self._is_expired(result):
                    await self.store.adelete(("memory",), key)
                    return default_value
                return result.get("value", default_value)
            return default_value
        except Exception as exc:
            logger.warning(
                f"[MemoryManager] 获取用户偏好失败: user={user_id}, key={preference_key}, error={exc}"
            )
            return default_value

    async def save_conversation_memory(
        self,
        user_id: str,
        thread_id: str,
        memory_type: str,
        content: Any,
        ttl: int = 86400 * 7,
    ):
        """保存对话记忆"""
        if not self.store:
            return

        timestamp = time.time()
        key = f"conversation:{user_id}:{thread_id}:{memory_type}:{int(timestamp)}"
        value = {
            "content": content,
            "timestamp": timestamp,
            "type": memory_type,
            "ttl": ttl,
        }
        await self.store.aput(("memory",), key=key, value=value)
        logger.debug(
            f"[MemoryManager] 保存对话记忆: user={user_id}, thread={thread_id}, type={memory_type}"
        )

    async def get_recent_conversation_memories(
        self, user_id: str, thread_id: str, memory_type: str, limit: int = 10
    ) -> List[Dict]:
        """获取最近的对话记忆，自动过滤已过期的条目"""
        if not self.store:
            return []

        prefix = f"conversation:{user_id}:{thread_id}:{memory_type}:"
        items = await self.store.asearch(
            ("memory",), filter={"prefix": prefix}, limit=100
        )
        valid_results = []
        for item in items:
            if not (item.value and isinstance(item.value, dict)):
                continue
            if self._is_expired(item.value):
                try:
                    await self.store.adelete(("memory",), item.key)
                except Exception:
                    pass
                continue
            valid_results.append(item.value)
        valid_results.sort(key=lambda x: x.get("timestamp", 0), reverse=True)
        logger.debug(
            f"[MemoryManager] 获取对话记忆: user={user_id}, thread={thread_id}, type={memory_type}, count={len(valid_results[:limit])}"
        )
        return valid_results[:limit]

    async def save_contextual_memory(
        self, user_id: str, context_key: str, context_value: Any, ttl: int = 86400 * 7
    ):
        """保存上下文相关的记忆"""
        if not self.store:
            return

        key = f"context:{user_id}:{context_key}"
        value = {"value": context_value, "timestamp": time.time(), "ttl": ttl}
        await self.store.aput(("memory",), key=key, value=value)
        logger.debug(
            f"[MemoryManager] 保存上下文记忆: user={user_id}, key={context_key}"
        )

    async def get_contextual_memory(
        self, user_id: str, context_key: str, default_value: Any = None
    ) -> Any:
        """获取上下文相关的记忆，自动检查 TTL 是否过期"""
        if not self.store:
            return default_value

        key = f"context:{user_id}:{context_key}"
        try:
            result = await self.store.aget(("memory",), key)
            if result and isinstance(result, dict):
                if self._is_expired(result):
                    await self.store.adelete(("memory",), key)
                    return default_value
                return result.get("value", default_value)
            return default_value
        except Exception as exc:
            logger.warning(
                f"[MemoryManager] 获取上下文记忆失败: user={user_id}, key={context_key}, error={exc}"
            )
            return default_value

    async def search_related_memories(
        self, user_id: str, query: str, limit: int = 5
    ) -> List[Dict]:
        """基于向量语义搜索相关记忆，自动过滤已过期的条目。

        优先使用 AsyncPostgresStore 原生的语义搜索（需要 LangGraph Store 配置了
        embedding 函数）；若语义搜索不可用，则回退到关键词匹配作为兜底。

        Args:
            user_id: 用户 ID，用于限定搜索范围
            query: 搜索查询文本
            limit: 返回结果数量上限

        Returns:
            相关记忆列表，按相关性降序排列
        """
        if not self.store:
            return []

        results = []

        # 优先尝试 AsyncPostgresStore 的语义搜索（传入 query 参数触发向量检索）
        try:
            items = await self.store.asearch(
                ("memory",),
                query=query,
                filter={"namespace_prefix": f"conversation:{user_id}"},
                limit=limit * 2,  # 多取一些，过滤过期后仍能满足 limit
            )
            for item in items:
                if not (item.value and isinstance(item.value, dict)):
                    continue
                if self._is_expired(item.value):
                    try:
                        await self.store.adelete(("memory",), item.key)
                    except Exception:
                        pass
                    continue
                results.append(item.value)
                if len(results) >= limit:
                    break

            if results:
                logger.debug(
                    f"[MemoryManager] 向量搜索相关记忆: user={user_id}, query={query[:30]}, count={len(results)}"
                )
                return results
        except Exception as semantic_error:
            logger.debug(
                f"[MemoryManager] 向量搜索不可用，回退到关键词匹配: {semantic_error}"
            )

        # 回退：关键词匹配（兜底策略）
        import jieba

        keywords = [word for word in jieba.cut(query) if len(word) > 1]
        if not keywords:
            return []

        prefixes = [f"conversation:{user_id}:", f"context:{user_id}:"]
        for prefix in prefixes:
            items = await self.store.asearch(
                ("memory",), filter={"prefix": prefix}, limit=100
            )
            for item in items:
                if not (item.value and isinstance(item.value, dict)):
                    continue
                if self._is_expired(item.value):
                    try:
                        await self.store.adelete(("memory",), item.key)
                    except Exception:
                        pass
                    continue
                content = str(item.value.get("content", item.value.get("value", "")))
                if any(keyword in content for keyword in keywords):
                    results.append(item.value)
                if len(results) >= limit:
                    break
            if len(results) >= limit:
                break

        logger.debug(
            f"[MemoryManager] 关键词搜索相关记忆: user={user_id}, keywords={keywords[:3]}, count={len(results)}"
        )
        return results


# 全局记忆管理器实例
memory_manager = MemoryManager()


def get_memory_manager(store: AsyncPostgresStore = None) -> MemoryManager:
    """获取记忆管理器实例"""
    if store:
        return MemoryManager(store)
    return memory_manager
