import asyncio
from dataclasses import dataclass
from typing import List, Tuple, Optional

from src.observability.logger import monitor_task_status
from src.services.storage import PostgreSQLConnector, MilvusExecutor, MilvusConfig


@dataclass
class RetrievedDoc:
    """检索结果的结构化表示，携带内容、来源和分数"""
    content: str
    source: str
    score: float


class FusionRetrieve:
    def __init__(self, use_parent_child: bool = False):
        self.use_parent_child = use_parent_child

    async def _search_single_query(self, query: str, collection_name: str = None) -> List[RetrievedDoc]:
        """异步搜索单个查询，返回带来源和分数的结构化结果"""
        try:
            vector_client = MilvusExecutor(MilvusConfig(collection_name=collection_name)).client
            # # weighted
            # result = await vector_client.asimilarity_search_with_score(
            #     query, k=4, ranker_type="weighted", ranker_params={"weights": [0.7, 0.3]}
            # )
            # rrf
            result = await vector_client.asimilarity_search_with_score(
                query, k=4, ranker_type="rrf", ranker_params={"k": 60}
            )
            # filtered = [(doc, score) for doc, score in result if score >= 0.2]

            if self.use_parent_child:
                return await self._resolve_parent_documents_structured(result)

            retrieved_docs = []
            for doc, score in result:
                source = doc.metadata.get("source", "未知来源")
                retrieved_docs.append(RetrievedDoc(
                    content=doc.page_content,
                    source=source,
                    score=float(score),
                ))
            monitor_task_status(f"搜索查询 【{query}】: 【{len(retrieved_docs)} 条结果】")
            return retrieved_docs
        except Exception as e:
            monitor_task_status(f"搜索查询 \'{query}\' 时发生错误: {e}", level='ERROR')
            return []

    async def _resolve_parent_documents_structured(self, search_results: List[Tuple]) -> List[RetrievedDoc]:
        """将子文档检索结果回溯到父文档，返回结构化结果。"""
        parent_ids = []
        fallback_docs = []
        score_map = {}
        for doc, score in search_results:
            parent_id = doc.metadata.get("parent_id")
            if parent_id:
                parent_ids.append(parent_id)
                if parent_id not in score_map or score > score_map[parent_id]:
                    score_map[parent_id] = float(score)
            else:
                source = doc.metadata.get("source", "未知来源")
                fallback_docs.append(RetrievedDoc(
                    content=doc.page_content,
                    source=source,
                    score=float(score),
                ))

        if not parent_ids:
            return fallback_docs

        unique_parent_ids = list(dict.fromkeys(parent_ids))
        parent_map = await asyncio.to_thread(
            PostgreSQLConnector().get_parent_documents_by_ids, unique_parent_ids
        )

        result_docs = []
        seen_parents = set()
        for parent_id in parent_ids:
            if parent_id in seen_parents:
                continue
            seen_parents.add(parent_id)
            parent_content = parent_map.get(parent_id)
            if parent_content:
                result_docs.append(RetrievedDoc(
                    content=parent_content,
                    source=parent_id,
                    score=score_map.get(parent_id, 0.0),
                ))

        result_docs.extend(fallback_docs)
        return result_docs

    async def search_queries(self, queries: List[str], collection_name: str) -> List[List[RetrievedDoc]]:
        """并行检索多个查询，返回每个查询的结构化结果列表"""
        tasks = [self._search_single_query(query, collection_name) for query in queries]
        results_list = await asyncio.gather(*tasks, return_exceptions=True)
        valid_results = []
        for query, result in zip(queries, results_list):
            if isinstance(result, Exception):
                monitor_task_status(f"查询 \'{query}\' 检索异常: {result}", level='ERROR')
                valid_results.append([])
            else:
                valid_results.append(result)
        monitor_task_status('Queries Search Results', f"{sum(len(r) for r in valid_results)} docs total")
        return valid_results


if __name__ == '__main__':
    from utils.async_task import async_run
    async_run(FusionRetrieve().search_queries(['黄独分布在哪些地区？'], collection_name='cmrc_dataset'))
