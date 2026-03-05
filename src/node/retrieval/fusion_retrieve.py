import asyncio
from typing import List, Tuple, Any

from src.monitoring.logger import monitor_task_status
from src.services.postgres_connector import PostgreSQLConnector
from src.services.vector_db.client import MilvusExecutor, MilvusConfig


class FusionRetrieve:
    def __init__(self, use_parent_child: bool = False):
        self.use_parent_child = use_parent_child

    async def _search_single_query(self, query: str, collection_name: str = None) -> List[str]:
        """异步搜索单个查询"""
        try:
            client_config = MilvusConfig(collection_name=collection_name)
            vector_client = MilvusExecutor(client_config).client
            # rrf重排序
            # result =  await vector_client.asimilarity_search_with_score(query,k=4,ranker_type="rrf",ranker_params={"k":100})
            # 权重重排序
            result = await vector_client.asimilarity_search_with_score(
                query, k=4, ranker_type="weighted", ranker_params={"weights": [0.7, 0.3]}
            )
            # 过滤分数小于0.2的
            filtered = [(doc, score) for doc, score in result if score >= 0.2]

            if self.use_parent_child:
                result_texts = self._resolve_parent_documents(filtered)
            else:
                result_texts = [doc.page_content for doc, score in filtered]

            monitor_task_status(f"搜索查询 【{query}】: 【{result_texts}】")
            return result_texts
        except Exception as e:
            monitor_task_status(f"搜索查询 '{query}' 时发生错误: {e}", level='ERROR')
            return []

    def _resolve_parent_documents(self, search_results: List[Tuple]) -> List[str]:
        """将子文档检索结果回溯到父文档，去重后返回父文档内容。"""
        parent_ids = []
        fallback_texts = []
        for doc, score in search_results:
            parent_id = doc.metadata.get("parent_id")
            if parent_id:
                parent_ids.append(parent_id)
            else:
                fallback_texts.append(doc.page_content)

        if not parent_ids:
            return fallback_texts

        unique_parent_ids = list(dict.fromkeys(parent_ids))
        parent_map = PostgreSQLConnector().get_parent_documents_by_ids(unique_parent_ids)

        result_texts = []
        seen_parents = set()
        for parent_id in parent_ids:
            if parent_id in seen_parents:
                continue
            seen_parents.add(parent_id)
            parent_content = parent_map.get(parent_id)
            if parent_content:
                result_texts.append(parent_content)

        result_texts.extend(fallback_texts)
        return result_texts

    async def search_queries(self, queries: List[str], collection_name: str):
        tasks = [self._search_single_query(query, collection_name) for query in queries]
        results_list = await asyncio.gather(*tasks, return_exceptions=True)
        results = []
        for query, result in zip(queries, results_list):
            if not isinstance(result, Exception):
                results.extend(result)
        monitor_task_status('Queries Search Results', results)
        return results_list


if __name__ == '__main__':
    from utils.async_task import async_run
    async_run(FusionRetrieve().search_queries(['黄独分布在哪些地区？'],collection_name='cmrc_dataset'))