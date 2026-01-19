import asyncio
from typing import List, Tuple, Any

from src.monitoring.logger import monitor_task_status
from src.services.vector_db.client import MilvusExecutor, MilvusConfig


class FusionRetrieve:
    def __init__(self):
        pass

    async def _search_single_query(self, query: str,collection_name:str=None) -> List[str]:
        """异步搜索单个查询"""
        try:
            client_config = MilvusConfig(collection_name=collection_name)
            vector_client = MilvusExecutor(client_config).client
            # rrf重排序
            # result =  await vector_client.asimilarity_search_with_score(query,k=4,ranker_type="rrf",ranker_params={"k":100})
            # 权重重排序
            result =  await vector_client.asimilarity_search_with_score(query,k=4,ranker_type="weighted",ranker_params={"weights":[0.7, 0.3]})
            # 过滤
            # print(result)
            result = [doc.page_content for doc,score in result if score >= 0.2]
            monitor_task_status(f"搜索查询 【{query}】: 【{result}】")
            return result
        except Exception as e:
            monitor_task_status(f"搜索查询 '{query}' 时发生错误: {e}",level='ERROR')
            return []

    async def search_queries(self,queries:List[str],collection_name:str):
        tasks = [self._search_single_query(query,collection_name) for query in queries]
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