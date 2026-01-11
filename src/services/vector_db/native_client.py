from pymilvus import MilvusClient, DataType
from pymilvus import AnnSearchRequest, RRFRanker

from config.Config import MilvusConfig
from src.monitoring.logger import monitor_task_status
from src.services.llm.models import get_embedding_model


class NativeMilvusClient:
    def __init__(self, config: MilvusConfig = None):
        if config is None:
            config = MilvusConfig()
        self.config = config
        # 初始化 Embedding 模型
        self.embeddings = get_embedding_model('qwen')
        self.client = self.init_client()

    def init_client(self) -> MilvusClient:
        # 创建Milvus Client。
        URI = f"http://{self.config.host}:{self.config.port}"
        client = MilvusClient(
            uri=URI,
            token=self.config.token,
            db_name=self.config.db_name
        )
        return client

    def _create_sparse_vector(self, query: str):
        """
        稀疏向量表示
        注意：实际的BM25稀疏向量需要专门的tokenizer
        """
        # 这里你需要使用与创建时相同的tokenizer来生成稀疏向量
        # 由于 langchain_milvus 内部使用了 BM25BuiltInFunction，
        # 你可能需要直接使用 langchain_milvus 的稀疏搜索功能

        # 作为临时解决方案，可以尝试使用原始查询文本
        # 但更推荐使用统一的方式处理
        return [query]  # 暂时返回原始查询文本

    def hybrid_search(self, query: str,top_k: int=5):
        # 生成稠密向量
        query_embeddings = self.embeddings.embed_query(query)
        # 稠密向量搜索参数
        search_params_dense = {
            "data": [query_embeddings],
            "anns_field": "dense",
            "param": {
                "metric_type": "IP",
                "params": {"nprobe": 2}
            },
            "limit": top_k
        }
        # 创建稠密向量搜索请求
        request_dense = AnnSearchRequest(**search_params_dense)

        # 稀疏向量搜索参数
        sparse_input = self._create_sparse_vector(query)
        search_params_sparse = {
            "data": sparse_input,
            "anns_field": "sparse",
            "param": {
                "metric_type": "BM25",
            },
            "limit": top_k
        }
        # 创建稀疏向量搜索请求
        request_sparse = AnnSearchRequest(**search_params_sparse)

        reqs = [request_dense, request_sparse]

        # 重排序
        ranker = RRFRanker(100)

        # 开始执行搜索
        result = self.client.hybrid_search(
            collection_name=self.config.collection_name,
            reqs=reqs,
            ranker=ranker,
            limit=top_k,
            output_fields=["text"]
        )
        monitor_task_status(result)
        return result

    def batch_vector_search(self, queries: list[str], top_k: int = 5):
        """纯向量搜索作为回退方案"""
        query_embeddings = []
        for query in queries:
            query_embeddings.append(self.embeddings.embed_query(query))
        search_params = {
            "metric_type": "IP",
            "params": {"nprobe": 2}
        }
        result = self.client.search(
            collection_name=self.config.collection_name,
            data=query_embeddings,
            anns_field="dense",  # 搜索稠密向量字段
            search_params=search_params,
            limit=top_k,
            output_fields=["text"]  # 获取需要的字段
        )
        monitor_task_status(result)
        return result

    def get_collections(self):
        collections = self.client.list_collections()
        monitor_task_status(collections)
        return collections


if __name__ == '__main__':
    # 测试代码
    query = "今天茅台的新闻"

    # 创建客户端
    client = NativeMilvusClient(MilvusConfig(collection_name='hybridRag_news'))

    # 混合搜索
    result = client.hybrid_search(query)
    # result = client.batch_vector_search([query,"兔子女警是谁？"])
    #
    # # 打印结果
    # print("Search Results:")
    # for hits in result:
    #     for hit in hits:
    #         # 根据实际的返回结构调整
    #         print(f"distance: {hit.get('distance', 'N/A')}")
    #         print(f"Text: {hit.get('entity', {}).get('text', 'N/A')}")
    #         print("-" * 50)

    # client.get_collections()