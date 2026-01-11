from langchain_milvus import Milvus, BM25BuiltInFunction

from config.Config import MilvusConfig
from src.services.llm.models import get_embedding_model


class MilvusExecutor:
    def __init__(self, config: MilvusConfig = None):
        if config is None:
            config = MilvusConfig()
        self.config = config

        self.dense_embedding = get_embedding_model('qwen')
        self.vector_store = self.init_client()

    def init_client(self) -> Milvus:
        """初始化Milvus客户端"""
        URI = f"http://{self.config.host}:{self.config.port}"
        client = Milvus(
            collection_name=self.config.collection_name,
            embedding_function=self.dense_embedding,
            connection_args={
                "uri": URI,
                "db_name": self.config.db_name,
                "token": self.config.token,
            },
            consistency_level="Bounded",
            index_params=[{"index_type": "HNSW", "metric_type": "IP"},{"index_type": "AUTOINDEX", "metric_type": "BM25"}],
            # search_params={'sparse':{'nprobe':2}},
            builtin_function=BM25BuiltInFunction(),
            vector_field=['dense','sparse'], # 稠密向量 + 稀疏向量
            auto_id=True,
        )
        return client

    @property
    def client(self) -> Milvus:
        return self.vector_store

if __name__ == '__main__':
    # 验证函数
    async def verify_milvus_setup():
        # 创建配置
        config = MilvusConfig(
            collection_name="hybridRag_news",
        )
        # 初始化客户端
        client = MilvusExecutor(config).client
        print(client.similarity_search_with_score('茅台的新闻', k=4, ranker_type="rrf", ranker_params={"k": 100}))


    import asyncio

    asyncio.run(verify_milvus_setup())