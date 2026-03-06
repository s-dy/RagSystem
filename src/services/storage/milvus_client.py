import threading

from langchain_milvus import Milvus, BM25BuiltInFunction

from config import MilvusConfig
from src.services.llm.models import get_embedding_model


class MilvusExecutor:
    """Milvus 客户端，按 collection_name 缓存实例（单例池），避免重复创建连接。"""

    _instances: dict[str, "MilvusExecutor"] = {}
    _lock = threading.Lock()

    def __new__(cls, config: MilvusConfig = None):
        if config is None:
            config = MilvusConfig()
        collection_name = config.collection_name

        with cls._lock:
            if collection_name not in cls._instances:
                instance = super().__new__(cls)
                instance._initialized = False
                cls._instances[collection_name] = instance
            return cls._instances[collection_name]

    def __init__(self, config: MilvusConfig = None):
        if self._initialized:
            return
        if config is None:
            config = MilvusConfig()
        self.config = config
        self.dense_embedding = get_embedding_model("qwen")
        self.vector_store = self._create_client()
        self._initialized = True

    def _create_client(self) -> Milvus:
        """创建 Milvus 客户端连接"""
        uri = f"http://{self.config.host}:{self.config.port}"
        return Milvus(
            collection_name=self.config.collection_name,
            embedding_function=self.dense_embedding,
            connection_args={
                "uri": uri,
                "db_name": self.config.db_name,
                "token": self.config.token,
            },
            consistency_level="Bounded",
            index_params=[
                {"index_type": "HNSW", "metric_type": "IP"},
                {"index_type": "AUTOINDEX", "metric_type": "BM25"},
            ],
            builtin_function=BM25BuiltInFunction(),
            vector_field=["dense", "sparse"],
            auto_id=True,
        )

    @property
    def client(self) -> Milvus:
        return self.vector_store
