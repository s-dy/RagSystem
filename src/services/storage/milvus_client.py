import threading

from langchain_milvus import Milvus, BM25BuiltInFunction
from pymilvus import connections, db

from config import MilvusConfig
from src.observability.logger import get_logger
from src.services.llm.models import get_embedding_model

logger = get_logger(__name__)

_milvus_db_ensured = set()

def ensure_milvus_database_exists(config: MilvusConfig = None):
    """检测目标 Milvus 数据库是否存在，不存在则自动创建"""
    if config is None:
        config = MilvusConfig()
    target_db = config.db_name
    if target_db in _milvus_db_ensured:
        return

    alias = f"db_check_{target_db}"
    try:
        connections.connect(
            alias=alias,
            host=config.host,
            port=config.port,
            token=config.token,
        )
        existing_dbs = db.list_database(using=alias)
        if target_db not in existing_dbs:
            db.create_database(target_db, using=alias)
            logger.info(f"[Milvus] 数据库自动创建: {target_db}")
        _milvus_db_ensured.add(target_db)
        connections.disconnect(alias)
    except Exception as error:
        logger.warning(f"[Milvus] 数据库检测/创建失败: {error}")


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
        ensure_milvus_database_exists(config)
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