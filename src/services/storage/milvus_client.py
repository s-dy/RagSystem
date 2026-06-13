import threading
import re
import hashlib

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


def sanitize_collection_name(name: str) -> str:
    """将任意用户输入的 collection 名规范化为 Milvus 可接受的格式。

    规则：
    - 非 ASCII 字母数字下划线的字符替换为下划线
    - 保证首字符为字母或下划线，否则前缀 `c_`
    - 为避免冲突，追加原始名字的 8 位哈希后缀
    """
    if not name:
        return "default"
    # 将连续的非法字符替换为单个下划线，并去除首尾下划线
    s = re.sub(r"[^A-Za-z0-9_]+", "_", name)
    s = re.sub(r"_+", "_", s).strip("_")
    if not re.match(r"^[A-Za-z_]", s):
        s = f"c_{s}"
    # 限制长度并追加哈希后缀以确保唯一性
    h = hashlib.sha1(name.encode("utf-8")).hexdigest()[:8]
    # 最终长度保守控制在 120 字符以内
    base = s[:110]
    return f"{base}_{h}"


class MilvusExecutor:
    """Milvus 客户端，按 collection_name 缓存实例（单例池），避免重复创建连接。"""

    _instances: dict[str, "MilvusExecutor"] = {}
    _lock = threading.Lock()

    def __new__(cls, config: MilvusConfig = None):
        if config is None:
            config = MilvusConfig()
        # 使用规范化后的 collection 名作为缓存 key，防止非法字符导致 Milvus 抛错
        collection_name = sanitize_collection_name(config.collection_name)

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
        # collection_name 应该已经是规范化后的名称（由调用方负责 sanitize）
        self.collection_name_raw = config.collection_name
        self.collection_name = config.collection_name

        ensure_milvus_database_exists(config)
        self.dense_embedding = get_embedding_model("qwen")
        self.vector_store = self._create_client()
        self._initialized = True

    def _create_client(self) -> Milvus:
        """创建 Milvus 客户端连接"""
        uri = f"http://{self.config.host}:{self.config.port}"
        return Milvus(
            collection_name=self.collection_name,
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