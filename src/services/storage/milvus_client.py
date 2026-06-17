import threading
import re
import hashlib
import logging

from langchain_milvus import Milvus, BM25BuiltInFunction
from pymilvus import connections, db

from config import MilvusConfig
from src.observability.logger import get_logger
from src.services.llm.models import get_embedding_model

logger = get_logger(__name__)


# ── PyMilvus 2.6+ 兼容性补丁 ─────────────────────────────────────
# PyMilvus 2.6 将 MilvusClient 的连接管理迁移到 ConnectionManager，
# 不再向 orm.connections.Connections 注册 alias。
# langchain_milvus 中的 Collection ORM 类仍从 Connections._alias_handlers
# 查找连接，导致 alias="cm-xxxxx" 永远找不到，抛 ConnectionNotExistException。
#
# 此处 patch Connections._fetch_handler：
#   1. 先走原始逻辑（查 _alias_handlers）
#   2. 若未找到，再到 ConnectionManager._registry 中按
#      alias == f"cm-{id(handler)}" 匹配 handler 并返回
#   3. 匹配成功后顺手写入 _alias_handlers 缓存，避免重复扫描
#
# 该补丁仅在 pymilvus >= 2.6.0 且存在 Connections._alias_handlers 时生效；
# 其他版本（2.5.x 及更早）不需要此补丁，try 块静默跳过即可。
try:
    from importlib.metadata import version as _get_pkg_ver
    _pymilvus_ver = tuple(int(x) for x in _get_pkg_ver("pymilvus").split(".")[:2])
except Exception:
    _pymilvus_ver = (0, 0)

if _pymilvus_ver >= (2, 6):
    try:
        from pymilvus.orm.connections import Connections
        from pymilvus.client.connection_manager import ConnectionManager
        from pymilvus.exceptions import ConnectionNotExistException

        _orig_fetch = Connections._fetch_handler

        def _patched_fetch_handler(self, alias=str):
            """兼容 PyMilvus 2.6+：优先查 _alias_handlers，未命中则搜 ConnectionManager。"""
            try:
                return _orig_fetch(self, alias)
            except ConnectionNotExistException:

                pass  # 原始逻辑未命中，继续搜 ConnectionManager

            mgr = ConnectionManager.get_instance()
            _registry = getattr(mgr, "_registry", {})
            for _managed in _registry.values():
                _handler = _managed.handler
                if f"cm-{id(_handler)}" == alias:
                    # 顺手注册到 _alias_handlers，后续调用直接命中
                    self._alias_handlers[alias] = _handler
                    logger.debug(
                        f"[MilvusExecutor] PyMilvus 2.6+ 补丁："
                        f"为 alias={alias} 注入 handler"
                    )
                    return _handler

            # 确实没有，重新抛原始异常
            raise ConnectionNotExistException(
                message=f"ConnectionNotExistException: alias={alias} not found (patched)"
            )

        Connections._fetch_handler = _patched_fetch_handler
        logger.info(
            "[MilvusExecutor] 已应用 PyMilvus 2.6+ 兼容补丁 "
            "(patch Connections._fetch_handler)"
        )
    except Exception as _patch_err:
        # 结构变化时不阻塞启动，仅记录
        logger.warning(
            f"[MilvusExecutor] PyMilvus 2.6+ 补丁应用失败: {_patch_err}"
        )
# ─────────────────────────────────────────────────────────────────────────────


_milvus_db_ensured = set()
_milvus_db_lock = threading.Lock()


def ensure_milvus_database_exists(config: MilvusConfig = None):
    """检测目标 Milvus 数据库是否存在，不存在则自动创建。

    注意：使用独立 alias 连接，但不再主动 disconnect，
    避免在某些 PyMilvus 版本中间接清理掉 langchain_milvus 正在使用的共享 gRPC 通道。
    """
    if config is None:
        config = MilvusConfig()
    target_db = config.db_name
    if target_db in _milvus_db_ensured:
        return

    with _milvus_db_lock:
        # double-check，防止并发重复创建
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
            # 不再 disconnect：该 alias 连接留给后续复用，
            # 进程退出时由 PyMilvus 统一清理。
        except Exception as error:
            logger.warning(f"[Milvus] 数据库检测/创建失败: {error}")


_SANITIZED_NAME_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*_[0-9a-f]{8}$")


def is_sanitized_name(name: str) -> bool:
    """判断 name 是否已经是 sanitize_collection_name() 输出的规范化名称。

    规范化名称的特征：以字母或下划线开头，仅含 ASCII 字母/数字/下划线，
    并以 8 位十六进制哈希后缀（_xxxxxxxx）结尾。
    """
    return bool(_SANITIZED_NAME_RE.match(name))


def sanitize_collection_name(name: str) -> str:
    """将任意用户输入的 collection 名规范化为 Milvus 可接受的格式。

    规则：
    - 非 ASCII 字母数字下划线的字符替换为下划线
    - 保证首字符为字母或下划线，否则前缀 `c_`
    - 为避免冲突，追加原始名字的 8 位哈希后缀

    幂等性：若 name 已经是规范化名称（由本函数生成），则原样返回，避免重复 sanitize
    导致集合名二次变形而找不到 Milvus 中已存在的集合。
    """
    if not name:
        return "default"
    # 幂等保护：已规范化则直接返回
    if is_sanitized_name(name):
        return name
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
    """Milvus 客户端，按 collection_name 缓存实例（单例池），避免重复创建连接。

    连接失效自动重建：当检测到底层 pymilvus 连接已断开时，清除缓存并重新初始化
    vector_store，避免因网络波动、Milvus 服务重启等原因导致后续请求持续报
    ConnectionNotExistException。
    """

    _instances: dict[str, "MilvusExecutor"] = {}
    _lock = threading.Lock()

    def __new__(cls, config: MilvusConfig = None):
        if config is None:
            config = MilvusConfig()
        # sanitize_collection_name 现在是幂等的：已规范化的名称原样返回，不会再加哈希
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
        self.collection_name_raw = config.collection_name
        # 规范化（幂等）：已规范化的 internal_name 原样保留
        self.collection_name = sanitize_collection_name(config.collection_name)

        ensure_milvus_database_exists(config)
        self.dense_embedding = get_embedding_model("qwen")
        self.vector_store = self._create_client()
        self._initialized = True

    def _create_client(self) -> Milvus:
        """创建 Milvus 客户端连接。

        PyMilvus 2.6+ 的兼容补丁已在模块加载时（文件顶部）应用，
        此处无需再手动注册 handler。
        """
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

    def _reconnect(self) -> None:
        """重建底层 vector_store 连接，并更新缓存中的实例状态。

        在检测到 ConnectionNotExistException 时调用，线程安全。
        """
        with self._lock:
            logger.warning(
                f"[MilvusExecutor] 检测到连接失效，正在重连: collection={self.collection_name}"
            )
            try:
                self.vector_store = self._create_client()
                logger.info(
                    f"[MilvusExecutor] 重连成功: collection={self.collection_name}"
                )
            except Exception as e:
                logger.error(
                    f"[MilvusExecutor] 重连失败: collection={self.collection_name}, error={e}"
                )
                raise

    @property
    def client(self) -> Milvus:
        return self.vector_store
