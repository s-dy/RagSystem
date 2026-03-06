"""HybridRAG 业务异常体系

异常分层：
    HybridRagError                  ← 基类，所有业务异常的父类
    ├── ConfigError                 ← 配置缺失或无效
    ├── StorageError                ← 存储层（Milvus / PostgreSQL）
    │   ├── MilvusConnectionError   ← Milvus 连接失败
    │   └── PostgresConnectionError ← PostgreSQL 连接失败
    ├── RetrievalError              ← 检索层
    │   ├── InternalRetrievalError  ← 内部向量检索失败
    │   └── ExternalSearchError     ← 外部搜索（MCP/Bing）失败
    ├── IngestError                 ← 文档入库流程失败
    └── GenerationError             ← LLM 生成/推理失败
"""


class HybridRagError(Exception):
    """HybridRAG 业务异常基类"""

    def __init__(self, message: str, cause: Exception = None):
        self.cause = cause
        full_message = f"{message}: {cause}" if cause else message
        super().__init__(full_message)


# ─── 配置层 ───

class ConfigError(HybridRagError):
    """配置缺失或无效"""
    pass


# ─── 存储层 ───

class StorageError(HybridRagError):
    """存储操作失败（Milvus / PostgreSQL 通用）"""
    pass


class MilvusConnectionError(StorageError):
    """Milvus 连接或操作失败"""
    pass


class PostgresConnectionError(StorageError):
    """PostgreSQL 连接或操作失败"""
    pass


# ─── 检索层 ───

class RetrievalError(HybridRagError):
    """检索失败（通用）"""
    pass


class InternalRetrievalError(RetrievalError):
    """内部向量检索失败（Milvus 查询异常）"""

    def __init__(self, collection_name: str, cause: Exception = None):
        self.collection_name = collection_name
        super().__init__(f"内部检索失败 [collection={collection_name}]", cause)


class ExternalSearchError(RetrievalError):
    """外部搜索失败（MCP / Bing 等）"""

    def __init__(self, message: str = "外部搜索失败", attempt: int = 0, max_retries: int = 0, cause: Exception = None):
        self.attempt = attempt
        self.max_retries = max_retries
        super().__init__(f"{message} (attempt {attempt}/{max_retries})", cause)


# ─── 入库层 ───

class IngestError(HybridRagError):
    """文档入库流程失败"""

    def __init__(self, collection_name: str, stage: str = "", cause: Exception = None):
        self.collection_name = collection_name
        self.stage = stage
        detail = f" [{stage}]" if stage else ""
        super().__init__(f"入库失败{detail} [collection={collection_name}]", cause)


# ─── 生成层 ───

class GenerationError(HybridRagError):
    """LLM 生成/推理失败"""
    pass
