from dataclasses import dataclass, field
import os

from dotenv import load_dotenv

load_dotenv()

@dataclass
class RagSystemConfig:
    # 是否开启 RAG 评估（基于 ragas 指标）
    enable_eval: bool = False
    # 是否启用父子文档检索策略（子文档检索 → 父文档回溯）
    enable_parent_child_retrieval: bool = False

@dataclass
class QueryEnhancementConfig:
    """查询增强配置"""
    # 同义改写提高召回
    paraphrase:bool = False
    # 基于专业水平改写
    formalize:bool = False
    # 扩展改写,扩展比较维度、步骤细节
    expand:bool = False
    # 查询分解,分解多步子问题
    enable_query_decomposition: bool = False
    # HyDE predict 生成假设答案辅助检索
    hyde_predict:bool = False

    # 性能限制
    max_enhanced_queries: int = 8


@dataclass
class MilvusConfig:
    collection_name: str = os.getenv("MILVUS_COLLECTION_NAME", "default")
    host: str = os.getenv("MILVUS_HOST", "localhost")
    port: int = int(os.getenv("MILVUS_PORT", "19530"))
    db_name: str = os.getenv("MILVUS_DB_NAME", "hybridRagSystem")
    token: str = os.getenv("MILVUS_TOKEN", "root:Milvus")

@dataclass
class PostgreSQLConfig:
    host: str = os.getenv("POSTGRES_HOST", "localhost")
    port: int = int(os.getenv("POSTGRES_PORT", "5432"))
    user: str = os.getenv("POSTGRES_USER", "postgres")
    password: str = os.getenv("POSTGRES_PASSWORD", "")
    dbname: str = os.getenv("POSTGRES_DBNAME", "hybridragsystem")
    autocommit: bool = True


REDIS_URI = os.getenv("REDIS_URI", "redis://localhost:6379")
os.environ['REDIS_URL'] = REDIS_URI

POSTGRESQL_URL = f"postgresql://{PostgreSQLConfig.user}:{PostgreSQLConfig.password}@{PostgreSQLConfig.host}:{PostgreSQLConfig.port}"
os.environ['POSTGRESQL_URL'] = POSTGRESQL_URL

# MCP服务
MCP_SERVER = {
    "bing_search": {
        "transport": "http",
        "url": os.getenv("MCP_BING_SEARCH_URL", "http://localhost:8080/mcp"),
    }
}