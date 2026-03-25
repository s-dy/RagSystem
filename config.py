import os
from dataclasses import dataclass

from dotenv import load_dotenv

load_dotenv()


@dataclass
class LoggingConfig:
    """日志配置"""

    # 日志级别：DEBUG, INFO, WARNING, ERROR, CRITICAL
    log_level: str = os.getenv("LOG_LEVEL", "INFO")
    # 日志文件目录
    log_dir: str = os.getenv("LOG_DIR", "logs")
    # 是否启用文件日志
    enable_file_logging: bool = (
        os.getenv("ENABLE_FILE_LOGGING", "true").lower() == "true"
    )
    # 是否启用控制台日志
    enable_console_logging: bool = (
        os.getenv("ENABLE_CONSOLE_LOGGING", "true").lower() == "true"
    )
    # 单个日志文件最大字节数（默认 10MB）
    max_bytes: int = int(os.getenv("LOG_MAX_BYTES", str(10 * 1024 * 1024)))
    # 保留的备份文件数量
    backup_count: int = int(os.getenv("LOG_BACKUP_COUNT", "5"))


@dataclass
class RagSystemConfig:
    # 是否开启 RAG 评估（基于 ragas 指标）
    enable_eval: bool = False
    # 是否启用父子文档检索策略（子文档检索 → 父文档回溯）
    enable_parent_child_retrieval: bool = True
    # 重排序阈值：低于此分数的文档将被过滤（CrossEncoder 输出范围通常 0~1）
    reranker_threshold: float = float(os.getenv("RERANKER_THRESHOLD", "0.8"))
    # 文档相关性评分阈值：余弦相似度低于此值判定为不相关
    grader_threshold: float = float(os.getenv("GRADER_THRESHOLD", "0.5"))

    # 推理上下文最大字符数
    max_reasoning_chars: int = 5000

    # === 对话压缩相关配置 ===
    # 是否启用跨轮对话历史压缩
    enable_conversation_compress: bool = True
    # 最大对话轮数（超过则触发压缩，1 轮 = 1 条 Human + 1 条 AI）
    max_conversation_turns: int = 10
    # 最大对话 token 数（超过则触发压缩）
    max_conversation_tokens: int = 4000
    # 压缩时保留最近的轮数（不被压缩）
    keep_recent_turns: int = 3
    # 单次压缩的最大 token 数（超过则分批压缩）
    max_compress_tokens: int = 6000
    # 渐进式摘要触发间隔（每隔多少轮触发一次增量摘要）
    incremental_summary_interval: int = 5
    # 对话上下文窗口自适应的最大 token 预算
    max_context_tokens: int = 2000


@dataclass
class QueryEnhancementConfig:
    """查询增强配置"""

    # 同义改写提高召回
    paraphrase: bool = False
    # 基于专业水平改写
    formalize: bool = False
    # 扩展改写,扩展比较维度、步骤细节
    expand: bool = False
    # 查询分解,分解多步子问题
    enable_query_decomposition: bool = False
    # HyDE predict 生成假设答案辅助检索
    hyde_predict: bool = False

    # 性能限制
    max_enhanced_queries: int = 8


@dataclass
class MultimodalConfig:
    """多模态 RAG 配置"""

    # CLIP 模型路径，支持本地路径（如 /models/clip-vit-base-patch32）或 HuggingFace 模型名
    clip_model_path: str = os.getenv("CLIP_MODEL_PATH", "openai/clip-vit-base-patch32")
    # 图片检索相似度阈值（CLIP 内积分数，范围 0~1），低于此值的图片将被过滤
    image_score_threshold: float = float(os.getenv("IMAGE_SCORE_THRESHOLD", "0.25"))
    # 每次查询最多传入 VLM 的图片数量（防止超出 token 限制）
    max_images_per_query: int = int(os.getenv("MAX_IMAGES_PER_QUERY", "3"))
    # VLM 生成图片 Caption 时使用的模型（为空则跳过 Caption 生成）
    caption_model_name: str = os.getenv("CAPTION_MODEL_NAME", "")


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
os.environ["REDIS_URL"] = REDIS_URI


def get_postgresql_url() -> str:
    """动态构建 PostgreSQL 连接 URL，每次调用时从环境变量读取最新值，避免模块加载时固化。"""
    cfg = PostgreSQLConfig()
    # 密码可能包含特殊字符，使用 urllib.parse.quote 进行编码
    from urllib.parse import quote

    encoded_password = quote(cfg.password, safe="")
    return (
        f"postgresql://{cfg.user}:{encoded_password}@{cfg.host}:{cfg.port}/{cfg.dbname}"
    )


# 保留模块级变量以兼容可能直接导入 POSTGRESQL_URL 的其他模块
# 注意：此变量在模块加载时固化，新代码应改用 get_postgresql_url()
POSTGRESQL_URL = get_postgresql_url()
os.environ["POSTGRESQL_URL"] = POSTGRESQL_URL

# MCP服务
MCP_SERVER = {
    "bing_search": {
        "transport": "http",
        "url": os.getenv("MCP_BING_SEARCH_URL", "http://localhost:8080/mcp"),
    }
}
