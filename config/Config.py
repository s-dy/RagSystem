from dataclasses import dataclass, field
import os


@dataclass
class QueryEnhancementConfig:
    """查询增强配置"""
    # 同义改写提高召回
    paraphrase:bool = False
    # 基于专业水平改写
    formalize:bool = False
    # 扩展改写,扩展比较维度、步骤细节
    expand:bool = True
    # 查询分解
    enable_query_decomposition: bool = True
    # HyDE predict 生成假设答案辅助检索
    hyde_predict:bool = True

    # 性能限制
    max_enhanced_queries: int = 8


@dataclass
class MilvusConfig:
    collection_name: str = "default"
    host: str = "localhost"
    port: int = 19530
    db_name: str = "hybridRagSystem"
    token: str = "root:Milvus"


REDIS_URI = "redis://localhost:6379"
os.environ['REDIS_URL'] = REDIS_URI