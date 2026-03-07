import json
from typing import List, Dict

import psycopg
from psycopg_pool import ConnectionPool

from config import PostgreSQLConfig
from src.observability.logger import monitor_task_status
from utils.decorator import singleton


def ensure_postgres_database_exists(config: PostgreSQLConfig = None):
    """检测目标 PostgreSQL 数据库是否存在，不存在则自动创建"""
    if config is None:
        config = PostgreSQLConfig()
    admin_conninfo = (
        f"host={config.host} port={config.port} "
        f"user={config.user} password={config.password} "
        f"dbname=postgres"
    )
    try:
        conn = psycopg.connect(admin_conninfo, autocommit=True)
        cursor = conn.cursor()
        cursor.execute(
            "SELECT 1 FROM pg_database WHERE datname = %s", (config.dbname,)
        )
        if not cursor.fetchone():
            cursor.execute(f'CREATE DATABASE "{config.dbname}"')
            monitor_task_status(f"PostgreSQL 数据库 '{config.dbname}' 不存在，已自动创建")
        cursor.close()
        conn.close()
    except Exception as error:
        monitor_task_status(
            f"检测/创建 PostgreSQL 数据库 '{config.dbname}' 失败: {error}",
            level="WARNING",
        )


@singleton
class PostgreSQLConnector:
    def __init__(self, config: PostgreSQLConfig = None):
        if config is None:
            config = PostgreSQLConfig()
        self.config = config

        # 在创建连接池之前，确保目标数据库存在
        ensure_postgres_database_exists(config)

        conninfo = (
            f"host={config.host} port={config.port} "
            f"user={config.user} password={config.password} "
            f"dbname={config.dbname}"
        )

        self.pool = ConnectionPool(
            conninfo=conninfo,
            min_size=2,
            max_size=10,
            max_idle=300,
            max_lifetime=3600,
            reconnect_timeout=60,
            kwargs={"autocommit": config.autocommit},
        )

        # 确保必要的表结构存在
        self._ensure_tables()

    def _ensure_tables(self):
        """确保必要的表结构存在，在初始化时自动调用"""
        self.create_knowledge_table()
        self.create_parent_documents_table()

    def execute(self, query: str, params=None):
        """从连接池获取连接执行查询，连接自动归还"""
        with self.pool.connection() as conn:
            with conn.cursor() as cursor:
                cursor.execute(query, params)
                if query.strip().upper().startswith('SELECT'):
                    return cursor.fetchall()
                else:
                    return cursor.rowcount

    def create_parent_documents_table(self):
        """创建父文档存储表"""
        check_sql = """
        SELECT EXISTS (
           SELECT FROM information_schema.tables
           WHERE table_schema = 'public'
           AND table_name = 'parent_documents'
        );
        """
        result = self.execute(check_sql)
        if result and result[0][0]:
            return

        sql = """
        CREATE TABLE IF NOT EXISTS parent_documents (
            parent_id VARCHAR(512) PRIMARY KEY,
            content TEXT NOT NULL,
            metadata JSONB,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        CREATE INDEX IF NOT EXISTS idx_parent_documents_parent_id ON parent_documents (parent_id);
        """
        self.execute(sql)

    def batch_insert_parent_documents(self, parent_store: dict):
        """批量插入父文档

        Args:
            parent_store: {parent_id: Document} 映射
        """
        if not parent_store:
            return
        sql = """
        INSERT INTO parent_documents (parent_id, content, metadata)
        VALUES (%s, %s, %s)
        ON CONFLICT (parent_id) DO UPDATE SET
            content = EXCLUDED.content,
            metadata = EXCLUDED.metadata
        """
        for parent_id, doc in parent_store.items():
            metadata_json = json.dumps(doc.metadata, ensure_ascii=False, default=str)
            self.execute(sql, (parent_id, doc.page_content, metadata_json))

    def get_parent_documents_by_ids(self, parent_ids: list[str]) -> dict[str, str]:
        """根据 parent_id 列表批量查询父文档内容

        Returns:
            {parent_id: content} 映射
        """
        if not parent_ids:
            return {}
        placeholders = ",".join(["%s"] * len(parent_ids))
        sql = f"SELECT parent_id, content FROM parent_documents WHERE parent_id IN ({placeholders})"
        rows = self.execute(sql, tuple(parent_ids))
        return {row[0]: row[1] for row in rows} if rows else {}

    def create_knowledge_table(self):
        sql = """
        CREATE TABLE IF NOT EXISTS knowledge_collections (
            id SERIAL PRIMARY KEY,
            collection_name VARCHAR(255) NOT NULL UNIQUE,
            description TEXT NOT NULL,
            domain VARCHAR(100) NOT NULL,
            keywords JSONB NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        """
        # 检查表是否存在
        check_table_sql = """
        SELECT EXISTS (
           SELECT FROM information_schema.tables 
           WHERE table_schema = 'public' 
           AND table_name = 'knowledge_collections'
        );
        """
        result = self.execute(check_table_sql)
        table_exists = result[0][0] if result else False
        
        if not table_exists:
            self.execute(sql)
            print("✅ Table 'knowledge_collections' created.")
        else:
            print("✅ Table 'knowledge_collections' already exists.")

    def insert_knowledge_collection(self, collection: Dict):
        """插入单个知识库配置"""
        sql = """
        INSERT INTO knowledge_collections (collection_name, description, domain, keywords)
        VALUES (%s, %s, %s, %s)
        ON CONFLICT (collection_name) DO UPDATE SET
            description = EXCLUDED.description,
            domain = EXCLUDED.domain,
            keywords = EXCLUDED.keywords
        """
        # 将 keywords 列表转为 JSON 字符串
        keywords_json = json.dumps(collection['keywords'], ensure_ascii=False)
        self.execute(sql, (
            collection['index'],
            collection['description'],
            collection['domain'],
            keywords_json
        ))
        print(f"💾 Inserted/Updated collection: {collection['index']}")

    def insert_all_collections(self, collections: List[Dict]):
        """批量插入所有知识库配置"""
        for col in collections:
            self.insert_knowledge_collection(col)

    def get_all_collections(self) -> List[Dict]:
        """从数据库读取所有知识库配置"""
        rows = self.execute("SELECT collection_name, description, domain, keywords FROM knowledge_collections")
        result = []
        for row in rows:
            result.append({
                "index": row[0],
                "description": row[1],
                "domain": row[2],
                "keywords": row[3]
            })

        monitor_task_status('从数据库读取的知识库配置:',result)
        return result

    def close(self):
        self.pool.close()


if __name__ == '__main__':
    # 1. 定义知识库配置（你的原始数据）
    KNOWLEDGE_CONFIGS = [
        {
            "index": "hybridRag_news",
            "description": "新闻文章集合",
            "domain": "news",
            "keywords": ["科技", "生活", "社会"]
        },
        # {
        #     "index": "cybersecurity",
        #     "description": "网络安全知识库，涵盖攻击防护、加密技术、安全协议等",
        #     "domain": "technology",
        #     "keywords": ["安全", "黑客", "加密", "漏洞"]
        # },
        # {
        #     "index": "medical",
        #     "description": "医学健康知识库，包含疾病症状、治疗方法、药物信息",
        #     "domain": "medical",
        #     "keywords": ["疾病", "治疗", "药物", "健康"]
        # }
    ]

    # 2. 初始化数据库
    db = PostgreSQLConnector()
    # 3. 创建表
    # db.create_knowledge_table()
    # 4. 写入配置
    # db.insert_all_collections(KNOWLEDGE_CONFIGS)

    # 5. 验证读取
    print("\n📚 从数据库读取的知识库配置:")
    configs_from_db = db.get_all_collections()
    print(configs_from_db)

    db.close()
