import json
from typing import List, Dict

import psycopg
from psycopg_pool import ConnectionPool

from config import PostgreSQLConfig
from src.observability.logger import get_logger
from utils.decorator import singleton

logger = get_logger(__name__)


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
            logger.info(f"[PostgreSQL] 数据库自动创建: {config.dbname}")
        cursor.close()
        conn.close()
    except Exception as error:
        logger.warning(f"[PostgreSQL] 数据库检测/创建失败: {error}")


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
            f"dbname={config.dbname} "
            f"options='-c timezone={config.timezone}'"
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
            self._ensure_timestamp_with_timezone('parent_documents', 'created_at')
            return

        sql = """
        CREATE TABLE IF NOT EXISTS parent_documents (
            parent_id VARCHAR(512) PRIMARY KEY,
            content TEXT NOT NULL,
            metadata JSONB,
            created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP
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

    def _ensure_timestamp_with_timezone(self, table_name: str, column_name: str):
        sql = """
        SELECT data_type FROM information_schema.columns
        WHERE table_schema = 'public' AND table_name = %s AND column_name = %s
        """
        rows = self.execute(sql, (table_name, column_name))
        if not rows:
            return

        data_type = rows[0][0]
        if data_type == 'timestamp without time zone':
            self.execute(
                f"ALTER TABLE {table_name} ALTER COLUMN {column_name} TYPE TIMESTAMPTZ USING {column_name} AT TIME ZONE 'UTC'"
            )

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

    def _get_table_columns(self, table_name: str) -> set[str]:
        sql = """
        SELECT column_name FROM information_schema.columns
        WHERE table_schema = 'public' AND table_name = %s
        """
        rows = self.execute(sql, (table_name,))
        return {row[0] for row in rows} if rows else set()

    def _ensure_knowledge_collection_columns(self):
        columns = self._get_table_columns('knowledge_collections')

        # 为旧 schema 添加缺失列
        if 'internal_name' not in columns:
            self.execute(
                "ALTER TABLE knowledge_collections ADD COLUMN internal_name VARCHAR(255)"
            )
            if 'collection_name' in columns:
                self.execute(
                    "UPDATE knowledge_collections SET internal_name = collection_name WHERE internal_name IS NULL OR internal_name = ''"
                )
            elif 'index' in columns:
                self.execute(
                    "UPDATE knowledge_collections SET internal_name = \"index\" WHERE internal_name IS NULL OR internal_name = ''"
                )

        if 'display_name' not in columns:
            self.execute(
                "ALTER TABLE knowledge_collections ADD COLUMN display_name VARCHAR(255)"
            )
            if 'internal_name' in columns:
                self.execute(
                    "UPDATE knowledge_collections SET display_name = internal_name WHERE display_name IS NULL OR display_name = ''"
                )
            elif 'collection_name' in columns:
                self.execute(
                    "UPDATE knowledge_collections SET display_name = collection_name WHERE display_name IS NULL OR display_name = ''"
                )
            elif 'index' in columns:
                self.execute(
                    "UPDATE knowledge_collections SET display_name = \"index\" WHERE display_name IS NULL OR display_name = ''"
                )

        if 'description' not in columns:
            self.execute(
                "ALTER TABLE knowledge_collections ADD COLUMN description TEXT NOT NULL DEFAULT ''"
            )

        if 'domain' not in columns:
            self.execute(
                "ALTER TABLE knowledge_collections ADD COLUMN domain VARCHAR(100) NOT NULL DEFAULT 'general'"
            )

        if 'keywords' not in columns:
            self.execute(
                "ALTER TABLE knowledge_collections ADD COLUMN keywords JSONB NOT NULL DEFAULT '[]'"
            )

        if 'created_at' in columns:
            self._ensure_timestamp_with_timezone('knowledge_collections', 'created_at')

        # 确保 internal_name 的唯一索引存在，以支持 ON CONFLICT
        self.execute(
            "CREATE UNIQUE INDEX IF NOT EXISTS idx_knowledge_collections_internal_name ON knowledge_collections (internal_name)"
        )

        # 填充默认值以满足新 schema 的非空约束
        if 'internal_name' in columns and 'index' in columns:
            self.execute(
                "UPDATE knowledge_collections SET internal_name = \"index\" WHERE (internal_name IS NULL OR internal_name = '') AND \"index\" IS NOT NULL"
            )
        self.execute(
            "UPDATE knowledge_collections SET display_name = internal_name WHERE display_name IS NULL OR display_name = ''"
        )

    def create_knowledge_table(self):
        sql = """
        CREATE TABLE IF NOT EXISTS knowledge_collections (
            id SERIAL PRIMARY KEY,
            internal_name VARCHAR(255) NOT NULL UNIQUE,
            display_name VARCHAR(255) NOT NULL,
            description TEXT NOT NULL,
            domain VARCHAR(100) NOT NULL,
            keywords JSONB NOT NULL,
            created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP
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
            self._ensure_knowledge_collection_columns()
            print("✅ Table 'knowledge_collections' already exists.")

    def insert_knowledge_collection(self, collection: Dict):
        """插入单个知识库配置"""
        columns = self._get_table_columns('knowledge_collections')
        insert_columns = []
        params = []

        internal = collection.get('internal_name') or collection.get('index') or 'default'
        display = collection.get('display_name') or internal
        description = collection.get('description', '')
        domain = collection.get('domain', 'default')
        keywords_json = json.dumps(collection.get('keywords', []), ensure_ascii=False)

        if 'collection_name' in columns:
            insert_columns.append('collection_name')
            params.append(internal)
        if 'internal_name' in columns:
            insert_columns.append('internal_name')
            params.append(internal)
        if 'display_name' in columns:
            insert_columns.append('display_name')
            params.append(display)
        if 'description' in columns:
            insert_columns.append('description')
            params.append(description)
        if 'domain' in columns:
            insert_columns.append('domain')
            params.append(domain)
        if 'keywords' in columns:
            insert_columns.append('keywords')
            params.append(keywords_json)

        if not insert_columns:
            raise ValueError('knowledge_collections table has no supported columns')

        placeholders = ', '.join(['%s'] * len(insert_columns))
        sql = f"""
        INSERT INTO knowledge_collections ({', '.join(insert_columns)})
        VALUES ({placeholders})
        ON CONFLICT (internal_name) DO UPDATE SET
            display_name = EXCLUDED.display_name,
            description = EXCLUDED.description,
            domain = EXCLUDED.domain,
            keywords = EXCLUDED.keywords
        """

        self.execute(sql, tuple(params))
        logger.debug(f"[PostgreSQL] 知识库配置已更新: {internal}")

    def delete_knowledge_collection(self, collection_name: str):
        """删除指定知识库配置（使用 internal_name）"""
        sql = "DELETE FROM knowledge_collections WHERE internal_name = %s"
        self.execute(sql, (collection_name,))
        logger.info(f"[PostgreSQL] 知识库配置已删除: {collection_name}")

    def get_all_collections(self) -> List[Dict]:
        """从数据库读取所有知识库配置"""
        columns = self._get_table_columns('knowledge_collections')
        if 'index' in columns:
            rows = self.execute(
                "SELECT internal_name, display_name, description, domain, keywords, \"index\" FROM knowledge_collections"
            )
        else:
            rows = self.execute(
                "SELECT internal_name, display_name, description, domain, keywords FROM knowledge_collections"
            )

        result = []
        for row in rows:
            if 'index' in columns:
                internal_name = row[0] or row[5]
                display_name = row[1] or internal_name or row[5]
                description = row[2]
                domain = row[3]
                keywords = row[4] or []
            else:
                internal_name = row[0]
                display_name = row[1] or internal_name
                description = row[2]
                domain = row[3]
                keywords = row[4] or []

            if not internal_name:
                continue

            result.append({
                "index": internal_name,
                "internal_name": internal_name,
                "display_name": display_name,
                "description": description,
                "domain": domain,
                "keywords": keywords,
            })

        logger.debug(f"[PostgreSQL] 知识库配置读取完成: count={len(result)}")
        return result

    def get_internal_name_by_display_name(self, display_name: str) -> str | None:
        """根据 display_name 查询对应的 internal_name"""
        if not display_name:
            return None
        sql = "SELECT internal_name FROM knowledge_collections WHERE display_name = %s LIMIT 1"
        rows = self.execute(sql, (display_name,))
        return rows[0][0] if rows else None

    def close(self):
        self.pool.close()