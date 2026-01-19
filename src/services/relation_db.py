from dataclasses import dataclass

import pymysql
import json
from typing import List, Dict

from src.monitoring.logger import monitor_task_status
from utils.decortor import singleton


@dataclass
class MySQLConfig:
    host:str = "localhost"
    port:int = 3306
    user:str = "root"
    password:str = "root123456"
    db:str = "hybridRagSystem"
    charset:str = "utf8mb4"
    autocommit:bool = True

@singleton
class MySQLConnector:
    def __init__(self, config: MySQLConfig=None):
        if config is None:
            config = MySQLConfig()
        self.config = config
        self.conn = pymysql.connect(**config.__dict__)

    def execute(self, query: str, params=None):
        with self.conn.cursor() as cursor:
            cursor.execute(query, params)
            if query.strip().upper().startswith('SELECT'):
                return cursor.fetchall()
            else:
                return cursor.rowcount

    def create_knowledge_table(self):
        sql = """
        CREATE TABLE IF NOT EXISTS knowledge_collections (
            id INT AUTO_INCREMENT PRIMARY KEY,
            collection_name VARCHAR(255) NOT NULL UNIQUE,
            description TEXT NOT NULL,
            domain VARCHAR(100) NOT NULL,
            keywords JSON NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
        """
        self.execute(sql)
        print("✅ Table 'knowledge_collections' created or exists.")

    def insert_knowledge_collection(self, collection: Dict):
        """插入单个知识库配置"""
        sql = """
        INSERT INTO knowledge_collections (collection_name, description, domain, keywords)
        VALUES (%s, %s, %s, %s)
        ON DUPLICATE KEY UPDATE
            description = VALUES(description),
            domain = VALUES(domain),
            keywords = VALUES(keywords)
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
                "keywords": json.loads(row[3])
            })

        monitor_task_status('从数据库读取的知识库配置:',result)
        return result

    def close(self):
        self.conn.close()


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
    db = MySQLConnector()
    # 3. 创建表
    db.create_knowledge_table()
    # 4. 写入配置
    db.insert_all_collections(KNOWLEDGE_CONFIGS)

    # 5. 验证读取
    print("\n📚 从数据库读取的知识库配置:")
    configs_from_db = db.get_all_collections()
    print(configs_from_db)

    db.close()
