"""
RAG 系统端到端评估脚本

流程：
1. 动态修改 config（enable_eval=False, enable_parent_child_retrieval=True, db_name=rag_system_test）
2. 创建 PostgreSQL 测试数据库
3. 使用 DataDBStorage.ingest 将 PDF 数据入库
4. 使用 LLM 生成单跳/多跳测试数据集 → qa_pair.json
5. 使用 Graph 运行 RAG 系统，RagEvaluator 评估
6. 生成测试报告，清理测试数据库
"""

import asyncio
import json
import os
import random
import sys
from datetime import datetime

import psycopg
from utils.environ import set_huggingface_hf_env
set_huggingface_hf_env()

# 确保项目根目录在 sys.path 中
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

# ──────────────────────────────────────────────
# 常量
# ──────────────────────────────────────────────
TEST_DB_NAME = "rag_system_test"
TEST_COLLECTION_NAME = "eval_test_collection"
TEST_PDF_PATH = os.path.join(PROJECT_ROOT, "tests", "test_data", "长安的荔枝 - 马伯庸.pdf")
QA_PAIR_PATH = os.path.join(PROJECT_ROOT, "tests", "qa_pair.json")
REPORT_PATH = os.path.join(PROJECT_ROOT, "tests", "eval_report.json")

# 测试集生成参数
SINGLE_HOP_COUNT = 1
MULTI_HOP_COUNT = 1


# ──────────────────────────────────────────────
# Step 1: 动态修改 config
# ──────────────────────────────────────────────
# 必须在 import config 之前设置环境变量，因为 config.py 中的 dataclass 默认值
# 使用 os.getenv() 在模块加载时一次性求值，之后修改环境变量不会生效。
os.environ["MILVUS_DB_NAME"] = TEST_DB_NAME
os.environ["MILVUS_COLLECTION_NAME"] = TEST_COLLECTION_NAME
os.environ["POSTGRES_DBNAME"] = TEST_DB_NAME


def patch_config():
    """动态修改 config 模块中的配置，指向测试数据库"""
    import config

    # 修改 RagSystemConfig 默认值（非 os.getenv 驱动的字段，直接改类属性即可）
    config.RagSystemConfig.enable_eval = False
    config.RagSystemConfig.enable_parent_child_retrieval = True

    # 重新生成 POSTGRESQL_URL 并更新环境变量（config.py 中的模块级变量也需要覆盖）
    new_url = (
        f"postgresql://{config.PostgreSQLConfig.user}:{config.PostgreSQLConfig.password}"
        f"@{config.PostgreSQLConfig.host}:{config.PostgreSQLConfig.port}/{TEST_DB_NAME}"
    )
    config.POSTGRESQL_URL = new_url
    os.environ["POSTGRESQL_URL"] = new_url

    print(f"✅ 配置已修改: db_name={TEST_DB_NAME}, POSTGRESQL_URL={new_url}")


# ──────────────────────────────────────────────
# Step 2: 创建 PostgreSQL 和 Milvus 测试数据库
# ──────────────────────────────────────────────
def create_test_databases():
    """复用项目中的模块级函数，创建 PostgreSQL 和 Milvus 测试数据库"""
    from src.services.storage.postgres_connector import ensure_postgres_database_exists
    from src.services.storage.milvus_client import ensure_milvus_database_exists

    ensure_postgres_database_exists()
    ensure_milvus_database_exists()

# ──────────────────────────────────────────────
# Step 3: 数据入库
# ──────────────────────────────────────────────
async def ingest_test_data():
    """使用 DataDBStorage 将 PDF 文档入库到测试数据库"""
    from src.services.data_load import DataDBStorage, IngestConfig

    ingest_config = IngestConfig(
        collection_name=TEST_COLLECTION_NAME,
        chunk_size=1024,
        chunk_overlap=128,
        use_parent_child=True,
    )

    storage = DataDBStorage(collection_name=TEST_COLLECTION_NAME)
    result = await storage.ingest(ingest_config, TEST_PDF_PATH)
    # 将配置写入 PostgreSQL（用于查询路由）
    from src.services.storage import PostgreSQLConnector
    PostgreSQLConnector().insert_knowledge_collection({
        "index": TEST_COLLECTION_NAME,
        "description": "测试知识库",
        "domain": "general",
        "keywords": [],
    })
    print(f"✅ 数据入库完成: {result}")
    return result


# ──────────────────────────────────────────────
# Step 4: 使用 LLM 生成测试数据集
# ──────────────────────────────────────────────
async def load_chunks_for_qa_generation():
    """加载文档并分块，用于生成测试数据集"""
    from src.services.data_load import DataDBStorage

    storage = DataDBStorage(collection_name=TEST_COLLECTION_NAME)
    chunks = await storage.load_data_and_chunk(TEST_PDF_PATH, chunk_size=2048, chunk_overlap=128)
    return chunks


async def generate_qa_dataset():
    """使用 LLM 随机抽取文档片段，生成单跳和多跳问答对"""
    from src.services.llm.models import get_qwen_model

    llm = get_qwen_model()
    chunks = await load_chunks_for_qa_generation()

    if len(chunks) < 2:
        raise ValueError(f"文档分块数量不足（{len(chunks)}），无法生成测试集")

    qa_pairs = []

    # 生成单跳问题：随机抽取单个片段
    single_hop_chunks = random.sample(chunks, min(SINGLE_HOP_COUNT, len(chunks)))
    for index, chunk in enumerate(single_hop_chunks):
        content = chunk.page_content.strip()
        if not content:
            continue

        prompt = f"""你是一个测试数据生成助手。请根据以下文本内容，生成一个**单跳问题**（即只需要从这段文本中直接找到答案的问题）。

文本内容：
{content}

请严格按照以下 JSON 格式返回（不要包含其他内容）：
{{
    "question": "你生成的问题",
    "reference": "从文本中提取的参考答案"
}}"""

        response = await llm.ainvoke(prompt)
        parsed = _parse_llm_json(response.content)
        if parsed:
            parsed["type"] = "single_hop"
            parsed["source_content"] = content
            qa_pairs.append(parsed)
            print(f"  ✅ 单跳问题 {index + 1}: {parsed['question']}")

    # 生成多跳问题：随机抽取两个片段组合
    multi_hop_count = min(MULTI_HOP_COUNT, len(chunks) // 2)
    multi_hop_indices = random.sample(range(len(chunks)), multi_hop_count * 2)
    for index in range(multi_hop_count):
        chunk_a = chunks[multi_hop_indices[index * 2]]
        chunk_b = chunks[multi_hop_indices[index * 2 + 1]]
        content_a = chunk_a.page_content.strip()
        content_b = chunk_b.page_content.strip()

        if not content_a or not content_b:
            continue

        prompt = f"""你是一个测试数据生成助手。请根据以下两段文本内容，生成一个**多跳问题**（即需要综合两段文本的信息才能回答的问题）。

文本片段一：
{content_a}

文本片段二：
{content_b}

请严格按照以下 JSON 格式返回（不要包含其他内容）：
{{
    "question": "你生成的需要综合两段内容才能回答的问题",
    "reference": "综合两段文本得出的参考答案"
}}"""

        response = await llm.ainvoke(prompt)
        parsed = _parse_llm_json(response.content)
        if parsed:
            parsed["type"] = "multi_hop"
            parsed["source_content"] = (content_a + " | " + content_b)
            qa_pairs.append(parsed)
            print(f"  ✅ 多跳问题 {index + 1}: {parsed['question']}")

    # 追加写入测试集：读取已有数据，合并新数据后保存
    existing_pairs = _load_existing_qa_pairs()
    merged_pairs = existing_pairs + qa_pairs
    with open(QA_PAIR_PATH, "w", encoding="utf-8") as file:
        json.dump(merged_pairs, file, ensure_ascii=False, indent=2)

    print(
        f"✅ 测试数据集已保存到 {QA_PAIR_PATH}，"
        f"新增 {len(qa_pairs)} 条，累计 {len(merged_pairs)} 条"
    )
    return merged_pairs


def _load_existing_qa_pairs() -> list[dict]:
    """读取已有的 qa_pair.json，如果文件不存在或内容无效则返回空列表"""
    if not os.path.exists(QA_PAIR_PATH):
        return []
    try:
        with open(QA_PAIR_PATH, "r", encoding="utf-8") as file:
            data = json.load(file)
        if isinstance(data, list):
            print(f"ℹ️  已加载 {len(data)} 条历史测试数据")
            return data
    except (json.JSONDecodeError, IOError) as error:
        print(f"⚠️ 读取已有 qa_pair.json 失败: {error}，将从空列表开始")
    return []


def _parse_llm_json(text: str) -> dict | None:
    """从 LLM 返回的文本中解析 JSON"""
    text = text.strip()
    # 去除 markdown 代码块标记
    if text.startswith("```"):
        lines = text.split("\n")
        lines = [line for line in lines if not line.strip().startswith("```")]
        text = "\n".join(lines)

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        # 尝试提取 JSON 部分
        start = text.find("{")
        end = text.rfind("}") + 1
        if start != -1 and end > start:
            try:
                return json.loads(text[start:end])
            except json.JSONDecodeError:
                pass
    print(f"  ⚠️ JSON 解析失败: {text}")
    return None


# ──────────────────────────────────────────────
# Step 5: 运行 RAG 系统并评估
# ──────────────────────────────────────────────
async def run_rag_and_evaluate(qa_pairs: list[dict]):
    """对每个测试问题调用 RAG 系统，收集结果并使用 RagEvaluator 评估"""
    from langchain_core.messages import HumanMessage
    from src.graph import Graph
    from config import RagSystemConfig
    from src.eval.ragas_eval import RagEvaluator, EvalSample

    rag_config = RagSystemConfig(
        enable_eval=False,
        enable_parent_child_retrieval=True,
    )
    graph = Graph(config=rag_config)
    evaluator = RagEvaluator()

    eval_samples = []
    rag_results = []

    for index, qa_item in enumerate(qa_pairs):
        question = qa_item["question"]
        reference = qa_item.get("reference", "")
        question_type = qa_item.get("type", "unknown")

        print(f"\n📝 [{index + 1}/{len(qa_pairs)}] ({question_type}) {question}")

        try:
            result = await graph.start(
                {"messages": [HumanMessage(content=question)]},
                {"configurable": {"thread_id": f"eval_{index}", "user_id": "eval_user"}},
            )

            answer = result.get("answer", "")
            retrieved_documents = result.get("retrieved_documents", [])

            rag_results.append({
                "question": question,
                "type": question_type,
                "answer": answer,
                "retrieved_documents": retrieved_documents,
                "reference": reference,
            })

            eval_samples.append(EvalSample(
                user_input=question,
                response=answer,
                retrieved_contexts=retrieved_documents,
                reference=reference if reference else None,
            ))

            print(f"  ✅ 回答: {answer}")

        except Exception as error:
            print(f"  ❌ RAG 调用失败: {error}")
            rag_results.append({
                "question": question,
                "type": question_type,
                "answer": "",
                "retrieved_documents": [],
                "reference": reference,
                "error": str(error),
            })

    # 使用 RagEvaluator 批量评估
    if eval_samples:
        print("\n🔍 开始 RAG 评估...")
        report = await evaluator.evaluate_batch(eval_samples)

        # 在 report 详情中补充问题类型
        for detail, qa_item in zip(report.details, qa_pairs):
            detail["type"] = qa_item.get("type", "unknown")

        RagEvaluator.print_report(report)
        return report, rag_results
    else:
        print("⚠️ 没有有效的评估样本")
        return None, rag_results


# ──────────────────────────────────────────────
# Step 6: 生成报告 & 清理测试数据库
# ──────────────────────────────────────────────
def generate_report(report, rag_results: list[dict]):
    """生成并保存评估报告"""
    from dataclasses import asdict

    report_data = {
        "timestamp": datetime.now().isoformat(),
        "test_data_source": TEST_PDF_PATH,
        "test_db_name": TEST_DB_NAME,
        "collection_name": TEST_COLLECTION_NAME,
        "summary": asdict(report) if report else None,
        "rag_results": rag_results,
    }

    # 按问题类型分组统计
    single_hop_details = [d for d in (report.details if report else []) if d.get("type") == "single_hop"]
    multi_hop_details = [d for d in (report.details if report else []) if d.get("type") == "multi_hop"]

    if single_hop_details:
        report_data["single_hop_summary"] = _compute_type_summary(single_hop_details)
    if multi_hop_details:
        report_data["multi_hop_summary"] = _compute_type_summary(multi_hop_details)

    with open(REPORT_PATH, "w", encoding="utf-8") as file:
        json.dump(report_data, file, ensure_ascii=False, indent=2)

    print(f"\n📊 评估报告已保存到: {REPORT_PATH}")


def _compute_type_summary(details: list[dict]) -> dict:
    """按问题类型计算平均分"""
    metrics = ["faithfulness", "answer_relevancy", "context_relevance", "context_recall"]
    summary = {"count": len(details)}
    for metric in metrics:
        values = [
            d["scores"][metric]
            for d in details
            if d.get("scores", {}).get(metric) is not None
        ]
        summary[f"avg_{metric}"] = sum(values) / len(values) if values else None
    return summary


def cleanup_test_databases():
    """清理测试数据库：删除 PostgreSQL 和 Milvus 的测试数据库"""
    _cleanup_postgres()
    _cleanup_milvus()


def _cleanup_postgres():
    """删除 PostgreSQL 测试数据库"""
    from config import PostgreSQLConfig

    conninfo = (
        f"host={PostgreSQLConfig.host} port={PostgreSQLConfig.port} "
        f"user={PostgreSQLConfig.user} password={PostgreSQLConfig.password} "
        f"dbname=postgres"
    )
    try:
        conn = psycopg.connect(conninfo, autocommit=True)
        cursor = conn.cursor()

        # 先断开所有到测试数据库的连接
        cursor.execute(f"""
            SELECT pg_terminate_backend(pg_stat_activity.pid)
            FROM pg_stat_activity
            WHERE pg_stat_activity.datname = '{TEST_DB_NAME}'
            AND pid <> pg_backend_pid()
        """)

        cursor.execute(f'DROP DATABASE IF EXISTS "{TEST_DB_NAME}"')
        print(f"✅ PostgreSQL 测试数据库 '{TEST_DB_NAME}' 已删除")
        cursor.close()
        conn.close()
    except Exception as error:
        print(f"⚠️ 删除 PostgreSQL 测试数据库失败: {error}")


def _cleanup_milvus():
    """删除 Milvus 测试数据库"""
    try:
        from pymilvus import connections, db, utility
        from config import MilvusConfig

        connections.connect(
            alias="cleanup",
            host=MilvusConfig.host,
            port=MilvusConfig.port,
            token=MilvusConfig.token,
        )

        # 切换到测试数据库，删除所有 collection
        existing_dbs = db.list_database(using="cleanup")
        if TEST_DB_NAME in existing_dbs:
            db.using_database(TEST_DB_NAME, using="cleanup")
            collections = utility.list_collections(using="cleanup")
            for collection_name in collections:
                utility.drop_collection(collection_name, using="cleanup")
                print(f"  🗑️ Milvus collection '{collection_name}' 已删除")

            # 切回 default 再删除测试数据库
            db.using_database("default", using="cleanup")
            db.drop_database(TEST_DB_NAME, using="cleanup")
            print(f"✅ Milvus 测试数据库 '{TEST_DB_NAME}' 已删除")
        else:
            print(f"ℹ️  Milvus 测试数据库 '{TEST_DB_NAME}' 不存在，跳过删除")

        connections.disconnect("cleanup")
    except Exception as error:
        print(f"⚠️ 删除 Milvus 测试数据库失败: {error}")


# ──────────────────────────────────────────────
# 主流程
# ──────────────────────────────────────────────
async def main():
    print("=" * 60)
    print("🚀 RAG 系统端到端评估")
    print("=" * 60)

    # Step 1: 修改配置
    print("\n📌 Step 1: 修改配置...")
    patch_config()

    # Step 2: 创建测试数据库（PostgreSQL + Milvus）
    print("\n📌 Step 2: 创建测试数据库...")
    create_test_databases()

    # Step 3: 数据入库
    print("\n📌 Step 3: 数据入库...")
    await ingest_test_data()

    # Step 4: 生成测试数据集
    print("\n📌 Step 4: 使用 LLM 生成测试数据集...")
    qa_pairs = await generate_qa_dataset()

    if not qa_pairs:
        print("❌ 未能生成任何测试数据，终止评估")
        cleanup_test_databases()
        return

    # Step 5: 运行 RAG 系统并评估
    print("\n📌 Step 5: 运行 RAG 系统并评估...")
    report, rag_results = await run_rag_and_evaluate(qa_pairs[:1])

    # Step 6: 生成报告并清理
    print("\n📌 Step 6: 生成报告并清理测试数据库...")
    generate_report(report, rag_results)
    cleanup_test_databases()

    print("\n" + "=" * 60)
    print("🎉 评估完成！")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
