# HybridRAG - 混合检索增强生成系统

一个基于 LangGraph 构建的高性能 RAG（Retrieval-Augmented Generation）系统，支持多策略检索、查询增强、对话记忆和流式响应。

## ✨ 核心特性

- **🔍 混合检索策略**：结合向量检索、BM25 关键词检索和外部搜索，实现高召回率
- **🚀 查询增强**：支持查询扩展、分解、重写和 HyDE 预测，并行化处理显著降低延迟
- **📊 RAG-Fusion**：多路检索结果融合，加权 RRF 重排序
- **🧠 对话记忆**：支持长对话压缩、用户画像、渐进式摘要
- **⚡ 流式响应**：SSE 实时推送生成进度，支持心跳保活
- **📝 结构化日志**：JSON 格式日志 + 请求 ID 追踪，便于问题定位
- **🔧 多知识库管理**：支持创建、删除、切换多个知识库

## 🏗️ 系统架构

```
┌─────────────────────────────────────────────────────────────────┐
│                         用户请求                                 │
└─────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│                      TaskAdapter（任务适配）                     │
│              分析查询意图，选择合适的处理流程                      │
└─────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│                      QueryEnhancer（查询增强）                   │
│    ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐      │
│    │ 同义改写  │  │ 查询扩展  │  │ 查询分解  │  │ HyDE预测  │      │
│    └──────────┘  └──────────┘  └──────────┘  └──────────┘      │
│                    RunnableParallel 并行执行                      │
└─────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│                        Router（智能路由）                        │
│              基于 LLM 语义理解，选择检索策略                       │
└─────────────────────────────────────────────────────────────────┘
                                │
                ┌───────────────┼───────────────┐
                ▼               ▼               ▼
        ┌───────────┐   ┌───────────┐   ┌───────────┐
        │ 向量检索   │   │ BM25检索  │   │ 外部搜索   │
        │ (Milvus)  │   │ (全文)    │   │ (Bing)    │
        └───────────┘   └───────────┘   └───────────┘
                │               │               │
                └───────────────┼───────────────┘
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│                    RAG-Fusion（结果融合）                        │
│              加权 RRF 重排序，CrossEncoder 精排                   │
└─────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│                      Generator（答案生成）                       │
│              基于 LangGraph 状态机，支持流式输出                   │
└─────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│                      Memory（对话记忆）                          │
│         对话历史压缩 │ 用户画像 │ 渐进式摘要                       │
└─────────────────────────────────────────────────────────────────┘
```

## 🚀 快速开始

### 环境要求

- Python 3.10+
- Milvus 2.3+
- PostgreSQL 14+
- Redis 6.0+
- Ollama（本地 Embedding）

### 安装依赖

```bash
pip install -r requirements.txt
```

### 配置环境变量

创建 `.env` 文件：

```env
# LLM 配置
QWEN_MODEL_NAME=qwen-plus
QWEN_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1
QWEN_API_KEY=your_api_key

# Milvus 配置
MILVUS_HOST=localhost
MILVUS_PORT=19530
MILVUS_DB_NAME=hybridRagSystem
MILVUS_TOKEN=root:Milvus

# PostgreSQL 配置
POSTGRES_HOST=localhost
POSTGRES_PORT=5432
POSTGRES_USER=postgres
POSTGRES_PASSWORD=your_password
POSTGRES_DBNAME=hybridragsystem

# Redis 配置
REDIS_URI=redis://localhost:6379

# 日志配置
LOG_LEVEL=INFO
LOG_DIR=logs
ENABLE_FILE_LOGGING=true
ENABLE_CONSOLE_LOGGING=true

# RAG 配置
RERANKER_THRESHOLD=0.8
GRADER_THRESHOLD=0.5

# MCP 服务
MCP_BING_SEARCH_URL=http://localhost:8080/mcp

# Huggingface Models Path
HF_MODELS_PATH=/.cache/huggingface/hub
```

### 启动服务

```bash
python main.py
```

服务将在 `http://0.0.0.0:8000` 启动。

## 📁 项目结构

```
hybridRag/
├── config.py                 # 配置管理
├── main.py                   # 应用入口，日志初始化
├── server.py                 # FastAPI 服务端点
├── src/
│   ├── core/
│   │   ├── memory_manager.py # 对话记忆管理
│   │   ├── tools_pool.py     # 工具池
│   │   ├── adapter.py        # 任务适配器
│   │   └── exceptions.py     # 自定义异常
│   ├── node/
│   │   ├── generate/         # 答案生成节点
│   │   ├── retrieval/        # 检索节点
│   │   └── route/            # 路由节点
│   ├── services/
│   │   ├── llm/              # LLM 模型封装
│   │   ├── embedding/        # Embedding 模型
│   │   ├── storage/          # 存储服务（PostgreSQL, Milvus）
│   │   ├── data_load/        # 数据加载与分块
│   │   ├── cross_encoder_ranker.py  # 重排序服务
│   │   ├── time_transformer.py      # 时间解析
│   │   ├── grade_model.py           # 文档评分
│   │   └── task_analyzer.py         # 任务分析
│   ├── observability/
│   │   ├── logger.py         # 结构化日志系统
│   │   └── langfuse_monitor.py # Langfuse 监控集成
│   ├── eval/
│   │   └── ragas_eval.py     # RAG 评估
│   └── graph.py              # LangGraph 状态机
├── frontend/                 # 前端界面
├── logs/                     # 日志文件
│   ├── app.log              # 全量日志（JSON）
│   └── error.log            # 错误日志（JSON）
└── tests/                    # 测试用例
```

## 🔌 API 接口

### 对话接口

| 方法   | 路径                 | 描述        |
|------|--------------------|-----------|
| POST | `/api/chat`        | 非流式对话     |
| POST | `/api/chat/stream` | 流式对话（SSE） |

### 知识库管理

| 方法     | 路径                                    | 描述          |
|--------|---------------------------------------|-------------|
| GET    | `/api/knowledge/collections`          | 列出所有知识库     |
| DELETE | `/api/knowledge/collections/{name}`   | 删除知识库       |
| POST   | `/api/knowledge/upload`               | 上传文档（新建知识库） |
| GET    | `/api/knowledge/ingest-status/{name}` | 查询入库状态      |
| GET    | `/api/knowledge/documents`            | 列出文档        |
| DELETE | `/api/knowledge/documents`            | 删除文档        |

### 会话管理

| 方法     | 路径                               | 描述     |
|--------|----------------------------------|--------|
| GET    | `/api/conversations`             | 列出所有会话 |
| GET    | `/api/conversations/{thread_id}` | 获取会话详情 |
| DELETE | `/api/conversations/{thread_id}` | 删除会话   |

## ⚙️ 配置说明

### 日志配置

| 环境变量                     | 默认值        | 描述         |
|--------------------------|------------|------------|
| `LOG_LEVEL`              | `INFO`     | 日志级别       |
| `LOG_DIR`                | `logs`     | 日志目录       |
| `ENABLE_FILE_LOGGING`    | `true`     | 启用文件日志     |
| `ENABLE_CONSOLE_LOGGING` | `true`     | 启用控制台日志    |
| `LOG_MAX_BYTES`          | `10485760` | 单文件最大 10MB |
| `LOG_BACKUP_COUNT`       | `5`        | 备份文件数量     |

### RAG 配置

| 配置项                             | 默认值     | 描述        |
|---------------------------------|---------|-----------|
| `enable_eval`                   | `false` | 启用 RAG 评估 |
| `enable_parent_child_retrieval` | `true`  | 父子文档检索    |
| `reranker_threshold`            | `0.8`   | 重排序过滤阈值   |
| `grader_threshold`              | `0.5`   | 文档相关性阈值   |
| `max_conversation_turns`        | `10`    | 最大对话轮数    |

## 📊 日志系统

系统采用结构化日志，支持：

- **JSON 格式**：便于日志收集和分析
- **请求 ID 追踪**：通过 `thread_id` 追踪完整请求链路
- **错误日志分离**：`error.log` 单独记录 ERROR 及以上级别
- **人类可读格式**：控制台输出友好格式

### 使用示例

```python
from src.observability.logger import get_logger, set_request_id

# 设置请求 ID
set_request_id("thread_123")

# 获取 logger
logger = get_logger(__name__)

# 记录日志
logger.info("[NodeName] 操作描述: key=value")
```

## 🔍 核心模块详解

### 查询增强（QueryEnhancer）

使用 LangChain 的 LCEL 表达式，将多个增强任务构造为 RunnableSerializable，使用 RunnableParallel 并行化执行：

- **扩展**：添加元数据、上下文或相关术语扩展原始查询
- **分解**：将复杂查询分解为多个子问题
- **重写**：优化查询使其更清晰、明确
- **HyDE 预测**：生成假设性文档辅助检索

**性能对比**：

- 同步请求：平均 4s
- 并行化后：平均 1~1.5s

### 智能路由（Router）

基于 LLM 语义理解，将用户查询路由到最合适的检索策略：

- 内部知识库检索
- 外部搜索引擎
- 混合检索

### 检索融合（RAG-Fusion）

多路检索结果融合技术：

- **加权 RRF**：根据检索器可靠性赋予不同权重
- **CrossEncoder 精排**：对融合结果进行精细化重排序

### 对话记忆（Memory）

- **对话历史记忆**：多轮对话历史辅助 Query 生成
- **用户画像**：抽象用户中长期兴趣指导检索
- **检索反馈**：记住已检索内容，避免重复召回

### 文档分块（Chunk）

支持多种分块策略，针对不同文档类型优化：

- **递归分块**：使用中文优化分隔符（句号、问号、感叹号等），适用于 PDF、DOCX 等非结构化文档
- **Markdown 结构化分块**：按标题层级（h1-h4）切分，保留文档结构信息
- **父子文档分块**：大 chunk 作为上下文（1500 字符），小 chunk 用于向量检索（400 字符），提升检索精度

**使用示例**：

```python
from src.services.data_load.chunk import ChunkHandler

handler = ChunkHandler()

# 递归分块
chunks = handler.recursive_chunk(documents, chunk_size=1024, chunk_overlap=128)

# Markdown 结构化分块
md_chunks = handler.markdown_chunk(markdown_documents)

# 父子文档分块（支持普通文档和 Markdown）
parent_store, child_docs = handler.parent_child_chunk(documents)
md_parent_store, md_child_docs = handler.markdown_parent_child_chunk(md_documents)
```

## 🧪 评估

系统支持基于 Ragas 指标的 RAG 评估：

```python
import asyncio
from src.eval.ragas_eval import RagEvaluator, EvalSample


async def main():
    evaluator = RagEvaluator()

    # 创建评估样本
    samples = [
        EvalSample(
            user_input="什么是机器学习？",
            response="机器学习是人工智能的一个分支...",
            retrieved_contexts=["机器学习是AI的子领域...", "机器学习包括监督学习..."],
            reference="机器学习是人工智能的一个分支，通过数据训练模型。"  # 可选
        )
    ]

    # 批量评估
    report = await evaluator.evaluate_batch(samples)
    RagEvaluator.print_report(report)


asyncio.run(main())
```

**评估指标**：

- **忠实度 (Faithfulness)**：回答是否忠实于检索上下文
- **答案相关性 (Answer Relevancy)**：回答与问题的匹配程度
- **上下文相关性 (Context Relevance)**：检索内容与问题的相关性
- **上下文召回率 (Context Recall)**：检索内容是否覆盖参考答案

## 📈 性能优化

### 查询增强并行化

通过 `RunnableParallel` 并行执行多个增强任务，将单个请求耗时从 **4s** 降低到 **1~1.5s**。

### 缓存机制

- 高频查询缓存
- 分布式检索并行化

## 🤝 贡献

欢迎提交 Issue 和 Pull Request。

## 📄 License

MIT License
