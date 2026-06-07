# HybridRAG - 混合检索增强生成系统

一个基于 LangGraph 构建的高性能 RAG（Retrieval-Augmented Generation）系统，支持多策略检索、查询增强、对话记忆和流式响应。

## ✨ 核心特性

- **🔍 混合检索策略**：结合向量检索、BM25 关键词检索和外部搜索，实现高召回率
- **🚀 查询增强**：支持查询扩展、分解、重写和 HyDE 预测，并行化处理显著降低延迟
- **📊 RAG-Fusion**：多路检索结果融合，加权 RRF 重排序 + CrossEncoder 精排
- **🧠 对话记忆**：支持长对话压缩、用户画像、渐进式摘要、自适应上下文窗口
- **⚡ 流式响应**：SSE 实时推送生成进度，支持心跳保活和前端取消
- **📝 结构化日志**：JSON 格式日志 + 请求 ID 追踪，便于问题定位
- **🔧 多知识库管理**：支持创建、删除、切换多个知识库
- **🖼️ 多模态 RAG**：CLIP 跨模态图片检索 + VLM 图文联合生成，支持表格提取与增强
- **🔄 多跳推理**：自动分解复杂问题为子问题队列，逐一检索后综合生成最终答案
- **🎯 智能评分**：基于 embedding 相似度的文档相关性评分，支持重试机制和兜底策略
- **📎 引用溯源**：自动追踪每个子问题的来源文档编号，生成完整的引用链

## 🏗️ 系统架构

```
┌─────────────────────────────────────────────────────────────────┐
│                         用户请求                                 │
└─────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│              retrieve_or_respond（统一入口）                     │
│    任务分析 | 对话压缩 | 渐进式摘要 | 判断是否需要检索            │
└─────────────────────────────────────────────────────────────────┘
                                │
                    ┌───────────┴───────────┐
                    │                       │
              need_retrieval=false   need_retrieval=true
                    │                       │
                    ▼                       ▼
              ┌─────────┐      ┌──────────────────────┐
              │  final  │      │ prepare_next_step     │
              └─────────┘      │ 子问题分解/准备       │
                               └──────────────────────┘
                                        │
                                        ▼
┌─────────────────────────────────────────────────────────────────┐
│           enhance_and_route_current（查询增强+路由）             │
│    ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐      │
│    │ 同义改写  │  │ 查询扩展  │  │ 查询分解  │  │ HyDE预测  │      │
│    └──────────┘  └──────────┘  └──────────┘  └──────────┘      │
│                    RunnableParallel 并行执行                      │
│                              │                                   │
│                              ▼                                   │
│                    QueryRouter 多知识库路由                       │
└─────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│                fusion_retrieve（融合检索）                       │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐          │
│  │ 向量检索      │  │ BM25检索     │  │ 外部搜索      │          │
│  │ (Milvus)     │  │ (全文)       │  │ (Bing MCP)   │          │
│  └──────────────┘  └──────────────┘  └──────────────┘          │
│         │                 │                 │                   │
│         └─────────────────┼─────────────────┘                   │
│                           ▼                                     │
│                  近似去重 (embedding 相似度)                      │
│                           ▼                                     │
│              CrossEncoder 重排序 + 阈值过滤                      │
│                           ▼                                     │
│              多模态图片检索 (CLIP, 并行)                          │
│                           ▼                                     │
│              混合图文 RRF 融合排序                                │
└─────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│                grade_documents（文档评分）                        │
│         基于 embedding 相似度逐文档评分                            │
│         good → 生成答案 | bad → 重新增强检索 (最多2次)            │
└─────────────────────────────────────────────────────────────────┘
                                │
                                ▼ (good)
┌─────────────────────────────────────────────────────────────────┐
│           generate_current_answer（答案生成）                    │
│    纯文字生成 | 多模态生成 (VLM) | 流式输出                       │
│    置信度计算 | 引用溯源追踪                                      │
└─────────────────────────────────────────────────────────────────┘
                                │
                    ┌───────────┴───────────┐
                    │                       │
              有子问题               无子问题
                    │                       │
                    ▼                       ▼
          prepare_next_step        synthesize（多跳综合）
              (循环)                      │
                                          ▼
                                    ┌─────────┐
                                    │  final  │
                                    └─────────┘
                                          │
                                          ▼
                                    ┌─────────┐
                                    │   END   │
                                    └─────────┘
```

## 🚀 快速开始

### 环境要求

- Python 3.10+
- Milvus 2.3+ (向量数据库)
- PostgreSQL 14+ (状态持久化、知识库元数据)
- Redis 6.0+ (可选，用于缓存)
- Ollama (本地 Embedding 模型服务)
- HuggingFace Transformers (CLIP 模型、CrossEncoder 重排序)

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

# 多模态 LLM 配置（可选，支持图片理解）
# QWEN_MODEL_NAME=qwen-vl-plus

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

# 多模态配置（可选）
CLIP_MODEL_PATH=openai/clip-vit-base-patch32
IMAGE_SCORE_THRESHOLD=0.25
MAX_IMAGES_PER_QUERY=3
CAPTION_MODEL_NAME=
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
| POST   | `/api/knowledge/chunk-config`         | 保存分块配置      |
| GET    | `/api/knowledge/chunk-config`         | 获取分块配置      |

### 系统信息

| 方法   | 路径                        | 描述        |
|------|---------------------------|-----------|
| GET  | `/api/system/models`      | 获取模型配置状态 |

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
| `enable_conversation_compress`  | `true`  | 启用对话压缩    |
| `max_conversation_tokens`       | `4000`  | 最大对话 token 数 |
| `keep_recent_turns`             | `3`     | 压缩时保留最近轮数 |
| `incremental_summary_interval`  | `5`     | 渐进式摘要触发间隔 |
| `max_context_tokens`            | `2000`  | 上下文窗口 token 预算 |

### 多模态配置

| 配置项                    | 默认值                            | 描述                                |
|-------------------------|--------------------------------|-----------------------------------|
| `CLIP_MODEL_PATH`       | `openai/clip-vit-base-patch32` | CLIP 模型路径，支持本地路径或 HuggingFace 模型名 |
| `IMAGE_SCORE_THRESHOLD` | `0.25`                         | 图片检索相似度阈值，低于此值的图片不传入 VLM          |
| `MAX_IMAGES_PER_QUERY`  | `3`                            | 每次查询最多传入 VLM 的图片数量                |
| `CAPTION_MODEL_NAME`    | `""`                           | VLM Caption 生成模型，为空则跳过 Caption 生成 |

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

### LangGraph 状态机（7个核心节点）

系统采用 LangGraph 构建有向状态图，包含以下核心节点：

| 节点                          | 职责                        |
|-----------------------------|---------------------------|
| `retrieve_or_respond`       | 统一入口：任务分析、对话压缩、判断是否需要检索   |
| `prepare_next_step`         | 子问题分解或准备当前子问题             |
| `enhance_and_route_current` | 查询增强 + 路由到目标知识库           |
| `fusion_retrieve`           | 融合检索（内部 + 外部）+ 近似去重 + 重排序 + 多模态图片检索 |
| `grade_documents`           | 文档评分：基于 embedding 相似度逐文档评分     |
| `generate_current_answer`   | 生成当前查询的答案（支持流式、多模态、置信度计算）   |
| `synthesize`                | 多跳场景下合并子问题答案 + 引用溯源        |
| `final`                     | 最终输出：存储问答对、可选触发评估         |

**流程特点：**
- **条件循环**：评分失败会回到增强节点重试（最多2次）
- **多跳推理**：自动分解复杂问题为子问题队列，逐一处理后综合
- **智能路由**：简单查询直接回复，复杂查询进入检索流程

### 查询增强（QueryEnhancer）

使用 LangChain 的 LCEL 表达式，将多个增强任务构造为 RunnableSerializable，使用 RunnableParallel 并行化执行：

- **扩展**：添加元数据、上下文或相关术语扩展原始查询
- **分解**：将复杂查询分解为多个子问题
- **重写**：优化查询使其更清晰、明确
- **HyDE 预测**：生成假设性文档辅助检索

**性能对比**：

- 同步请求：平均 4s
- 并行化后：平均 1~1.5s

### 智能路由（QueryRouter）

基于 LLM 语义理解，将用户查询路由到最合适的知识库：

- 支持多知识库管理，每个知识库可配置领域、关键词等元数据
- 根据查询意图自动选择相关的知识库集合
- 兜底机制：路由结果为空时使用所有知识库

### 检索融合（RAG-Fusion）

多路检索结果融合技术：

- **混合检索**：Dense (向量) + Sparse (BM25) + 外部搜索 (Bing)
- **加权 RRF**：根据检索器可靠性赋予不同权重
- **CrossEncoder 精排**：`bge-reranker-v2-m3` 对融合结果进行精细化重排序
- **近似去重**：基于 embedding 余弦相似度（阈值 0.92），替代精确文本匹配
- **兜底机制**：重排序过滤掉所有文档时，保留原始排序的 top-1

### 多模态图片检索（CLIP）

新增跨模态图片检索通道，与文字检索并行执行：

- **CLIP 模型**：`openai/clip-vit-base-patch32`（512维），支持本地缓存
- **并行检索**：对所有知识库 collection 并发发起图片检索
- **阈值过滤**：默认 score ≥ 0.25 才保留，避免低相关图片干扰
- **image_id 去重**：跨 collection 按 image_id 去重，保留最高分
- **混合图文 RRF**：图片 CLIP score 归一化后与文字 rerank score 对齐，统一排序
- **数量控制**：最终答案阶段最多传入 3 张图片，防止超出 VLM token 限制

### 对话记忆（Memory）

- **对话历史记忆**：多轮对话历史辅助 Query 生成
- **用户画像**：抽象用户中长期兴趣指导检索
- **检索反馈**：记住已检索内容，避免重复召回
- **对话压缩策略**：
  - **跨轮压缩**：超过 10 轮或 4000 tokens 时触发压缩，保留最近 3 轮
  - **渐进式摘要**：每隔 5 轮触发增量摘要，累积跨轮对话摘要
  - **自适应窗口**：根据 token 预算动态调整保留的原文和摘要比例

### 文档分块（Chunk）

支持多种分块策略，针对不同文档类型优化：

- **递归分块**：使用中文优化分隔符（句号、问号、感叹号等），适用于 PDF、DOCX 等非结构化文档
- **Markdown 结构化分块**：按标题层级（h1-h4）切分，保留文档结构信息
- **父子文档分块**：大 chunk 作为上下文（1500 字符），小 chunk 用于向量检索（400 字符），提升检索精度
- **表格提取与增强**：
  - Markdown 格式 chunk：保留表格结构
  - 自然语言摘要 chunk：逐行转换为自然语言描述，提升检索召回率

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

### 文档评分与重试机制

- **评分模型**：`bge-large-zh-v1.5` embedding 余弦相似度
- **阈值过滤**：默认 grader_threshold=0.5，低于此值的文档被过滤
- **重试机制**：评分失败时重新进入增强检索节点，最多重试 2 次
- **防死循环**：通过 grade_retry_count 计数器防止无限循环

### 置信度计算与引用溯源

- **置信度标注**：基于重排序分数的平均值自动计算置信度（高/中/低）
- **来源追踪**：自动记录每个子问题引用的文档编号 [1][2][3]
- **引用溯源**：最终答案附加完整的引用链，展示每个子问题的来源

## 🖼️ 多模态 RAG

系统支持对 PDF 中的**图片**和**表格**进行索引与检索，并在生成阶段融合文字和图片内容，实现真正的多模态问答。

### 数据流

```
PDF 文件
   │
   ├─→ PaddleOCR（文字 OCR）→ 文本 chunks → Milvus 文本 Collection
   │
   ├─→ pypdf（嵌入图片提取）
   │       ├─→ VLM 生成 Caption（可选）
   │       └─→ CLIP Embedding → Milvus 图片 Collection
   │
   └─→ pdfplumber（表格提取）
           ├─→ Markdown 格式 chunk → Milvus 文本 Collection
           └─→ 逐行自然语言摘要 chunk → Milvus 文本 Collection
```

### 检索流程

```
用户查询
   │
   ├─→ 文字检索：Dense(qwen-embedding) + Sparse(BM25) + RRF 融合 → 重排序
   │
   └─→ 图片检索（并行）：CLIP 文字向量 → 各 collection 并行检索
           ├─→ score 阈值过滤（默认 0.25）
           ├─→ image_id 跨 collection 去重
           └─→ 混合图文 RRF 融合排序
```

### 生成策略

- **最终答案 + 合格图片**：调用多模态 VLM（如 qwen-vl-plus），传入文字 + top-N 图片 base64
- **子问题阶段 / 无合格图片**：退化为纯文字 LLM 生成，避免中间答案被无关图片干扰

### 启用多模态

1. 配置多模态 LLM：
```env
QWEN_MODEL_NAME=qwen-vl-plus
QWEN_API_KEY=your_api_key
```

2. （可选）启用 VLM Caption 生成：
```env
CAPTION_MODEL_NAME=qwen-vl-plus
```

3. 调整图片检索阈值：
```env
IMAGE_SCORE_THRESHOLD=0.25  # 降低阈值可召回更多图片，但可能引入噪声
MAX_IMAGES_PER_QUERY=3      # 控制传入 VLM 的图片数量
```

> 若使用不支持视觉的模型，多模态生成会自动退化为纯文字生成，不会报错。

## 🧪 评估

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
