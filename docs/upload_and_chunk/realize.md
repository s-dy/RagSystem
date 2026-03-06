## HybridRAG 数据切分策略实现详解

本文档详细描述 HybridRAG 项目当前已实现的所有数据切分策略，包括实现原理、代码位置、配置方式和数据流转。

---

### 一、切分策略总览

| 策略 | 适用文件类型 | 核心类/方法 | 配置方式 |
|------|-------------|------------|----------|
| 中文优化递归切分 | PDF、DOCX | `ChunkHandler.recursive_chunk()` | 默认启用 |
| Markdown 结构化切分 | `.md` / `.markdown` | `ChunkHandler.markdown_chunk()` | 按 `file_type` 自动路由 |
| 父子文档切分 | 所有格式 | `ChunkHandler.parent_child_chunk()` | `RagSystemConfig.enable_parent_child_retrieval` |

---

### 二、中文优化递归切分

**文件**：`src/services/data_load/chunk.py` → `ChunkHandler.recursive_chunk()`

**核心改进**：使用中文优先的分隔符列表替代 LangChain 默认的英文分隔符，确保切分点落在中文标点处。

**分隔符优先级**（从高到低）：

```
"\n\n" → "\n" → "。" → "！" → "？" → "；" → "，" → " " → ""
```

**默认参数**：

| 参数 | 值 | 说明 |
|------|:--:|------|
| chunk_size | 1024 | 每个分块最大字符数 |
| chunk_overlap | 128 | 相邻分块重叠字符数 |

**数据流**：

```
原始文档（PDF/DOCX）
  → RecursiveCharacterTextSplitter（中文分隔符）
  → Document 列表
  → 存入 Milvus 向量库
```

---

### 三、Markdown 结构化切分

**文件**：`src/services/data_load/chunk.py` → `ChunkHandler.markdown_chunk()`

**实现原理**：

1. **第一层**：使用 `MarkdownHeaderTextSplitter` 按标题层级（`#` / `##` / `###` / `####`）切分，每个章节/小节成为一个独立 chunk
2. **第二层**：对超过 `chunk_size` 的章节，使用 `RecursiveCharacterTextSplitter`（中文分隔符）进行二次切分

**标题层级配置**：

| Markdown 标记 | 元数据键 |
|:-------------:|:--------:|
| `#` | h1 |
| `##` | h2 |
| `###` | h3 |
| `####` | h4 |

**元数据合并**：切分后的 chunk 同时包含原始文件元数据（source、file_type 等）和标题层级元数据（h1、h2、h3 等），便于检索时按章节过滤。

**自动路由**：在 `DataDBStorage.load_data_and_chunk()` 中，根据文档 `metadata["file_type"]` 自动选择切分策略：

```
文档加载
  ├── file_type == "md" / "markdown" → markdown_chunk()
  └── 其他（pdf、docx）             → recursive_chunk()
```

**代码位置**：`src/services/data_load/data_storage.py` → `DataDBStorage.load_data_and_chunk()`

---

### 四、父子文档切分

**核心思想**：用小 chunk 做向量检索（精准匹配），命中后回溯到对应的大 chunk（父文档）作为 LLM 上下文（信息更完整）。

```
原始文档
  └── 父文档（1500 字符）  ← 存入 PostgreSQL，作为 LLM 上下文返回
        ├── 子文档 1（400 字符）  ← 存入 Milvus，用于向量检索
        ├── 子文档 2（400 字符）
        └── 子文档 3（400 字符）
```

#### 4.1 切分阶段

父子文档切分根据文件类型自动路由到不同的切分策略：

```
文档加载
  ├── file_type == "md" / "markdown" → markdown_parent_child_chunk()
  └── 其他（pdf、docx）             → parent_child_chunk()
```

**4.1.1 非 Markdown 文件**

**文件**：`src/services/data_load/chunk.py` → `ChunkHandler.parent_child_chunk()`

**默认参数**：

| 参数 | 值 | 说明 |
|------|:--:|------|
| parent_size | 1500 | 父文档最大字符数 |
| parent_overlap | 200 | 父文档重叠字符数 |
| child_size | 400 | 子文档最大字符数 |
| child_overlap | 64 | 子文档重叠字符数 |

**parent_id 格式**：`{source文件路径}_{父文档序号}`，例如 `data/guide.pdf_3`

**4.1.2 Markdown 文件**

**文件**：`src/services/data_load/chunk.py` → `ChunkHandler.markdown_parent_child_chunk()`

**流程**：先用 `markdown_chunk()` 按标题层级切分出结构化的父文档（保留标题元数据），再对每个父文档用 `RecursiveCharacterTextSplitter` 切分出子文档。如果父文档本身不超过 `child_size`，则直接作为子文档使用。

**parent_id 格式**：`{source文件路径}_md_{父文档序号}`，例如 `data/guide.md_md_5`

**优势**：相比直接用 `parent_child_chunk()` 处理 Markdown，保留了标题层级结构信息（h1、h2、h3 等元数据），父文档边界与章节边界对齐。

**通用返回值**：`(parent_store, child_docs)`
- `parent_store`：`{parent_id: Document}` 父文档映射
- `child_docs`：子文档列表，每个子文档的 `metadata` 中包含 `parent_id`

#### 4.2 存储阶段

**文件**：`src/services/data_load/data_storage.py` → `DataDBStorage.save_to_vector(use_parent_child=True)`

**数据流**：

```
原始文档
  ├── Markdown → markdown_parent_child_chunk()
  └── 其他     → parent_child_chunk()
  结果：
  ├── 子文档 → 存入 Milvus 向量库
  └── 父文档 → 存入 PostgreSQL（parent_documents 表）
```

**PostgreSQL 表结构**（`parent_documents`）：

| 字段 | 类型 | 说明 |
|------|------|------|
| parent_id | VARCHAR(512) PK | 父文档唯一标识 |
| content | TEXT | 父文档文本内容 |
| metadata | JSONB | 元数据（source、file_type 等） |
| created_at | TIMESTAMP | 创建时间 |

**代码位置**：`src/services/storage/postgres_connector.py` → `PostgreSQLConnector.batch_insert_parent_documents()`

#### 4.3 检索阶段

**文件**：`src/node/retrieval/fusion_retrieve.py` → `FusionRetrieve`

**流程**：

```
用户查询
  → Milvus 混合检索（稠密向量 0.7 + BM25 0.3）
  → 命中子文档（包含 parent_id）
  → 根据 parent_id 从 PostgreSQL 批量查询父文档
  → 去重后返回父文档内容
  → 交叉编码器重排序
  → 作为 LLM 上下文
```

**关键方法**：`FusionRetrieve._resolve_parent_documents()`
- 从检索结果中提取所有 `parent_id`
- 去重后批量查询 PostgreSQL
- 按检索顺序返回父文档内容
- 对于没有 `parent_id` 的文档（非父子模式入库的），直接返回原始内容作为兜底

**代码位置**：`src/services/storage/postgres_connector.py` → `PostgreSQLConnector.get_parent_documents_by_ids()`

#### 4.4 配置开关

**文件**：`config/Config.py` → `RagSystemConfig`

```python
config = RagSystemConfig(enable_parent_child_retrieval=True)
```

- `enable_parent_child_retrieval=False`（默认）：检索时直接返回子文档内容
- `enable_parent_child_retrieval=True`：检索时自动回溯父文档

该配置通过 `Graph.__init__` → `_retrieve_internal` → `FusionRetrieve(use_parent_child=...)` 传递到检索层。

---

### 五、文件结构

```
src/services/data_load/
├── chunk.py                          # 切分核心模块
│   ├── CHINESE_SEPARATORS            # 中文分隔符列表
│   ├── MARKDOWN_HEADERS_TO_SPLIT     # Markdown 标题层级配置
│   └── ChunkHandler
│       ├── recursive_chunk()              # 中文优化递归切分
│       ├── markdown_chunk()               # Markdown 结构化切分
│       ├── parent_child_chunk()           # 父子文档切分（PDF/DOCX）
│       └── markdown_parent_child_chunk()  # Markdown 父子文档切分
├── data_storage.py                   # 数据入库流程
│   └── DataDBStorage
│       ├── load_data_and_chunk()           # 普通切分（自动路由 MD/其他）
│       ├── load_and_chunk_parent_child()   # 父子文档切分
│       └── save_to_vector()               # 统一入库入口
└── file_tool.py                      # 文档加载（PDF/DOCX/Markdown）

src/services/storage/
├── milvus_client.py                  # Milvus 向量数据库客户端
│   ├── MilvusConfig
│   └── MilvusExecutor
└── postgres_connector.py             # PostgreSQL 连接器
    └── PostgreSQLConnector
        ├── create_parent_documents_table()    # 建表
        ├── batch_insert_parent_documents()    # 批量插入父文档
        └── get_parent_documents_by_ids()      # 批量查询父文档

src/node/retrieval/
└── fusion_retrieve.py                # 融合检索
    └── FusionRetrieve
        ├── _search_single_query()         # 单查询检索
        └── _resolve_parent_documents()    # 父文档回溯

config/
└── Config.py
    └── RagSystemConfig
        └── enable_parent_child_retrieval     # 父子文档检索开关
```

---

### 六、使用示例

#### 6.1 普通模式入库（自动路由 Markdown / 其他）

```python
from src.services.data_load import DataDBStorage

storage = DataDBStorage()
await storage.save_to_vector("data/knowledge_base/")
```

#### 6.2 父子文档模式入库

```python
storage = DataDBStorage()
await storage.save_to_vector("data/knowledge_base/", use_parent_child=True)
```

#### 6.3 启用父子文档检索

```python
from config import RagSystemConfig
from src.graph import Graph

config = RagSystemConfig(enable_parent_child_retrieval=True)
graph = Graph(config)
await graph.start(messages=[...])
```