# 多模态 RAG 架构文档

## 概述

本文档描述了 hybridRag 系统的多模态 RAG 扩展实现，支持对 PDF 中的**图片**和**表格**进行索引与检索，并在生成阶段融合文字和图片内容，实现真正的多模态问答。

---

## 整体架构

```
用户输入（文字查询）
        │
        ▼
  ┌─────────────────────────────────────────────┐
  │              路由 & 任务分析                  │
  └─────────────────────────────────────────────┘
        │
        ▼
  ┌─────────────────────────────────────────────────────────────┐
  │                  融合检索（fusion_retrieve）                   │
  │  ┌──────────────────────┐  ┌──────────────────────────────┐ │
  │  │   文字检索（Milvus）   │  │      图片检索（CLIP）          │ │
  │  │  Dense + Sparse      │  │  并行跨模态向量检索             │ │
  │  │  + RRF 融合 + 重排序  │  │  阈值过滤 + image_id 去重     │ │
  │  └──────────────────────┘  └──────────────────────────────┘ │
  │                    ↓ 混合图文 RRF 融合排序                    │
  └─────────────────────────────────────────────────────────────┘
        │
        ▼
  ┌─────────────────────────────────────────────────────────────┐
  │                       多模态生成                              │
  │  最终答案 + 图片 score ≥ 阈值 → generate_multimodal_answer    │
  │  子问题 / 无图片          → generate_answer_for_query         │
  └─────────────────────────────────────────────────────────────┘
        │
        ▼
     最终答案（含引用溯源）
```

---

## 数据流

### 文档入库阶段

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

> **图片入库与文字入库解耦**：图片/表格入库通过 `asyncio.create_task` 异步提交，文字入库完成后立即返回，不阻塞主流程。

### 检索阶段

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

### 生成阶段

```
检索结果
   │
   ├─ 最终答案 + 图片 score ≥ 阈值 → 多模态 prompt（文字 + top-N 图片 base64）→ VLM
   │
   ├─ 子问题阶段 → 纯文字 prompt → LLM（图片不传入，避免干扰中间答案）
   │
   └─ 无合格图片 → 纯文字 prompt → LLM
```

---

## 模块说明

### Phase 1：多模态文档解析（`src/services/data_load/parser.py`）

`PaddleOCRParser` 新增三个方法：

| 方法                               | 功能                 | 依赖           |
|----------------------------------|--------------------|--------------|
| `_extract_pdf_embedded_images()` | 提取 PDF 内嵌图片字节      | `pypdf`      |
| `_extract_pdf_tables()`          | 提取表格并转为 Markdown   | `pdfplumber` |
| `_table_to_markdown()`           | 二维列表 → Markdown 表格 | 无            |

`parse_pdf()` 现在返回三种内容块：

```python
# 文字块
{"type": "text", "text": "...", "page_idx": 0}

# 图片块
{"type": "image", "data": b"...", "name": "img1.png", "page_idx": 1, "source": "/path/to.pdf"}

# 表格块
{"type": "table", "text": "| 列1 | 列2 |\n|---|---|\n| 值1 | 值2 |", "page_idx": 2, "source": "/path/to.pdf"}
```

**控制参数：**

```python
parser.parse_pdf(pdf_path, extract_images=True, extract_tables=True)
```

---

### Phase 2：CLIP Embedding + 图片 Milvus Collection

#### `src/services/embedding/clip_embedding.py`

```python
from src.services.embedding.clip_embedding import get_clip_embedding

clip = get_clip_embedding()  # 单例，懒加载，自动读取 CLIP_MODEL_PATH 环境变量

# 图片向量化（512 维）
vector = clip.embed_image_bytes(image_bytes)

# 文字向量化（与图片在同一向量空间）
vector = clip.embed_text("查询文本")
```

- 模型：优先读取 `CLIP_MODEL_PATH` 环境变量，默认 `openai/clip-vit-base-patch32`（512 维）
- 支持本地模型路径（生产环境离线部署）和 HuggingFace Hub 在线下载
- L2 归一化，余弦相似度等价于内积（IP）
- 单例池，按模型名缓存，避免重复加载

#### `src/services/storage/milvus_image_client.py`

图片专用 Milvus Collection，命名规则：`{文本collection名}_images`

**Collection Schema：**

| 字段             | 类型                | 说明                    |
|----------------|-------------------|-----------------------|
| `image_id`     | VARCHAR(256)      | 主键，MD5 唯一标识           |
| `clip_vector`  | FLOAT_VECTOR(512) | CLIP 图片向量             |
| `caption`      | VARCHAR(2048)     | VLM 生成的图片描述（增强文字检索召回） |
| `source`       | VARCHAR(512)      | 来源文件路径                |
| `page_idx`     | INT64             | 来源页码                  |
| `image_base64` | VARCHAR(65535)    | 图片 base64 编码          |

**检索接口（含阈值过滤和去重）：**

```python
from src.services.storage.milvus_image_client import MilvusImageClient
from config import MilvusConfig

client = MilvusImageClient(MilvusConfig(collection_name="my_collection"))

# 文字查询图片（跨模态），自动过滤 score < IMAGE_SCORE_THRESHOLD 的结果
images = client.search_by_text("流程图", top_k=3)

# 图片查询相似图片
images = client.search_by_image_bytes(image_bytes, top_k=3)

# 异步批量入库（CLIP 推理在线程池中执行）
count = await client.insert_images_async(image_records)
```

**`_search()` 优化细节：**

- 多取候选（`top_k * 3`）后按阈值过滤，再按 `image_id` 去重
- 阈值从 `MultimodalConfig.image_score_threshold` 读取（默认 0.25）

---

### Phase 3：图片检索通道（`src/node/retrieval/retrieval_node.py`）

`__fusion_retrieve` 节点在文字检索完成后，并行调用 `__retrieve_images()` 进行图片检索，并执行混合图文 RRF 融合：

```python
# 并行检索所有 collection（asyncio.gather）
results_per_collection = await asyncio.gather(
    *[search_one_collection(name) for name in router_index.keys()]
)

# 跨 collection 按 image_id 去重，按 score 降序
merged.sort(key=lambda img: img.score, reverse=True)

# 混合图文 RRF 融合：CLIP score 归一化后与文字 rerank score 对齐
image.rrf_score = image.score / max(max_text_score, 1e-6)
retrieved_images.sort(key=lambda img: img.rrf_score, reverse=True)
```

**容错设计：**

- 图片检索失败时静默跳过，不影响文字检索主流程
- 每个 collection 独立检索，单个失败不影响其他
- CLIP 推理通过 `asyncio.to_thread` 放入线程池，不阻塞事件循环

---

### Phase 4：多模态生成（`src/node/generate/generate.py`）

新增 `generate_multimodal_answer()` 函数：

```python
from src.node.generate.generate import generate_multimodal_answer

answer = await generate_multimodal_answer(
    llm=llm,                          # 需支持多模态（如 qwen-vl-plus）
    query="这张图表说明了什么？",
    docs_content="[来源1] ...",
    conversation_context="...",
    image_base64_list=["iVBOR..."],   # 图片 base64 列表（已按 score 过滤和数量限制）
)
```

**退化策略：**

- `image_base64_list` 为空 → 退化为纯文字生成
- VLM 调用异常 → 捕获异常，退化为纯文字生成，记录 warning 日志

---

### Phase 5：全流程串联

#### State 字段（`src/graph.py`）

```python
class State(MessagesState):
    # ... 原有字段 ...
    retrieved_images: List  # CLIP 图片检索结果（RetrievedImage 列表），用于多模态生成
```

#### 生成节点图片筛选逻辑（`src/node/generate/generate_node.py`）

```python
# 读取配置
image_score_threshold = MultimodalConfig().image_score_threshold  # 默认 0.25
max_images_per_query = MultimodalConfig().max_images_per_query    # 默认 3

# 子问题阶段不传图片（is_final=False），避免中间答案被无关图片干扰
# 最终答案阶段：按 score 阈值过滤，取 top-N 张
qualified_images = []
if is_final:
    qualified_images = [
        img for img in retrieved_images
        if img.score >= image_score_threshold and img.image_base64
    ]
    qualified_images = sorted(qualified_images, key=lambda img: img.score, reverse=True)[:max_images_per_query]

# 有合格图片 → 多模态生成，否则 → 纯文字生成
if qualified_images and is_final:
    answer = await generate_multimodal_answer(...)
else:
    answer = await generate_answer_for_query(...)
```

#### 图片入库（`src/services/data_load/data_storage.py`）

`DataDBStorage.ingest()` 文字入库完成后，通过 `asyncio.create_task` 异步触发图片和表格入库：

```python
# 图片入库（异步，不阻塞主流程）
asyncio.create_task(self._ingest_images_from_path(data_path, collection_name))

# _ingest_images_from_path 内部并行执行：
#   - _ingest_tables_from_path：表格 Markdown + 逐行摘要写入文字 Collection
#   - 图片 CLIP 向量化（线程池）+ VLM Caption 生成（可选）→ 图片 Collection
```

**新增方法一览：**

| 方法                                 | 功能                                       |
|------------------------------------|------------------------------------------|
| `_generate_image_caption()`        | 调用 VLM 为图片生成描述（需配置 `CAPTION_MODEL_NAME`） |
| `_ingest_tables_from_path()`       | 提取表格，写入 Markdown + 逐行摘要两种 chunk          |
| `_table_markdown_to_row_summary()` | Markdown 表格 → 自然语言逐行摘要                   |

---

## 配置说明

### `config.py` - `MultimodalConfig`

| 环境变量                    | 默认值                            | 说明                                |
|-------------------------|--------------------------------|-----------------------------------|
| `CLIP_MODEL_PATH`       | `openai/clip-vit-base-patch32` | CLIP 模型路径，支持本地路径或 HuggingFace 模型名 |
| `IMAGE_SCORE_THRESHOLD` | `0.25`                         | 图片检索相似度阈值，低于此值的图片不传入 VLM          |
| `MAX_IMAGES_PER_QUERY`  | `3`                            | 每次查询最多传入 VLM 的图片数量                |
| `CAPTION_MODEL_NAME`    | `""`                           | VLM Caption 生成模型，为空则跳过 Caption 生成 |

### 多模态 LLM 配置

若要启用真正的多模态生成（图片理解），需要将 `QWEN_MODEL_NAME` 配置为支持视觉的模型：

```env
QWEN_MODEL_NAME=qwen-vl-plus
QWEN_API_KEY=your_api_key
QWEN_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1
```

若要启用 VLM Caption 生成（增强图片文字检索召回）：

```env
CAPTION_MODEL_NAME=qwen-vl-plus
```

> 若使用不支持视觉的模型，多模态生成会自动退化为纯文字生成，不会报错。

### CLIP 模型配置

```env
# 使用 HuggingFace 在线模型（默认）
CLIP_MODEL_PATH=openai/clip-vit-base-patch32

# 使用本地模型（生产环境离线部署）
CLIP_MODEL_PATH=/models/clip-vit-base-patch32
```

---

## 依赖说明

### 核心依赖（已在 `pyproject.toml` 中声明）

```bash
pip install pypdf pdfplumber transformers torch pillow
```

| 包              | 用途          |
|----------------|-------------|
| `pypdf`        | 提取 PDF 内嵌图片 |
| `pdfplumber`   | 提取 PDF 表格   |
| `transformers` | CLIP 模型加载   |
| `torch`        | CLIP 推理     |
| `pillow`       | 图片字节解码      |

### 可选依赖组（`[multimodal]`）

```bash
# 安装多模态扩展依赖
pip install -e ".[multimodal]"
```

> 以上依赖均为**可选降级**：若未安装，对应功能会跳过并记录 warning，不影响原有文字检索功能。

---

## 优化记录

### 检索质量优化

- **图片相似度阈值过滤**：`_search()` 多取候选后按 `IMAGE_SCORE_THRESHOLD` 过滤，避免低相关图片干扰生成
- **image_id 去重**：`_search()` 和 `__retrieve_images()` 两层去重，同一图片只保留最高分
- **表格文字检索增强**：表格写入两种格式的 chunk（Markdown + 逐行自然语言摘要），提升表格内容的检索召回率

### 性能优化

- **CLIP 推理异步化**：`insert_images_async()` 通过 `asyncio.to_thread` 将 CPU 密集型推理放入线程池
- **图片检索并行化**：`__retrieve_images()` 对所有 collection 使用 `asyncio.gather` 并行检索

### 架构优化

- **图片入库与文字入库解耦**：`ingest()` 通过 `asyncio.create_task` 异步提交图片入库任务，文字入库完成后立即返回
- **表格入库与图片入库并行**：`_ingest_images_from_path()` 内部并行执行表格和图片入库

### 效果优化

- **混合图文 RRF 融合**：图片 CLIP score 归一化后与文字 rerank score 对齐，统一排序
- **图片数量控制**：最终答案阶段最多传入 `MAX_IMAGES_PER_QUERY` 张图片，防止超出 VLM token 限制
- **子问题阶段不传图片**：`is_final=False` 时不传图片，避免中间答案被无关图片干扰

### 工程优化

- **CLIP 模型本地缓存**：支持 `CLIP_MODEL_PATH` 环境变量指定本地模型路径，生产环境无需访问 HuggingFace
- **`MultimodalConfig` 统一配置**：所有多模态相关配置集中在 `config.py` 的 `MultimodalConfig` 中管理
- **`[multimodal]` 可选依赖组**：`pyproject.toml` 新增可选依赖组，与核心依赖分离
- **VLM Caption 入库**：配置 `CAPTION_MODEL_NAME` 后，入库时自动为每张图片生成文字描述，增强文字检索召回

---

## 扩展方向

1. **图片 OCR**：对图片中的文字内容进行 OCR，补充到文字检索索引
2. **更大的 CLIP 模型**：替换为 `openai/clip-vit-large-patch14`（768 维）提升检索精度
3. **图片查询输入**：前端支持用户上传图片作为查询，调用 `search_by_image_bytes()` 检索相似图片
4. **图片 base64 存储优化**：改为存储图片到对象存储（OSS/MinIO），Milvus 只存 URL，突破 VARCHAR 长度限制
