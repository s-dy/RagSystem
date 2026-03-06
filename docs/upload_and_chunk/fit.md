## 数据切分策略优化方案

### 一、现状分析

当前项目使用 `ChunkHandler.recursive_chunk()` 方法，基于 LangChain 的 `RecursiveCharacterTextSplitter` 进行文本切分：

| 配置项 | 当前值 | 说明 |
|--------|:------:|------|
| chunk_size | 1024 | 每个分块的最大字符数 |
| chunk_overlap | 128 | 相邻分块的重叠字符数 |
| 分隔符 | 默认（英文优先） | 未针对中文优化 |

**主要局限性：**

- 容易在句子中间、段落中间断开，破坏语义完整性
- 完全忽略文档结构（标题、段落、列表等）
- 固定大小对不同信息密度的文本效果差异大
- 默认分隔符对中文文档不友好

---

### 二、优化策略

#### 策略 1：中文分隔符优化（🔴 高优先级） -- 采用

**问题**：`RecursiveCharacterTextSplitter` 默认分隔符是英文优先的（`\n\n` → `\n` → ` ` → `""`），对中文文档的句子边界识别不佳。

**方案**：自定义中文优先的分隔符列表：

```python
chinese_separators = [
    "\n\n",      # 段落分隔
    "\n",        # 换行
    "。",        # 句号
    "！", "？",  # 感叹号、问号
    "；",        # 分号
    "，",        # 逗号
    " ",         # 空格
    "",          # 字符级兜底
]

splitter = RecursiveCharacterTextSplitter(
    chunk_size=1024,
    chunk_overlap=128,
    separators=chinese_separators,
)
```

**预期收益**：切分点落在中文标点处，保持句子完整性，显著提升检索质量。

---

#### 策略 2：Markdown 结构化切分（🔴 高优先级） -- 采用

**问题**：项目已支持 Markdown 格式的知识库文件，但切分时未利用 Markdown 的标题层级结构。

**方案**：对 Markdown 文件使用 `MarkdownHeaderTextSplitter`，按标题层级切分：

```python
from langchain_text_splitters import MarkdownHeaderTextSplitter

headers_to_split_on = [
    ("#", "h1"),
    ("##", "h2"),
    ("###", "h3"),
]

md_splitter = MarkdownHeaderTextSplitter(
    headers_to_split_on=headers_to_split_on,
    strip_headers=False,  # 保留标题文本
)
```

**优势**：

- 每个 chunk 对应一个完整的章节/小节
- 标题信息自动注入 metadata，便于检索时过滤
- 对于超长章节，可二次使用 `RecursiveCharacterTextSplitter` 切分

---

#### 策略 3：动态 chunk_size（🟡 中优先级）

**问题**：固定 1024 字符对所有文档一视同仁，但不同类型文档的最佳 chunk_size 差异很大。

**方案**：根据文件类型和内容特征自动选择切分参数：

| 文档类型 | 建议 chunk_size | 建议 overlap | 原因 |
|----------|:--------------:|:------------:|------|
| 新闻/短文 | 512 | 64 | 信息密集，小 chunk 更精准 |
| 技术文档 | 1024 | 128 | 中等密度，保持段落完整 |
| 法律/学术论文 | 1500-2000 | 200 | 长句多，需要更大上下文 |
| FAQ/问答 | 按条目切分 | 0 | 每个 QA 对天然是一个 chunk |

**实现思路**：在 `ChunkHandler` 中根据文件元数据中的 `file_type` 字段自动匹配参数。

---

#### 策略 4：语义感知切分（🟢 低优先级）

**问题**：固定大小切分无视语义边界，可能把一个完整论点切成两半。

**方案**：使用基于嵌入相似度的语义切分，在语义转折点处断开：

```python
from langchain_experimental.text_splitter import SemanticChunker

semantic_splitter = SemanticChunker(
    embeddings=embedding_model,
    breakpoint_threshold_type="percentile",
    breakpoint_threshold_amount=75,
)
```

**工作原理**：

1. 将文本按句子拆分
2. 计算相邻句子的嵌入向量余弦相似度
3. 当相似度低于阈值时，在该位置断开

**适用场景**：PDF 论文、长篇文档等信息密度不均匀的文本。

**注意**：需要调用 Embedding 模型，切分速度较慢，适合离线预处理。

---

#### 策略 5：父子文档策略（🟢 低优先级）--采用

**问题**：小 chunk 检索精准但缺乏上下文，大 chunk 上下文丰富但检索噪声大。

**方案**：两层切分，用小 chunk 做检索，返回时附带其所属的大 chunk（父文档）作为上下文：

```
原始文档
  └── 父文档（1500-2000 字符）  ← 作为 LLM 上下文
        ├── 子文档 1（256-512 字符）  ← 用于向量检索
        ├── 子文档 2（256-512 字符）
        └── 子文档 3（256-512 字符）
```

LangChain 提供 `ParentDocumentRetriever` 可直接实现，需配合文档存储（如 InMemoryStore 或 Redis）保存父子关系。

---

#### 策略 6：元数据增强

**问题**：当前切分后的 chunk 元数据较少，不利于检索时的过滤和排序。

**建议**：在切分时为每个 chunk 注入更丰富的元数据：

```python
metadata = {
    "source": file_path,         # 来源文件
    "file_type": "pdf",          # 文件类型
    "chunk_index": 3,            # 在原文中的位置序号
    "total_chunks": 15,          # 原文总 chunk 数
    "section_title": "第三章",    # 所属章节标题（如有）
    "char_count": 856,           # 字符数
}
```

**收益**：支持按文件类型、章节等维度过滤检索结果，提升检索精准度。

---

### 三、实施优先级总结

| 优先级 | 优化项 | 实施难度 | 预期收益 | 建议阶段 |
|:------:|--------|:--------:|:--------:|:--------:|
| 🔴 高 | 中文分隔符优化 | 低 | 高 | 立即实施 |
| 🔴 高 | Markdown 结构化切分 | 中 | 高 | 立即实施 |
| 🟡 中 | 动态 chunk_size | 中 | 中 | 短期规划 |
| 🟢 低 | 语义感知切分 | 高 | 高 | 中期规划 |
| 🟢 低 | 父子文档策略 | 高 | 高 | 中期规划 |
| 🟢 低 | 元数据增强 | 低 | 中 | 按需实施 |

### 四、评估验证

每项优化实施后，建议通过项目的 RAG 评估体系（`tests/eval/ragas_eval.py`）验证效果：

- **Context Relevance**：衡量切分优化后检索精准度是否提升
- **Context Recall**：衡量切分优化后检索召回率是否改善
- **Faithfulness**：衡量更好的上下文是否减少了生成幻觉

建议在同一批评估样本上对比优化前后的指标变化，确保改进有据可依。