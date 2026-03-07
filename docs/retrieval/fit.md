# RAG 检索阶段问题分析

本文档总结当前项目 RAG 检索阶段存在的不足，按严重性分级。

---

## 🔴 高严重性

### 1. ~~DocumentGrader 将所有文档合并为一个字符串评分~~ ✅ 已修复

**位置**：`src/services/grade_model.py` → `grade()` 方法

**现状**：`combined_docs = "\n".join(docs)`，将所有检索文档拼接后计算与 query 的整体余弦相似度，返回一个布尔值。

**问题**：
- 这是"整体评分"而非"逐文档评分"
- 如果 4 篇文档中有 1 篇高度相关、3 篇完全无关，合并后的向量会被无关文档稀释，可能导致整体分数低于阈值而误判为"不相关"
- 反之，大量弱相关文档合并后也可能虚高通过

**改进方向**：逐文档评分，过滤掉不相关的单篇文档，而非整体判断。

---

### 2. 向量检索固定 k=4，无法根据查询复杂度自适应

**位置**：`src/node/retrieval/fusion_retrieve.py` → `_search_single_query()` 第 28 行

**现状**：`asimilarity_search_with_score(query, k=4)` 硬编码为 4 条。

**问题**：
- 简单事实查询可能只需 1-2 条高质量文档，固定 k=4 引入噪声
- 复杂分析型查询可能需要 8-10 条，k=4 导致信息不足

**改进方向**：根据 `TaskCharacteristics` 的任务类型动态调整 k 值。

---

### 3. ~~重排序后无兜底机制~~ ✅ 已修复

**位置**：`src/node/retrieval/retrieval_node.py` → `__fusion_retrieve()` 第 168-180 行

**现状**：交叉编码器重排序使用 `threshold=self.config.reranker_threshold`（默认 0.8）过滤，如果所有文档分数都低于阈值，`ordered_docs` 为空，`search_content` 为空字符串。

**问题**：
- 高阈值 + 严格过滤 = 可能所有文档都被过滤掉
- 生成阶段完全没有参考资料，但流程仍然继续生成答案，幻觉风险极高

**改进方向**：至少保留 top-1 文档作为兜底，或在全部被过滤时触发重新检索/降低阈值重试。

---

## 🟡 中严重性

### 4. ~~融合去重策略过于粗暴~~ ✅ 已修复

**位置**：`src/node/retrieval/retrieval_node.py` → `__fusion_retrieve()` 第 145-151 行

**现状**：去重逻辑为 `re.sub(r'[^\w\s]', '', content.lower().strip())`，移除所有标点后做精确匹配。

**问题**：
- 只能去除完全相同的文档
- 两篇文档有 90% 重叠但略有不同（如来自不同分块策略），无法识别为近似重复
- 不同来源的同一段落（如父文档和子文档）也无法去重

**改进方向**：使用 MinHash/SimHash 或基于 embedding 余弦相似度的近似去重。

---

### 5. 查询增强的多个 LLM 调用缺乏质量控制

**位置**：`src/node/route/query_enhancer.py` → `QueryEnhancer`

**现状**：同义改写、扩展改写、专业化改写、HyDE 预测、查询分解共 5 个 LLM 调用并行执行。

**问题**：
- 没有对增强结果的质量评估，低质量改写可能引入噪声检索
- `_deduplicate_queries` 只做精确去重（移除标点后比较），语义相近但措辞不同的查询无法去重
- 所有增强策略对所有查询类型一视同仁，没有根据查询特征选择性启用

**改进方向**：根据 `TaskCharacteristics` 选择性启用增强策略；对增强结果做语义去重。

---

### 6. 查询路由依赖 LLM 判断，无缓存和回退机制

**位置**：`src/node/route/query_router.py` → `QueryRouter`

**现状**：每个查询都调用 LLM 判断路由到哪个知识库。

**问题**：
- 相同或相似的查询每次都重新调用 LLM，浪费资源
- LLM 路由判断可能不稳定（同一查询多次调用结果不同）
- 如果 LLM 返回格式错误或空数组，没有回退到默认知识库的机制

**改进方向**：增加路由结果缓存；LLM 路由失败时回退到全库检索。

---

### 7. CrossEncoderRanker 的 max_length=512 限制

**位置**：`src/services/cross_encoder_ranker.py` → `CrossEncoderRanker.__init__()` 第 16 行

**现状**：`CrossEncoder('BAAI/bge-reranker-v2-m3', max_length=512)`，交叉编码器截断超过 512 token 的文档。

**问题**：
- 如果使用父子文档策略，父文档通常较长（1000+ token），截断后丢失关键信息，导致重排序分数不准确

**改进方向**：对长文档采用滑动窗口取最高分，或使用支持更长输入的重排序模型。

---

### 8. 向量检索的相似度阈值硬编码为 0.2

**位置**：`src/node/retrieval/fusion_retrieve.py` → `_search_single_query()` 第 30 行

**现状**：`filtered = [(doc, score) for doc, score in result if score >= 0.2]`

**问题**：
- 0.2 是一个非常低的阈值，几乎不过滤任何结果，等于没有初筛
- 不同 embedding 模型的分数分布不同，硬编码阈值缺乏适应性

**改进方向**：将阈值提取为配置项，或使用动态阈值（如 top-k 的平均分 × 系数）。

---

## 🟢 低严重性 / 优化建议

### 9. grade_documents 评分与重排序功能重叠

**位置**：`src/node/retrieval/retrieval_node.py` → `__grade_documents()` 第 197-215 行

**现状**：重排序已经按相关性排序并过滤了低分文档，`grade_documents` 又用另一个模型（`bge-large-zh-v1.5`）做整体相关性评分。

**问题**：
- 两次评分使用不同模型、不同粒度，可能产生矛盾判断
- grade 失败后触发重新检索（回到 `enhance_and_route_current`），但查询增强策略不变，容易陷入相同结果的循环

**改进方向**：统一评分标准，或让 grade 失败时调整查询增强策略（如启用更多改写方式）。

---

### 10. 外部检索的查询改写独立于查询增强器

**位置**：`src/node/retrieval/retrieval_node.py` → `__retrieve_external()` 第 70-78 行

**现状**：外部搜索前有一个独立的 LLM 调用来改写搜索查询，但这个改写逻辑与 `QueryEnhancer` 完全独立。

**改进方向**：复用 `QueryEnhancer` 的改写结果，避免重复 LLM 调用。

---

### 11. 检索结果没有元数据过滤能力

**现状**：当前只支持按 collection 路由，不支持按文档的时间、标签、类型等元数据过滤。

**改进方向**：在 Milvus 检索时支持 metadata filter 表达式，结合 `TimeTransformer` 的时间解析结果做时间范围过滤。

---

## 问题优先级排序

| 优先级 | 问题 | 影响 |
|:------:|------|------|
| P0 | DocumentGrader 整体评分 | 评分不准确，影响检索质量判断 |
| P0 | 重排序后无兜底机制 | 可能导致无参考资料生成，幻觉风险 |
| P1 | 向量检索固定 k=4 | 简单查询噪声多，复杂查询信息不足 |
| P1 | 融合去重策略粗暴 | 近似重复文档无法去除，浪费 token |
| P1 | CrossEncoder max_length=512 | 长文档重排序不准确 |
| P2 | 查询增强缺乏质量控制 | 低质量改写引入噪声 |
| P2 | 查询路由无缓存和回退 | 资源浪费、稳定性差 |
| P2 | 相似度阈值硬编码 | 缺乏适应性 |
| P3 | 评分与重排序重叠 | 矛盾判断、循环风险 |
| P3 | 外部检索改写独立 | 重复 LLM 调用 |
| P3 | 无元数据过滤 | 检索精度受限 |
