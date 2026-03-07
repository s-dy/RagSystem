## HybridRAG 评估体系

本文档详细描述 HybridRAG 项目的 RAG（Retrieval-Augmented Generation）评估体系，涵盖数据集选择、评估指标、实验设置、结果分析和案例研究。

---

### 一、系统架构概览

HybridRAG 是一个基于 LangGraph 构建的混合检索增强生成系统，核心流程如下：

```
用户输入 → 任务分析 → 查询增强 → 路由分发 → 融合检索（内部向量库 + 外部搜索）
         → 交叉编码器重排序 → 文档评分 → 答案生成 → [可选] RAG 评估 → 输出
```

评估体系基于 [ragas](https://docs.ragas.io/) 框架实现，集成在 `src/eval/ragas_eval.py` 中，通过 `RagSystemConfig.enable_eval` 配置开关控制是否在运行时自动触发评估。

---

### 二、数据集选择

#### 2.1 数据来源

HybridRAG 的知识库数据源支持以下三种格式：

| 格式 | 扩展名 | 加载方式 |
|------|--------|----------|
| PDF | `.pdf` | `pypdf.PdfReader` 逐页提取文本 |
| Word 文档 | `.doc` / `.docx` | `python-docx` 段落提取 |
| Markdown | `.md` / `.markdown` | 直接读取文本内容 |

数据通过 `src/services/data_load/file_tool.py` 的 `load_document()` 函数加载，支持单文件和目录递归加载。加载后经过 `ChunkHandler` 进行文本切分（支持中文优化递归切分、Markdown 结构化切分、父子文档切分），处理后存入 Milvus 向量数据库。

#### 2.2 评估数据集构建

评估数据集由 `EvalSample` 数据结构定义，每条样本包含：

| 字段 | 类型 | 说明 | 是否必填 |
|------|------|------|:--------:|
| `user_input` | `str` | 用户原始问题 | ✅ |
| `response` | `str` | 系统生成的回答 | ✅ |
| `retrieved_contexts` | `list[str]` | 检索到的上下文文档列表 | ✅ |
| `reference` | `str` | 人工标注的参考答案 | ❌ |

**数据集构建方式：**

- **在线采集**：开启 `enable_eval=True` 后，系统在每次检索请求结束时自动从 Graph State 中提取 `original_query`、`answer`、`retrieved_documents` 构建评估样本
- **离线构建**：手动构建 `EvalSample` 列表，通过 `RagEvaluator.evaluate_batch()` 进行批量评估
- **标注数据**：对于需要 Context Recall 指标的场景，需提供人工标注的 `reference` 参考答案

#### 2.3 样本分布建议

为确保评估的全面性，建议数据集覆盖以下维度：

| 维度 | 类别 | 建议占比 |
|------|------|:--------:|
| **任务类型** | 事实检索（FACT_RETRIEVAL） | 30% |
| | 分析比较（ANALYTICAL_COMPARISON） | 20% |
| | 流程查询（PROCEDURAL_QUERY） | 15% |
| | 多步执行（MULTI_STEP_EXECUTION） | 15% |
| | 实时交互（REAL_TIME_INTERACTION） | 10% |
| | 其他（创意生成、验证等） | 10% |
| **问题复杂度** | 单跳问题 | 60% |
| | 多跳问题（需子问题分解） | 40% |
| **知识库覆盖** | 知识库内问题 | 70% |
| | 需外部检索的问题 | 30% |

---

### 三、评估指标

HybridRAG 采用 ragas 框架的 4 个核心指标，覆盖 **检索质量** 和 **生成质量** 两个维度。

#### 3.1 指标总览

| 指标 | 维度 | 分数范围 | 是否需要参考答案 | 评估对象 |
|------|------|:--------:|:----------------:|----------|
| Faithfulness（忠实度） | 生成质量 | [0, 1] | ❌ | 回答 vs 上下文 |
| Answer Relevancy（答案相关性） | 生成质量 | [0, 1] | ❌ | 回答 vs 问题 |
| Context Relevance（上下文相关性） | 检索质量 | [0, 1] | ❌ | 上下文 vs 问题 |
| Context Recall（上下文召回率） | 检索质量 | [0, 1] | ✅ | 上下文 vs 参考答案 |

#### 3.2 Faithfulness（忠实度）

**定义**：衡量生成的回答是否忠实于检索到的上下文，即回答中的每个主张（claim）是否都能从上下文中推断出来。

**计算方法**：

1. LLM 从回答中识别所有事实性主张
2. 逐一检查每个主张是否可以从检索到的上下文中推断
3. 忠实度 = 可推断的主张数 / 总主张数

**应用场景**：检测模型是否产生了"幻觉"（hallucination），即生成了上下文中不存在的信息。

**期望基准**：≥ 0.85

#### 3.3 Answer Relevancy（答案相关性）

**定义**：衡量生成的回答与用户问题的匹配程度。

**计算方法**：

1. 根据回答反向生成一组问题
2. 计算每个生成问题与原始用户问题的余弦相似度（使用 `BAAI/bge-large-zh-v1.5` 嵌入模型）
3. 答案相关性 = 所有余弦相似度的平均值

**应用场景**：检测回答是否偏题、是否包含过多无关信息。

**期望基准**：≥ 0.80

#### 3.4 Context Relevance（上下文相关性）

**定义**：评估检索到的上下文是否与用户问题相关。

**计算方法**：

1. 两个独立的 LLM 评判者分别对上下文与问题的相关性打分（0、1、2）
2. 每个评分归一化到 [0, 1]
3. 最终得分 = 两个归一化评分的平均值

**应用场景**：评估检索管道的精准度，识别是否检索到了大量无关文档。

**期望基准**：≥ 0.75

#### 3.5 Context Recall（上下文召回率）

**定义**：衡量检索到的上下文是否覆盖了回答问题所需的全部关键信息。

**计算方法**：

1. LLM 将参考答案分解为多个关键信息点
2. 检查每个信息点是否能在检索到的上下文中找到
3. 召回率 = 被覆盖的信息点数 / 总信息点数

**应用场景**：评估检索管道的召回能力，需要提供人工标注的参考答案。

**期望基准**：≥ 0.80

---

### 四、实验设置

#### 4.1 软件环境

| 组件 | 版本/配置 |
|------|-----------|
| Python | ≥ 3.11 |
| ragas | ≥ 0.2.0 |
| LangGraph | ≥ 0.4.0 |
| LangChain | ≥ 1.2.0 |
| Milvus | 向量数据库（混合检索：稠密向量 + BM25 稀疏向量） |
| PostgreSQL | 检查点存储 + 知识库配置存储 |

#### 4.2 模型配置

| 用途 | 模型 | 说明 |
|------|------|------|
| **评估 LLM** | qwen-plus | 通过 AsyncOpenAI 客户端调用，用于 ragas 指标计算 |
| **评估 Embedding** | BAAI/bge-large-zh-v1.5 | HuggingFace 模型，用于 Answer Relevancy 的余弦相似度计算 |
| **RAG 主 LLM** | 通过 `QWEN_MODEL_NAME` 环境变量配置 | 用于查询增强、答案生成等 |
| **RAG Embedding** | qwen3-embedding | Ollama 本地部署，用于文档向量化和检索 |
| **重排序模型** | BAAI/bge-reranker-v2-m3 | 交叉编码器，用于检索结果重排序 |
| **文档评分模型** | BAAI/bge-large-zh-v1.5 | 语义相似度评分，阈值 0.7 |

#### 4.3 评估配置

在 `config/Config.py` 中通过 `RagSystemConfig` 控制评估开关：

```python
from config import RagSystemConfig

# 开启评估
config = RagSystemConfig(enable_eval=True)
```

评估所需的环境变量：

| 变量名 | 说明 |
|--------|------|
| `QWEN_API_KEY` | 通义千问 API Key |
| `QWEN_BASE_URL` | 通义千问 API Base URL |
| `HF_HOME` | HuggingFace 模型缓存目录 |

#### 4.4 评估触发方式

| 方式 | 说明 | 适用场景 |
|------|------|----------|
| **在线自动评估** | 设置 `enable_eval=True`，每次检索请求结束后自动在 `_final` 节点触发评估 | 开发调试、小规模验证 |
| **离线批量评估** | 构建 `EvalSample` 列表，调用 `evaluate_batch()` | 系统性评测、版本对比 |
| **单条手动评估** | 调用 `evaluate_sample()` 评估单条样本 | 问题排查、Case 分析 |

---

### 五、结果分析

#### 5.1 指标解读框架

| 指标组合 | 可能原因 | 改进方向 |
|----------|----------|----------|
| 高忠实度 + 高答案相关性 + 高上下文相关性 | 系统表现良好 | 维持现状 |
| 低忠实度 + 高上下文相关性 | LLM 产生幻觉，未忠实于上下文 | 优化生成 Prompt，增加上下文引用约束 |
| 高忠实度 + 低答案相关性 | 回答虽然忠实但偏题 | 优化查询增强策略，改进问题理解 |
| 低上下文相关性 + 低忠实度 | 检索质量差，导致生成也差 | 优化检索管道：调整重排序阈值、扩展知识库 |
| 高上下文相关性 + 低上下文召回率 | 检索精准但不全面 | 增加检索数量（top-k）、优化查询扩展策略 |

#### 5.2 影响因素分析

**检索阶段的关键因素：**

- **查询增强策略**：系统根据任务类型（8 种）动态选择增强策略（同义改写、查询分解、HyDE 预测等），不同策略对检索质量影响显著
- **重排序阈值**：交叉编码器 `CrossEncoderRanker` 使用 0.8 的分数阈值过滤低相关文档，阈值过高会降低召回率，过低会引入噪声
- **混合检索权重**：Milvus 混合检索使用 `weighted` 策略，稠密向量权重 0.7、稀疏向量（BM25）权重 0.3，权重配比影响不同类型查询的检索效果
- **知识库路由**：`QueryRouter` 将查询路由到不同知识库集合，路由准确性直接影响检索结果

**生成阶段的关键因素：**

- **上下文窗口**：检索文档经过重排序后拼接为上下文，过长可能导致关键信息被稀释
- **多跳推理**：复杂问题被分解为子问题逐步求解，子问题分解质量影响最终答案
- **对话上下文**：系统保留最近 5-6 轮对话历史，历史信息可能引入干扰

#### 5.3 改进建议

| 问题 | 建议 | 预期效果 |
|------|------|----------|
| 检索召回率不足 | 增加 `max_enhanced_queries` 配置值，启用更多查询增强策略 | 提升 Context Recall |
| 幻觉问题 | 在生成 Prompt 中增加"仅基于以下上下文回答"的约束 | 提升 Faithfulness |
| 检索噪声过多 | 提高 `CrossEncoderRanker` 的分数阈值（如 0.85） | 提升 Context Relevance |
| 多跳问题表现差 | 优化子问题分解 Prompt，增加推理上下文传递 | 提升整体答案质量 |
| 评估成本过高 | 仅在开发环境开启 `enable_eval`，生产环境关闭 | 降低 API 调用成本 |

---

### 六、案例研究

#### 6.1 案例一：单跳事实检索

**场景**：用户查询知识库中的具体事实信息。

**输入**：
```
用户问题：黄独分布在哪些地区？
```

**系统处理流程**：

1. **任务分析**：识别为 `FACT_RETRIEVAL` 类型，单跳问题
2. **查询增强**：启用 `paraphrase`（同义改写），生成多个查询变体
3. **检索**：通过 Milvus 混合检索（稠密向量 + BM25）获取候选文档
4. **重排序**：`CrossEncoderRanker` 对候选文档打分，过滤 score < 0.8 的文档
5. **生成**：基于筛选后的上下文生成回答

**评估结果示例**：

| 指标 | 分数 | 分析 |
|------|:----:|------|
| Faithfulness | 0.95 | 回答完全基于检索到的文档内容 |
| Answer Relevancy | 0.88 | 回答直接针对"分布地区"这一问题 |
| Context Relevance | 0.82 | 检索到的文档高度相关 |

**优势体现**：对于知识库覆盖的事实性问题，混合检索 + 重排序的组合能够精准定位相关文档，生成高质量回答。

---

#### 6.2 案例二：多跳复杂问题

**场景**：用户提出需要综合多个信息源的复杂问题。

**输入**：
```
用户问题：李呈瑞于哪一年参加红军？他获得过哪些勋章？他于哪一年逝世？他在抗战中担任过哪些职位？
```

**系统处理流程**：

1. **任务分析**：识别为 `FACT_RETRIEVAL` 类型，`is_multi_hop=True`
2. **子问题分解**：
   - Q1：李呈瑞于哪一年参加红军？
   - Q2：李呈瑞获得过哪些勋章？
   - Q3：李呈瑞于哪一年逝世？
   - Q4：李呈瑞在抗战中担任过哪些职位？
3. **逐步检索与回答**：每个子问题独立进行检索 → 重排序 → 生成，前序答案作为推理上下文注入后续步骤
4. **答案合成**：`_synthesize` 节点将所有子答案整合为最终回答

**评估结果示例**：

| 指标 | 分数 | 分析 |
|------|:----:|------|
| Faithfulness | 0.90 | 各子答案均基于检索文档，合成时可能引入少量推理 |
| Answer Relevancy | 0.85 | 最终回答覆盖了所有子问题 |
| Context Relevance | 0.78 | 部分子问题的检索结果包含少量无关信息 |

**优势体现**：多跳推理机制将复杂问题拆解为可管理的子任务，每步都有独立的检索和评分保障，最终通过合成生成完整回答。相比直接回答复杂问题，这种方式显著提升了回答的完整性和准确性。

**改进空间**：子问题之间的推理上下文传递可以进一步优化，确保后续子问题能充分利用前序答案中的关键信息。

---

### 七、使用指南

#### 7.1 在线评估（集成到 Graph）

```python
from config import RagSystemConfig
from src.graph import Graph

# 开启评估模式
config = RagSystemConfig(enable_eval=True)
graph = Graph(config)

# 正常运行，评估会在每次检索请求结束后自动触发
# 评估结果通过 monitor_task_status 记录到日志
await graph.start(messages=[...])
```

#### 7.2 离线批量评估

```python
from src.eval.ragas_eval import RagEvaluator, EvalSample

evaluator = RagEvaluator()

samples = [
    EvalSample(
        user_input="黄独分布在哪些地区？",
        response="黄独主要分布在中国南方地区...",
        retrieved_contexts=["黄独，又名...", "分布于长江以南..."],
        reference="黄独分布在中国长江以南各省区...",  # 可选
    ),
    EvalSample(
        user_input="如何安装Python？",
        response="安装Python的步骤如下...",
        retrieved_contexts=["Python安装指南...", "环境配置..."],
    ),
]

# 批量评估
report = await evaluator.evaluate_batch(samples)

# 打印报告
RagEvaluator.print_report(report)

# 保存为 JSON
RagEvaluator.save_report(report, "eval_report.json")
```

#### 7.3 评估报告输出示例

```
============================================================
RAG 评估报告
============================================================
样本数量: 2
----------------------------------------
  忠实度 (Faithfulness): 0.9250
  答案相关性 (Answer Relevancy): 0.8650
  上下文相关性 (Context Relevance): 0.8000
  上下文召回率 (Context Recall): 0.8500
----------------------------------------
各样本详情:

  [1] 黄独分布在哪些地区？
      faithfulness: 0.9500
      answer_relevancy: 0.8800
      context_relevance: 0.8200
      context_recall: 0.8500

  [2] 如何安装Python？
      faithfulness: 0.9000
      answer_relevancy: 0.8500
      context_relevance: 0.7800
      context_recall: N/A
============================================================
```

---

### 八、文件结构

```
src/eval/
├── __init__.py
└── ragas_eval.py          # 核心评估模块
    ├── EvalSample         # 评估样本数据结构
    ├── EvalScores         # 单条评估分数
    ├── EvalReport         # 批量评估报告
    └── RagEvaluator       # 评估器（支持单条/批量评估）

config/Config.py
└── RagSystemConfig
    └── enable_eval        # 评估开关（默认 False）

src/graph.py
└── Graph
    ├── _final()           # 最终节点，可选触发评估
    └── _run_eval()        # 评估执行方法
```
