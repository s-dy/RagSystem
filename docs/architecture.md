# HybridRAG 系统架构文档

## 一、整体架构

```
前端 (frontend/)
  │
  ├─ POST /api/chat/stream  ──→  server.py（SSE 流式接口）
  │                                   │
  │                                   └─ Graph.start_stream()
  │                                         │
  │                                         └─ LangGraph StateGraph
  │                                              │
  │                                              ├─ RouteNodeMixin     (路由/入口节点)
  │                                              ├─ RetrievalNodeMixin (检索节点)
  │                                              └─ GenerateNodeMixin  (生成节点)
  │
  └─ 知识库管理 API ──→ server.py（文档上传/分块/索引）
```

---

## 二、项目目录结构

```
src/
├── graph.py                    # LangGraph 状态机定义（State、Graph 类）
├── core/
│   ├── adapter.py              # 任务适配器
│   ├── memory_manager.py       # 记忆管理（PostgresStore）
│   ├── tools_pool.py           # MCP 工具池（外部搜索等）
│   └── exceptions.py           # 自定义异常
├── node/
│   ├── route/
│   │   ├── route_node.py       # RouteNodeMixin（入口/路由节点）
│   │   ├── query_enhancer.py   # 查询增强器
│   │   └── query_router.py     # 查询路由器
│   ├── retrieval/
│   │   ├── retrieval_node.py   # RetrievalNodeMixin（检索/评分节点）
│   │   └── fusion_retrieve.py  # 融合检索（Dense + Sparse + RRF）
│   └── generate/
│       └── generate_node.py    # GenerateNodeMixin（生成/合成/最终节点）
├── services/
│   ├── task_analyzer.py        # 任务分析器（类型识别、多跳检测）
│   ├── grade_model.py          # 文档评分器（逐文档 embedding 相似度）
│   ├── cross_encoder_ranker.py # 交叉编码器重排序
│   ├── time_transformer.py     # 时间表达式解析
│   ├── llm/                    # LLM 模型管理
│   ├── embedding/              # Embedding 模型管理
│   ├── storage/                # 向量数据库客户端
│   └── data_load/              # 数据加载
├── eval/
│   └── ragas_eval.py           # RAG 评估（RAGAS 指标）
└── observability/
    ├── logger.py               # 日志与监控
    └── langfuse_monitor.py     # Langfuse 追踪
```

---

## 三、LangGraph 状态机

### State 定义

```python
class State(MessagesState):
    original_query: str  # 原始查询
    task_characteristics: TaskCharacteristics  # 任务分析特征
    need_retrieval: bool  # 是否需要增强检索
    router_index: Dict[str, List[str]]  # 查询路由（collection → queries）
    search_content: str  # 检索内容整合（带来源编号）
    retrieved_documents: List[str]  # 检索到的文档列表
    retrieved_images: List  # CLIP 图片检索结果（RetrievedImage 列表），用于多模态生成
    retrieval_scores: List[float]  # 重排序分数
    run_count: int  # 运行次数
    grade_retry_count: int  # 评分重试计数
    answer: str  # 最终答案
    sub_questions: List[str]  # 待解决的子问题队列
    current_sub_question: str  # 当前正在处理的子问题
    reasoning_context: str  # 已解决的上下文
    reasoning_steps: List[dict]  # 推理过程记录
    conversation_summary: str  # 渐进式摘要（跨轮对话累积）
```

### 节点定义

| 节点                          | 所属 Mixin             | 职责                        |
|-----------------------------|----------------------|---------------------------|
| `retrieve_or_respond`       | `RouteNodeMixin`     | 入口节点：任务分析、对话压缩、判断是否需要检索   |
| `prepare_next_step`         | `RouteNodeMixin`     | 子问题分解或准备当前子问题             |
| `enhance_and_route_current` | `RouteNodeMixin`     | 查询增强 + 路由到目标知识库           |
| `fusion_retrieve`           | `RetrievalNodeMixin` | 融合检索（内部 + 外部）+ 近似去重 + 重排序 |
| `generate_current_answer`   | `GenerateNodeMixin`  | 生成当前查询的答案（支持流式）           |
| `synthesize`                | `GenerateNodeMixin`  | 多跳场景下合并子问题答案              |
| `final`                     | `GenerateNodeMixin`  | 最终输出：存储问答对、可选触发评估         |

### 图结构（边定义）

```
START
  │
  ▼
retrieve_or_respond
  │
  ├─ need_retrieval=false ──→ final ──→ END
  │
  └─ need_retrieval=true ──→ prepare_next_step
                                │
                                ▼
                        enhance_and_route_current
                                │
                                ▼
                          fusion_retrieve
                                │
                        ┌───────┴───────┐
                        │               │
                    grade=good      grade=bad
                        │               │
                        ▼               └──→ enhance_and_route_current（重试）
                generate_current_answer
                        │
                ┌───────┴───────┐
                │               │
          有子问题          无子问题
                │               │
                ▼               ▼
        prepare_next_step   synthesize
                                │
                                ▼
                              final ──→ END
```

---

## 四、核心模块详解

### 4.1 路由与入口（RouteNodeMixin）

**`retrieve_or_respond`**：统一入口节点

- 对话历史压缩（策略 1：跨轮压缩，策略 6：渐进式摘要）
- 对话上下文自适应提取（策略 2：预算内保留原文，超出部分 LLM 摘要）
- 任务分析（`TaskAnalyzer`）：识别任务类型、多跳特征
- 简单查询直接回复，复杂查询进入检索流程

**`prepare_next_step`**：子问题管理

- 多跳问题自动分解为子问题队列
- 从队列中取出下一个子问题

**`enhance_and_route_current`**：查询增强 + 路由

- 调用 `QueryEnhancer` 并行执行多种增强策略
- 调用 `QueryRouter` 将查询路由到目标 collection

### 4.2 检索（RetrievalNodeMixin）

**`fusion_retrieve`**：融合检索节点

- **内部检索**：Milvus 混合检索（Dense + Sparse），RRF 融合（k=60）
- **外部检索**：MCP 工具调用（Bing 搜索 + 网页爬取），支持重试
- **近似去重**：基于 embedding 余弦相似度（阈值 0.92），替代精确文本匹配
- **交叉编码器重排序**：`bge-reranker-v2-m3`，按阈值过滤
- **兜底机制**：重排序过滤掉所有文档时，保留原始排序的 top-1

**`grade_documents`**：检索结果评分（条件边）

- 逐文档评分（`bge-large-zh-v1.5` embedding 余弦相似度）
- 返回通过阈值的相关文档列表
- 评分失败触发重新增强检索，最多重试 2 次

### 4.3 生成（GenerateNodeMixin）

**`generate_current_answer`**：答案生成

- 结合检索内容 + 对话上下文 + 推理上下文生成答案
- 支持流式 token 输出

**`synthesize`**：多跳答案合成

- 仅在多跳场景下触发
- 将所有子问题的推理上下文整合为最终答案

**`final`**：最终输出

- 统一存储问答对到 `MemoryManager`（PostgresStore）
- 可选触发 RAGAS 评估

---

## 五、查询增强模块（QueryEnhancer）

### 增强策略

| 策略      | 说明           |
|---------|--------------|
| 同义改写    | 使用不同词汇表达相同含义 |
| 扩展改写    | 扩展查询以包含更多上下文 |
| 专业化改写   | 使用专业术语重写     |
| 查询分解    | 将复杂查询分解为子查询  |
| HyDE 预测 | 生成假设文档辅助检索   |

### 实现机制

- 5 种策略**并行执行**，提高效率
- 基于 `TaskCharacteristics` 动态选择策略
- 时间表达式解析（`TimeTransformer`）：将"上个月"等自然语言转换为具体日期

---

## 六、任务分析器（TaskAnalyzer）

### 任务类型

| 类型                        | 说明    | 示例                      |
|---------------------------|-------|-------------------------|
| `FACT_RETRIEVAL`          | 事实检索  | "什么是 RAG？"              |
| `PROCEDURAL_QUERY`        | 流程查询  | "如何安装 Python？"          |
| `ANALYTICAL_COMPARISON`   | 分析对比  | "比较 Python 和 Java 的优缺点" |
| `CREATIVE_GENERATION`     | 创造性生成 | "帮我写一首关于春天的诗"           |
| `COMPLEX_PLANNING`        | 复杂规划  | "制定一个项目计划"              |
| `MULTI_STEP_EXECUTION`    | 多步骤执行 | "首先…然后…最后…"             |
| `REAL_TIME_INTERACTION`   | 实时交互  | "现在北京天气怎么样？"            |
| `VALIDATION_VERIFICATION` | 验证核查  | "验证这个公式是否正确"            |

### 分析能力

- **正则模式匹配**：基于预定义模式识别任务类型
- **语言特征提取**：jieba 分词 + 词性标注，提取实体、动作动词
- **多跳问题检测**：加权评分（问句数量、连接词、查询长度、实体数量）
- **置信度评估**：多规则加权计算分类置信度

---

## 七、持久化与记忆

### 三层持久化

| 层级   | 技术                                           | 用途                  |
|------|----------------------------------------------|---------------------|
| 对话状态 | `AsyncPostgresSaver`（LangGraph Checkpointer） | 自动持久化 State，支持跨请求恢复 |
| 长期记忆 | `AsyncPostgresStore`（LangGraph Store）        | 存储用户偏好、问答对等结构化记忆    |
| 向量索引 | Milvus                                       | 文档 embedding 存储与检索  |

### 对话压缩策略

| 策略          | 触发条件                   | 机制                                  |
|-------------|------------------------|-------------------------------------|
| 策略 1：跨轮压缩   | 轮数 > 10 或 token > 4000 | 旧消息 LLM 摘要 → SystemMessage，保留最近 K 轮 |
| 策略 2：上下文自适应 | 每次生成前                  | 预算内保留原文，超出部分 LLM 摘要补充               |
| 策略 6：渐进式摘要  | 每 5 轮触发                | 带锚点防护的增量摘要，防止关键信息漂移                 |

---

## 八、数据流处理

### 单跳查询流程

```
用户输入 → retrieve_or_respond（任务分析 + 压缩）
         → prepare_next_step
         → enhance_and_route_current（查询增强 + 路由）
         → fusion_retrieve（混合检索 + 去重 + 重排序）
         → grade_documents（逐文档评分）
         → generate_current_answer（答案生成）
         → synthesize（跳过）
         → final（存储 + 输出）
```

### 多跳查询流程

```
用户输入 → retrieve_or_respond（识别多跳特征）
         → prepare_next_step（分解为子问题 [Q1, Q2, Q3]）
         ┌─────────────────────────────────────────────┐
         │ 循环处理每个子问题：                          │
         │   enhance_and_route_current → fusion_retrieve │
         │   → grade_documents → generate_current_answer │
         │   → prepare_next_step（取下一个子问题）        │
         └─────────────────────────────────────────────┘
         → synthesize（整合所有子问题答案）
         → final（存储 + 输出）
```

### 直接回复流程

```
用户输入 → retrieve_or_respond（判断无需检索，直接生成答案）
         → final（存储 + 输出）
```

---

## 九、关键配置项

| 配置                              | 默认值     | 说明               |
|---------------------------------|---------|------------------|
| `enable_eval`                   | `True`  | 是否启用 RAGAS 评估    |
| `enable_parent_child_retrieval` | `False` | 是否启用父子文档检索       |
| `reranker_threshold`            | `0.8`   | 重排序过滤阈值          |
| `grader_threshold`              | `0.7`   | 文档评分阈值           |
| `enable_conversation_compress`  | `True`  | 是否启用对话压缩         |
| `max_conversation_turns`        | `10`    | 触发压缩的最大轮数        |
| `max_conversation_tokens`       | `4000`  | 触发压缩的最大 token 数  |
| `keep_recent_turns`             | `3`     | 压缩时保留的最近轮数       |
| `max_context_tokens`            | `2000`  | 上下文自适应的 token 预算 |
| `incremental_summary_interval`  | `5`     | 渐进式摘要触发间隔（轮数）    |

---

## 十、多模态配置项（`MultimodalConfig`）

| 环境变量                    | 默认值                            | 说明                                 |
|-------------------------|--------------------------------|------------------------------------|
| `CLIP_MODEL_PATH`       | `openai/clip-vit-base-patch32` | CLIP 模型路径，支持本地路径或 HuggingFace 模型名  |
| `IMAGE_SCORE_THRESHOLD` | `0.25`                         | 图片检索相似度阈值，低于此值的图片不传入 VLM           |
| `MAX_IMAGES_PER_QUERY`  | `3`                            | 每次查询最多传入 VLM 的图片数量                 |
| `CAPTION_MODEL_NAME`    | `""`                           | VLM Caption 生成模型名，为空则跳过 Caption 生成 |

多模态功能依赖可选依赖组，安装方式：

```bash
pip install -e ".[multimodal]"
# 等价于：pip install pypdf pdfplumber transformers torch pillow
```

详细说明见 [`docs/multimodal_rag/architecture.md`](./multimodal_rag/architecture.md)。
