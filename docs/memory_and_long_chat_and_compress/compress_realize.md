# 对话压缩与上下文管理 — 实现方案

本文档记录跨轮对话历史压缩（策略 1）、对话上下文窗口自适应（策略 2）、渐进式摘要（策略 6）的具体实现细节。

---

## 一、涉及文件与修改概览

| 文件 | 修改内容 |
|------|----------|
| `config.py` | `RagSystemConfig` 新增 7 个压缩相关配置项 |
| `utils/message_util.py` | 新增 3 个 prompt 模板、2 个 token 估算函数、6 个核心函数 |
| `src/graph.py` | `State` 新增 `conversation_summary` 字段 |
| `src/node/route/route_node.py` | `__retrieve_or_respond` 入口集成策略 1/2/6 |
| `src/node/generate/generate_node.py` | `get_conversation_context` → `get_conversation_context_adaptive` |

---

## 二、配置项（`config.py` → `RagSystemConfig`）

```python
# === 对话压缩相关配置 ===
enable_conversation_compress: bool = True       # 是否启用跨轮对话历史压缩
max_conversation_turns: int = 10                # 最大对话轮数（超过则触发压缩）
max_conversation_tokens: int = 4000             # 最大对话 token 数（超过则触发压缩）
keep_recent_turns: int = 3                      # 压缩时保留最近的轮数
max_compress_tokens: int = 6000                 # 单次压缩的最大 token 数（超过则分批）
incremental_summary_interval: int = 5           # 渐进式摘要触发间隔（每隔多少轮）
max_context_tokens: int = 2000                  # 自适应上下文窗口的 token 预算
```

---

## 三、策略 1：跨轮对话历史压缩

### 触发条件

在 `__retrieve_or_respond` 入口处，**双重触发条件**（任一满足即触发）：

| 触发条件 | 场景 |
|----------|------|
| 轮数超限 | `len(messages) > max_conversation_turns * 2` |
| Token 超限 | `estimate_messages_tokens(messages) > max_conversation_tokens` |

### 核心函数

**`compress_conversation_history(llm, messages, keep_recent, max_compress_tokens)`**

- 保留最近 `keep_recent` 轮（每轮 2 条消息）的原始消息
- 将更早的 Human/AI 消息拼接为文本，调用 LLM 压缩为摘要
- 如果旧消息总 token 数超过 `max_compress_tokens`，调用 `_batch_compress` 分批压缩
- 返回 `[SystemMessage("[对话历史摘要]\n{summary}")] + 最近 K 轮原始消息`

**`_batch_compress(llm, texts, max_tokens_per_batch)`**

- 将文本列表按 token 预算分成若干批次
- 每批次独立调用 LLM 压缩为段摘要
- 多个批次时，将段摘要合并后再做一次最终压缩

### State 更新机制

LangGraph 的 `MessagesState` 默认使用 `add_messages`（追加语义），不能直接返回压缩后的消息列表。通过 `build_remove_and_replace_messages` 函数：

1. 对所有旧消息生成 `RemoveMessage(id=msg.id)` 删除指令
2. 将压缩后的消息（SystemMessage 摘要 + 保留的近期消息）作为新消息追加

```python
compress_result["messages"] = build_remove_and_replace_messages(messages, compressed)
```

### 调用位置

`route_node.py` → `__retrieve_or_respond` 方法最前面，检索/生成之前执行。压缩结果通过 `**compress_result` 合并到返回值中。

---

## 四、策略 2：对话上下文窗口自适应

### 核心函数

**`get_conversation_context_adaptive(messages, llm, max_context_tokens, min_messages, max_messages)`**

两层策略：

| 层级 | 处理对象 | 方式 | 信息保真度 |
|------|----------|------|:----------:|
| 第一层 | 最近的消息（预算内） | 保留原文 | 100% |
| 第二层 | 更早的消息（预算外） | LLM 摘要 | 关键信息保留 |

**第一层**：从最新消息向前遍历，在 `max_context_tokens` 的 token 预算内尽可能多地保留原始消息（至少 `min_messages` 条，至多 `max_messages` 条）。

**第二层**：对超出预算的更早消息：
- 优先检查是否有策略 1 生成的 `[对话历史摘要]` SystemMessage，有则直接复用，避免二次压缩
- 否则调用 LLM 生成 2-3 句话的摘要作为补充上下文

### 与策略 1 的协作

策略 1 先执行（入口节点最前面），策略 2 后执行（基于策略 1 处理后的 messages 提取上下文）。策略 2 在处理第二层时，如果检测到 `SystemMessage` 类型且内容包含 `[对话历史摘要]` 标记，则直接复用该摘要，不再二次压缩。

### 替换范围

| 文件 | 原调用 | 新调用 |
|------|--------|--------|
| `route_node.py` → `__retrieve_or_respond` | `get_conversation_context(messages, num_messages=5)` | `get_conversation_context_adaptive(messages, self.llm, max_context_tokens=...)` |
| `generate_node.py` → `__generate_current_answer` | `get_conversation_context(state['messages'], num_messages=6)` | `get_conversation_context_adaptive(state['messages'], self.llm, max_context_tokens=...)` |

注意：`route_node.py` 中 `__prepare_next_step` 和 `__enhance_and_route_current` 的 `get_conversation_context` 调用保持不变，因为这些是子问题分解和查询增强的上下文，不需要自适应处理。

---

## 五、策略 6：渐进式摘要

### 触发条件

**`should_trigger_incremental_summary(messages, interval)`**

每隔 `interval` 轮（默认 5 轮）触发一次，判断逻辑：`human_count % interval == 0`。

### 核心函数

**`incremental_summarize_with_anchors(llm, existing_summary, new_messages, anchor_facts)`**

- 将新增消息与已有摘要合并，生成更新后的完整摘要
- 支持**锚点防护**：`anchor_facts` 列表中的关键事实会作为硬约束注入 prompt，防止多次迭代后信息漂移丢失
- 当前实现中 `anchor_facts` 默认为空，后续策略 5（分级保留）实施后，可将 `HIGH` 重要性消息提取为锚点

### State 字段

`State` 中新增 `conversation_summary: str` 字段，用于持久化渐进式摘要。每次触发增量摘要后，更新该字段：

```python
compress_result["conversation_summary"] = updated_summary
```

该字段通过 LangGraph 的 checkpointer 自动持久化到 PostgreSQL，下次对话直接读取。

### 调用位置

`route_node.py` → `__retrieve_or_respond`，在策略 1 之后、策略 2 之前执行。

---

## 六、执行流程

```
用户消息进入 __retrieve_or_respond
    │
    ├─ 策略 1：检查是否需要压缩（轮数超限 OR token 超限）
    │   └─ 是 → compress_conversation_history → build_remove_and_replace_messages
    │
    ├─ 策略 6：检查是否触发渐进式摘要（每 N 轮）
    │   └─ 是 → incremental_summarize_with_anchors → 更新 conversation_summary
    │
    ├─ 策略 2：get_conversation_context_adaptive（自适应提取上下文）
    │   ├─ 第一层：预算内保留原文
    │   └─ 第二层：预算外摘要补充（复用策略 1 的摘要或新生成）
    │
    ├─ 任务分析 + 检索判断
    │
    └─ 返回结果（合并 compress_result）
```

---

## 七、与现有功能的关系

| 维度 | `compress_conversation_history` | `compress_reasoning_context` |
|------|:------:|:------:|
| 作用范围 | 跨轮对话（`messages` 列表） | 单次请求内多跳推理（`reasoning_context` 字符串） |
| 触发条件 | `messages` 轮数超限 **或** token 数超限 | `reasoning_context` 字符数 > `max_reasoning_chars` |
| 压缩对象 | 旧的 Human/AI 消息 → SystemMessage 摘要 | 子问题解答记录 → 精简摘要 |
| 保留策略 | 保留最近 K 轮原始消息 | 保留所有关键事实和结论 |

两者互不干扰，分别服务于不同层级的上下文管理。

---

## 八、Token 估算

采用简单的字符数估算方式，适用于混合中英文场景：

```python
def estimate_token_count(text: str) -> int:
    return int(len(text) * 0.6)  # 字符数 * 0.6 ≈ token 数
```

该估算方式在中文为主的场景下误差约 ±15%，足以满足触发条件判断的精度要求。如需更精确的估算，可替换为 `tiktoken` 库。

---

## 九、后续优化方向

1. **策略 5（分级保留）**：为消息标记重要性等级（HIGH/MEDIUM/LOW），压缩时将 HIGH 消息原文作为硬约束注入 prompt，渐进式摘要时提取为锚点
2. **精确 Token 计算**：引入 `tiktoken` 替代字符数估算，提升触发条件的精度
3. **异步压缩**：将压缩操作放到后台任务，避免阻塞用户请求的首 token 延迟
