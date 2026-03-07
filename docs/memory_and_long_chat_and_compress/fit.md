## 多轮对话与上下文管理优化方案

### 一、现状分析

当前项目的多轮对话能力分散在多个模块中：`utils/message_util.py` 提供对话上下文提取，`src/node/generate/generate.py` 中的 `compress_reasoning_context()` 处理单次请求内的推理上下文压缩，`src/core/memory_manager.py` 提供基于 PostgreSQL 的用户记忆管理，`src/graph.py` 通过 LangGraph `AsyncPostgresSaver` checkpointer 持久化 `messages` 状态。

| 模块 | 当前实现 | 说明 |
|------|:--------:|------|
| 对话历史持久化 | `AsyncPostgresSaver` checkpointer | 基于 `thread_id` 自动持久化 `MessagesState.messages` |
| 对话上下文提取 | `get_conversation_context()` 取最近 N 条 | 固定取最近 3-5 条消息拼接为文本 |
| 推理上下文压缩 | `compress_reasoning_context()` | 单次请求内多跳子问题的 `reasoning_context` 超长时 LLM 压缩 |
| 用户记忆管理 | `MemoryManager` + PostgreSQL Store | 偏好、对话、上下文记忆的 CRUD，检索功能已实现 |
| 对话历史压缩 | 无 | 跨轮对话 `messages` 无限增长，无压缩机制 |
| 对话摘要 | 无 | 长对话无法生成阶段性摘要供后续轮次参考 |

**主要局限性：**

- `messages` 随对话轮次无限增长，每次请求都将完整历史传入 LLM，可能超出上下文窗口或显著增加 token 消耗
- `get_conversation_context()` 固定取最近 3-5 条消息，早期对话中的关键信息（如用户偏好、已确认的事实）会被丢弃
- `compress_reasoning_context()` 仅服务于单次请求内的多跳推理，不处理跨轮对话的历史压缩
- `MemoryManager` 的 `search_related_memories()` 和 `get_recent_conversation_memories()` 已实现基于键前缀扫描的检索，但尚未在业务流程中集成调用
- 无法区分"重要消息"和"闲聊消息"，所有消息等权重保留
- 缺少对话主题切换检测，主题切换后旧主题的上下文仍会干扰新主题的检索和生成

---

### 二、优化策略

#### 策略 1：跨轮对话历史压缩（🔴 高优先级）--采用

**问题**：`MessagesState.messages` 随对话轮次无限增长。当对话超过 10-20 轮时，完整历史可能超出 LLM 上下文窗口（如 8K/32K），导致生成失败或关键信息被截断。此外，即使轮数在允许范围内，如果单条消息内容很长（如用户粘贴了大段代码或文档），总 token 数也可能远超上下文窗口。

**方案**：在 `__retrieve_or_respond` 入口节点**同时基于轮数和 token 数**双重判断是否需要压缩，任一条件触发即执行压缩。使用 LLM 将旧消息压缩为摘要，保留最近 K 轮原始消息：

```python
# config.py 新增配置
@dataclass
class RagSystemConfig:
    # ... 现有配置 ...
    # 对话历史压缩
    enable_conversation_compress: bool = True
    max_conversation_turns: int = 10       # 超过此轮数触发压缩
    max_conversation_tokens: int = 8000    # 超过此 token 数触发压缩
    keep_recent_turns: int = 4             # 压缩后保留最近 K 轮原始消息


def estimate_token_count(text: str) -> int:
    """粗略估算文本的 token 数

    中文：1 个字符 ≈ 1.5 token
    英文：1 个单词 ≈ 1 token，平均每个单词约 5 个字符
    混合场景取折中值：字符数 * 0.6
    """
    return int(len(text) * 0.6)


def estimate_messages_tokens(messages: List[AnyMessage]) -> int:
    """估算消息列表的总 token 数"""
    return sum(
        estimate_token_count(msg.content)
        for msg in messages
        if isinstance(msg, (HumanMessage, AIMessage, SystemMessage)) and msg.content
    )


# src/node/generate/generate.py 新增压缩函数
conversation_compress_prompt = """
请将以下多轮对话历史压缩为简洁的摘要，保留所有关键信息。

对话历史：
{conversation_history}

输出要求：
- 保留用户的核心问题和关键意图
- 保留助手给出的重要结论和事实
- 保留用户确认或否定的关键信息
- 去除寒暄、重复、过渡性内容
- 以第三人称客观描述
"""

async def compress_conversation_history(
    llm: BaseChatModel,
    messages: List[AnyMessage],
    keep_recent: int = 4,
    max_compress_tokens: int = 6000,
) -> List[AnyMessage]:
    """将旧对话历史压缩为摘要 SystemMessage，保留最近 K 轮原始消息

    Args:
        max_compress_tokens: 单次压缩调用的 token 安全上限。
            当旧消息总 token 超过此值时，分批压缩再合并，
            避免压缩调用本身超出 LLM 上下文窗口。
    """
    # 每轮 = 1 条 HumanMessage + 1 条 AIMessage
    keep_count = keep_recent * 2
    if len(messages) <= keep_count:
        return messages

    old_messages = messages[:-keep_count]
    recent_messages = messages[-keep_count:]

    # 将旧消息按角色拼接为文本
    old_texts = [
        f"{'用户' if isinstance(m, HumanMessage) else '助手'}: {m.content}"
        for m in old_messages
        if isinstance(m, (HumanMessage, AIMessage))
    ]
    history_text = "\n".join(old_texts)

    # 防护：如果旧消息总量超出单次压缩的安全上限，分批压缩
    if estimate_token_count(history_text) > max_compress_tokens:
        summary = await _batch_compress(llm, old_texts, max_compress_tokens)
    else:
        prompt = ChatPromptTemplate.from_messages([
            ("system", conversation_compress_prompt),
            ("human", "请压缩上述对话历史。")
        ])
        chain = prompt | llm | StrOutputParser()
        summary = await chain.ainvoke({"conversation_history": history_text})

    # 用 SystemMessage 承载摘要，放在最前面
    summary_message = SystemMessage(content=f"[对话历史摘要]\n{summary}")
    return [summary_message] + recent_messages


async def _batch_compress(
    llm: BaseChatModel,
    texts: List[str],
    max_tokens_per_batch: int,
) -> str:
    """分批压缩：将文本列表按 token 预算分段，逐段压缩后合并为最终摘要

    流程：
    1. 将 texts 按 max_tokens_per_batch 分成若干批次
    2. 每批次独立调用 LLM 压缩为段摘要
    3. 将所有段摘要合并，再做一次最终压缩
    """
    # 分批
    batches = []
    current_batch = []
    current_tokens = 0
    for text in texts:
        text_tokens = estimate_token_count(text)
        if current_tokens + text_tokens > max_tokens_per_batch and current_batch:
            batches.append("\n".join(current_batch))
            current_batch = [text]
            current_tokens = text_tokens
        else:
            current_batch.append(text)
            current_tokens += text_tokens
    if current_batch:
        batches.append("\n".join(current_batch))

    # 逐批压缩
    prompt = ChatPromptTemplate.from_messages([
        ("system", conversation_compress_prompt),
        ("human", "请压缩上述对话历史。")
    ])
    chain = prompt | llm | StrOutputParser()

    batch_summaries = []
    for batch_text in batches:
        batch_summary = await chain.ainvoke({"conversation_history": batch_text})
        batch_summaries.append(batch_summary)

    # 如果只有一个批次，直接返回
    if len(batch_summaries) == 1:
        return batch_summaries[0]

    # 多个批次：合并段摘要，做最终压缩
    merged_text = "\n---\n".join(batch_summaries)
    final_summary = await chain.ainvoke({"conversation_history": merged_text})
    return final_summary
```

**调用位置**：在 `RouteNodeMixin.__retrieve_or_respond()` 入口处，检索/生成之前：

```python
async def __retrieve_or_respond(self, state, config, store):
    messages = state["messages"]

    # 跨轮对话历史压缩：轮数超限 或 token 数超限，任一触发即压缩
    if self.config.enable_conversation_compress:
        turns_exceeded = len(messages) > self.config.max_conversation_turns * 2
        tokens_exceeded = estimate_messages_tokens(messages) > self.config.max_conversation_tokens
        if turns_exceeded or tokens_exceeded:
            compressed = await compress_conversation_history(
                self.llm, messages, keep_recent=self.config.keep_recent_turns
            )
            # 回写 State：MessagesState 默认使用 add_messages（追加语义），
            # 不能直接返回 {"messages": compressed}，否则会追加而非替换。
            # 需要使用 LangGraph 的 RemoveMessage 机制：
            #   1. 对所有旧消息生成 RemoveMessage(id=msg.id) 删除指令
            #   2. 将压缩后的消息（SystemMessage 摘要 + 保留的近期消息）作为新消息追加
            # 示例：
            remove_ops = [RemoveMessage(id=msg.id) for msg in messages]
            return {
                "messages": remove_ops + compressed,
                # ... 其他字段 ...
            }

    query = get_last_user_msg(messages)
    conversation_context = get_conversation_context(messages, num_messages=5)
    # ... 后续逻辑不变 ...
```

**双重触发条件说明**：

| 触发条件 | 场景 | 示例 |
|----------|------|------|
| 轮数超限 | 对话轮次多但每条消息较短 | 20 轮简短问答，每条 50 字 |
| Token 超限 | 轮次少但单条消息很长 | 3 轮对话，但用户粘贴了 5000 字的代码片段 |
| 双重触发 | 轮次多且消息长 | 15 轮对话，其中多条包含长文本 |

这样即使在允许的轮数范围内，如果总 token 数已经接近 LLM 上下文窗口上限，也会提前触发压缩，避免生成失败。

**与 `compress_reasoning_context` 的关系**：

| 维度 | `compress_conversation_history` | `compress_reasoning_context` |
|------|:------:|:------:|
| 作用范围 | 跨轮对话（`messages` 列表） | 单次请求内多跳推理（`reasoning_context` 字符串） |
| 触发条件 | `messages` 轮数超限 **或** token 数超限 | `reasoning_context` 字符数 > `max_reasoning_chars` |
| 压缩对象 | 旧的 Human/AI 消息 → SystemMessage 摘要 | 子问题解答记录 → 精简摘要 |
| 保留策略 | 保留最近 K 轮原始消息 | 保留所有关键事实和结论 |

两者互不干扰，分别服务于不同层级的上下文管理。

**预期收益**：避免长对话 token 溢出（无论是轮次多还是单条消息长），降低 API 调用成本，同时通过摘要保留早期对话的关键信息。

---

#### 策略 2：对话上下文窗口自适应（🔴 高优先级）--采用

**问题**：`get_conversation_context()` 固定取最近 `num_messages` 条消息，存在两个缺陷：
1. 无法根据对话复杂度和 LLM 上下文窗口动态调整——简单追问可能只需 1-2 条上下文，而复杂的多轮讨论可能需要更多
2. 如果仅基于 token 预算截断，超出预算的旧消息会被直接丢弃，其中可能包含对当前问题至关重要的信息（如用户在第 2 轮确认的关键约束条件）

**方案**：采用**"token 预算内保留原文 + 超出部分 LLM 摘要补充"**的两层策略，既控制 token 消耗，又避免信息丢失：

```python
async def get_conversation_context_adaptive(
    messages: List[AnyMessage],
    llm: BaseChatModel,
    max_context_tokens: int = 2000,
    min_messages: int = 2,
    max_messages: int = 10,
) -> str:
    """自适应对话上下文提取：预算内保留原文，超出部分摘要补充

    两层策略：
    1. 从最新消息向前遍历，在 token 预算内尽可能多地保留原始消息
    2. 对超出预算的更早消息，使用 LLM 生成摘要作为补充上下文
    这样既控制了总 token 数，又不会丢失早期对话中的关键信息
    """
    if not messages:
        return ""

    # 粗略估算：混合中英文场景，字符数 * 0.6 ≈ token 数
    char_budget = int(max_context_tokens / 0.6)

    # 第一层：从最新消息向前，在预算内保留原文
    retained_parts = []
    total_chars = 0
    cutoff_index = len(messages)  # 记录预算内覆盖到的最早消息位置

    for i, msg in enumerate(reversed(messages)):
        if not isinstance(msg, (HumanMessage, AIMessage)):
            continue
        role = "用户: " if isinstance(msg, HumanMessage) else "助手: "
        text = f"{role}{msg.content}"

        if total_chars + len(text) > char_budget and len(retained_parts) >= min_messages:
            cutoff_index = len(messages) - i
            break
        if len(retained_parts) >= max_messages:
            cutoff_index = len(messages) - i
            break

        retained_parts.append(text)
        total_chars += len(text)

    retained_parts.reverse()

    # 第二层：对预算外的更早消息，生成摘要补充
    earlier_messages = [
        msg for msg in messages[:cutoff_index]
        if isinstance(msg, (HumanMessage, AIMessage))
    ]

    summary_prefix = ""
    if earlier_messages:
        # 检查是否有已压缩的 SystemMessage 摘要（策略 1 生成的），
        # 如果有则直接复用，避免对已有摘要二次压缩导致信息损失
        existing_summaries = [
            msg.content for msg in messages[:cutoff_index]
            if isinstance(msg, SystemMessage) and "[对话历史摘要]" in msg.content
        ]
        if existing_summaries:
            # 直接复用策略 1 生成的摘要，不再二次压缩
            summary_prefix = existing_summaries[-1] + "\n\n"
        else:
            earlier_text = "\n".join(
                f"{'用户' if isinstance(m, HumanMessage) else '助手'}: {m.content}"
                for m in earlier_messages
            )
            # 防护：如果早期消息本身也很长，截取最近的部分做摘要
            # （更早的消息与当前问题的相关性通常更低）
            max_summary_chars = int(4000 / 0.6)  # 约 4000 token 的安全上限
            if len(earlier_text) > max_summary_chars:
                earlier_text = earlier_text[-max_summary_chars:]

            prompt = ChatPromptTemplate.from_template(
                "请用 2-3 句话概括以下对话的要点，只保留关键事实和结论：\n\n{history}"
            )
            chain = prompt | llm | StrOutputParser()
            summary = await chain.ainvoke({"history": earlier_text})
            summary_prefix = f"[早期对话摘要] {summary}\n\n"

    return summary_prefix + "\n".join(retained_parts)
```

**两层策略对比**：

| 层级 | 处理对象 | 方式 | 信息保真度 |
|------|----------|------|:----------:|
| 第一层 | 最近的消息（预算内） | 保留原文 | 100% |
| 第二层 | 更早的消息（预算外） | LLM 摘要 | 关键信息保留 |

**预期收益**：在 token 预算内最大化利用对话历史，同时通过摘要补充避免早期关键信息丢失。相比简单截断，信息完整性显著提升。

---

#### 策略 3：MemoryManager 对话记忆检索增强（🟡 中优先级） -- 采用

**问题**：`MemoryManager` 当前具备偏好、对话、上下文三类记忆的保存能力，但检索侧的 `get_recent_conversation_memories()` 和 `search_related_memories()` 均返回空列表。策略 4（主题切换检测）保存到 PostgreSQL 的旧主题上下文无法被后续对话检索恢复，用户回到旧主题时上下文丢失。

**现有 MemoryManager 功能**：

| 功能 | 方法 | 状态 |
|------|------|:----:|
| 用户偏好 | `save/get_user_preference` | ✅ 已实现 |
| 对话记忆保存 | `save_conversation_memory` | ✅ 已实现 |
| 对话记忆检索 | `get_recent_conversation_memories` | ✅ 已实现 |
| 上下文记忆 | `save/get_contextual_memory` | ✅ 已实现 |
| 相关记忆搜索 | `search_related_memories` | ✅ 已实现 |

**方案**：实现 `get_recent_conversation_memories()` 的真实检索逻辑，基于 PostgreSQL Store 键前缀扫描获取指定类型的对话记忆，并在入口节点注入相关记忆：

**配套改动 — 入口节点注入记忆**：

在 `__retrieve_or_respond` 中，检索前查询用户相关的对话记忆，注入到 `conversation_context`：

```python
# 检索与当前 query 相关的历史记忆
related_memories = await memory_manager.search_related_memories(
    user_id=user_id, query=query, limit=3
)
if related_memories:
    memory_context = "\n".join(
        f"[历史记忆] {m.get('content', '')}" for m in related_memories
    )
    conversation_context = f"{memory_context}\n\n{conversation_context}"
```

**预期收益**：策略 4 保存的主题上下文可被检索恢复，用户回到旧主题时不再丢失上下文。同时支持跨会话的对话记忆检索，提升多轮对话的连贯性。

---

#### 策略 4：对话主题切换检测（🟡 中优先级）

**问题**：用户在同一会话中切换话题时（如从"量子计算"切换到"今天天气"），旧主题的对话上下文会干扰新主题的检索和生成，导致答案偏离。

**方案**：采用**"轻量级初筛 + LLM 精确判断"**两级检测，避免每轮都调用 LLM 增加延迟和成本：

```python
from langchain_core.embeddings import Embeddings
import numpy as np

def quick_topic_similarity(
    query: str,
    last_query: str,
    embedding_model: Embeddings,
    threshold: float = 0.3,
) -> bool:
    """轻量级初筛：用 embedding 余弦相似度判断是否可能切换主题

    Returns:
        True 表示相似度低于阈值，可能发生了主题切换，需要进一步 LLM 确认
        False 表示相似度足够高，大概率同一主题，跳过 LLM 调用
    """
    if not last_query:
        return False
    embeddings = embedding_model.embed_documents([query, last_query])
    similarity = np.dot(embeddings[0], embeddings[1]) / (
        np.linalg.norm(embeddings[0]) * np.linalg.norm(embeddings[1])
    )
    return similarity < threshold


topic_switch_prompt = """
判断用户的最新问题是否与之前的对话主题相关。

最近对话：
{conversation_context}

最新问题：{query}

如果主题发生了明显切换，输出 "TOPIC_SWITCH"。
如果仍在同一主题下，输出 "SAME_TOPIC"。
只输出上述两个标签之一，不要解释。
"""

async def detect_topic_switch(
    llm: BaseChatModel,
    query: str,
    conversation_context: str,
    last_query: str,
    embedding_model: Embeddings,
) -> bool:
    """两级主题切换检测：embedding 初筛 + LLM 精确判断

    第一级：计算当前 query 与上一轮 query 的 embedding 余弦相似度，
           相似度高于阈值则直接判定为同一主题，跳过 LLM 调用（节省 0.5-2 秒延迟）。
    第二级：相似度低于阈值时，调用 LLM 做精确判断。
    """
    if not conversation_context:
        return False

    # 第一级：轻量级初筛
    might_switch = quick_topic_similarity(query, last_query, embedding_model)
    if not might_switch:
        return False  # 相似度高，大概率同一主题，跳过 LLM

    # 第二级：LLM 精确判断
    prompt = ChatPromptTemplate.from_template(topic_switch_prompt)
    chain = prompt | llm | StrOutputParser()
    result = await chain.ainvoke({
        "query": query,
        "conversation_context": conversation_context,
    })
    return "TOPIC_SWITCH" in result.upper()
```

**调用位置**：在 `__retrieve_or_respond` 中，获取 `conversation_context` 后判断：

```python
if await detect_topic_switch(self.llm, query, conversation_context, last_query, self.embedding):
    # 主题切换：将旧主题上下文压缩为摘要保存到 MemoryManager，
    # 而非直接清空（用户可能回到旧主题）
    await memory_manager.save_conversation_memory(
        user_id=user_id,
        thread_id=thread_id,
        memory_type="topic_context",
        content={"topic_summary": conversation_context, "query": last_query},
    )
    conversation_context = ""  # 当前请求不传旧上下文
    monitor_task_status("检测到主题切换，旧主题上下文已保存到记忆，当前上下文已重置")
```

**主题切换后的上下文保留策略**：

| 操作 | 直接清空（旧方案） | 保存到记忆（新方案） |
|------|:------------------:|:--------------------:|
| 当前请求 | 不传旧上下文 ✅ | 不传旧上下文 ✅ |
| 用户回到旧主题 | 上下文丢失 ❌ | 从 MemoryManager 恢复 ✅ |
| 存储开销 | 无 | PostgreSQL 一条记录 |

**预期收益**：大部分同主题追问场景跳过 LLM 调用（节省延迟和成本），主题切换后旧上下文可恢复。

---

#### 策略 5：对话历史分级保留（🟢 低优先级）

**问题**：当前所有消息等权重保留，压缩时无法区分重要消息（如用户确认的事实、关键决策）和低价值消息（如寒暄、确认收到）。

**方案**：为消息添加重要性标记，压缩时优先保留高重要性消息。

重要性判断采用**"规则初筛 + 对话结构分析"**结合的方式，而非仅依赖消息长度或开头短语（长消息可能只是用户粘贴的无关文本，短消息"杭州"可能是关键的地点确认）：

```python
from enum import IntEnum

class MessageImportance(IntEnum):
    LOW = 1       # 寒暄、确认、过渡
    MEDIUM = 2    # 普通问答
    HIGH = 3      # 关键事实、用户偏好、决策确认

def classify_message_importance(
    message: AnyMessage,
    prev_message: AnyMessage = None,
) -> MessageImportance:
    """基于规则和对话结构判断消息重要性

    判断逻辑：
    1. 对话结构分析：如果上一条是助手的提问/确认请求，
       当前用户回复大概率是关键信息（确认、否定、补充约束）→ HIGH
    2. 内容模式匹配：寒暄/确认短语 → LOW
    3. 信息密度：包含数字、专有名词、列表等结构化信息 → HIGH
    4. 默认 → MEDIUM
    """
    content = message.content.strip()

    # 规则 1：对话结构 — 用户对助手提问的直接回答通常是关键信息
    if isinstance(message, HumanMessage) and prev_message and isinstance(prev_message, AIMessage):
        prev_content = prev_message.content
        # 助手上一条包含提问/确认请求的标志
        question_indicators = ["？", "?", "请确认", "是否", "需要", "选择", "哪个"]
        if any(indicator in prev_content for indicator in question_indicators):
            return MessageImportance.HIGH

    # 规则 2：纯寒暄/确认短语 → LOW
    low_patterns = ["好的", "谢谢", "明白了", "收到", "嗯", "ok", "OK", "好", "行"]
    if content in low_patterns or (len(content) <= 5 and any(content.startswith(p) for p in low_patterns)):
        return MessageImportance.LOW

    # 规则 3：包含结构化信息（数字、列表、代码块等）→ HIGH
    import re
    has_numbers = bool(re.search(r'\d{2,}', content))  # 包含 2 位以上数字
    has_list = bool(re.search(r'^\s*[\d\-\*]\s*\.?\s+', content, re.MULTILINE))  # 列表格式
    has_code = '```' in content or '`' in content  # 代码片段
    if has_numbers or has_list or has_code:
        return MessageImportance.HIGH

    return MessageImportance.MEDIUM
```

**配套改动**：

- 在 `AIMessage` 和 `HumanMessage` 的 `additional_kwargs` 中存储 `importance` 字段
- `compress_conversation_history()` 压缩时，将 `HIGH` 重要性的消息原文作为"不可省略的关键信息"注入压缩 prompt，确保摘要中保留这些内容

**预期收益**：压缩后的摘要质量更高，关键信息（用户确认的事实、约束条件、结构化数据）不易丢失。

---

#### 策略 6：渐进式摘要（Incremental Summarization）（🟢 低优先级）--采用

**问题**：策略 1 的压缩方案在每次触发时需要将所有旧消息一次性压缩，当旧消息量很大时，压缩本身的 LLM 调用也可能超出上下文窗口。

**方案**：采用渐进式摘要，每隔固定轮数生成一次增量摘要，与已有摘要合并：

```python
async def incremental_summarize(
    llm: BaseChatModel,
    existing_summary: str,
    new_messages: List[AnyMessage],
) -> str:
    """将新消息与已有摘要合并，生成更新后的摘要"""
    new_text = "\n".join(
        f"{'用户' if isinstance(m, HumanMessage) else '助手'}: {m.content}"
        for m in new_messages
        if isinstance(m, (HumanMessage, AIMessage))
    )

    prompt = ChatPromptTemplate.from_template("""
    已有对话摘要：
    {existing_summary}

    新增对话内容：
    {new_messages}

    请将新增内容整合到已有摘要中，生成更新后的完整摘要。
    保留所有关键信息，去除冗余。
    """)
    chain = prompt | llm | StrOutputParser()
    return await chain.ainvoke({
        "existing_summary": existing_summary or "（无）",
        "new_messages": new_text,
    })
```

**实现方式**：

- 在 `State` 中新增 `conversation_summary: str` 字段
- 每隔 `incremental_summary_interval`（如 5 轮）触发一次增量摘要
- 摘要结果持久化到 checkpointer，下次对话直接读取

**摘要漂移（Summary Drift）防护**：

渐进式摘要的固有风险是"传话游戏"效应——每次增量摘要都基于"已有摘要 + 新消息"生成，经过多次迭代后，早期信息会被逐步稀释甚至丢失。

解决方案：引入**"锚点信息"机制**，将用户明确确认的关键事实（策略 5 中 `HIGH` 重要性的消息）单独存储为不可压缩的锚点列表，每次增量摘要时将锚点作为硬约束注入 prompt：

```python
async def incremental_summarize_with_anchors(
    llm: BaseChatModel,
    existing_summary: str,
    new_messages: List[AnyMessage],
    anchor_facts: List[str],
) -> str:
    """带锚点防护的增量摘要，防止关键信息在多次迭代中漂移丢失"""
    new_text = "\n".join(
        f"{'用户' if isinstance(m, HumanMessage) else '助手'}: {m.content}"
        for m in new_messages
        if isinstance(m, (HumanMessage, AIMessage))
    )

    anchors_text = "\n".join(f"- {fact}" for fact in anchor_facts) if anchor_facts else "（无）"

    prompt = ChatPromptTemplate.from_template("""
    已有对话摘要：
    {existing_summary}

    新增对话内容：
    {new_messages}

    ⚠️ 以下关键事实必须在摘要中完整保留，不可省略或改写：
    {anchor_facts}

    请将新增内容整合到已有摘要中，生成更新后的完整摘要。
    保留所有关键信息，去除冗余。确保上述关键事实在摘要中原样保留。
    """)
    chain = prompt | llm | StrOutputParser()
    return await chain.ainvoke({
        "existing_summary": existing_summary or "（无）",
        "new_messages": new_text,
        "anchor_facts": anchors_text,
    })
```

**预期收益**：避免一次性压缩大量消息导致的 token 溢出，摘要质量更稳定。锚点机制确保关键事实在多次迭代中不被稀释。

---

### 跨策略协作与边界说明

上述 6 个策略并非独立运行，部分策略之间存在功能重叠和执行依赖，需要明确边界以避免冲突：

#### 策略 1 与策略 2 的执行顺序和去重

策略 1（跨轮对话历史压缩）在入口节点压缩 `messages` 列表，策略 2（对话上下文窗口自适应）从 `messages` 中提取 `conversation_context`。两者存在功能重叠：

| 场景 | 策略 1 | 策略 2 | 潜在问题 |
|------|:------:|:------:|----------|
| 长对话（20 轮） | 压缩旧消息为 SystemMessage 摘要 | 从压缩后的 messages 提取上下文 | 策略 2 可能对策略 1 的摘要再做一次摘要，导致信息二次损失 |
| 短对话（5 轮）但单条消息很长 | 可能触发 token 超限压缩 | 从压缩后的 messages 提取上下文 | 同上 |
| 短对话且消息短 | 不触发 | 直接提取原始消息 | 无冲突 |

**执行顺序**：策略 1 先执行（入口节点最前面），策略 2 后执行（基于策略 1 处理后的 messages 提取上下文）。

**去重规则**：策略 2 的 `get_conversation_context_adaptive()` 在处理第二层（超出预算的早期消息）时，如果检测到 `SystemMessage` 类型且内容包含 `[对话历史摘要]` 标记，则直接复用该摘要作为前缀，不再二次压缩。此逻辑已在策略 2 的代码中实现。

#### 策略 3 与策略 4 的联动

策略 4（主题切换检测）在检测到主题切换时，将旧主题上下文通过 `save_conversation_memory(memory_type="topic_context")` 保存到 `MemoryManager`。策略 3（对话记忆检索）在入口节点通过 `get_recent_conversation_memories()` 或 `search_related_memories()` 从 `MemoryManager` 检索相关记忆。两者联动可实现：

- 用户切换到新主题 → 策略 4 保存旧主题上下文到 PostgreSQL（`save_conversation_memory`）
- 用户回到旧主题 → 策略 3 从 PostgreSQL 检索到旧主题的上下文摘要（`get_recent_conversation_memories` 或 `search_related_memories`），注入到 `conversation_context`

#### 策略 5 与策略 1、策略 6 的联动

策略 5（分级保留）为消息标记重要性等级。策略 1 和策略 6 在压缩时可利用这些标记：

- 策略 1：将 `HIGH` 重要性的消息原文作为硬约束注入压缩 prompt
- 策略 6：将 `HIGH` 重要性的消息提取为锚点信息，防止增量摘要漂移

**建议实施顺序**：策略 5 应在策略 1 和策略 6 之前实施，为后两者提供重要性标记基础。

---

#### 策略 7：工作记忆存储优化（🟡 中优先级）

**问题**：当前各业务节点（`route_node.py`、`retrieval_node.py`、`generate_node.py`）中散落了 6 处直接调用 `store.aput` 的工作记忆存储，存在以下问题：

1. **只写不读**：所有 `store.aput` 写入的数据没有被任何后续节点读取，实际作用等同于调试日志
2. **绕过 MemoryManager**：直接操作底层 Store，与 `MemoryManager` 的统一管理职责冲突
3. **namespace 混用**：业务节点使用 `(user_id, thread_id)` namespace，`MemoryManager` 使用 `("memory",)` namespace，两套体系并存
4. **无容量限制**：长对话场景下 Store 中的数据无限增长，无 TTL 清理和淘汰机制

**当前散落的 `store.aput` 调用**：

| 位置 | key 前缀 | 存储内容 | 后续是否被读取 |
|------|----------|----------|:--------------:|
| `route_node.py` | `need_retrieval_` | `{query, result}` | ❌ |
| `route_node.py` | `query_enhancer_` | `{enhanced_queries, enhancer_config}` | ❌ |
| `retrieval_node.py` | `final_retrieval_` | `{unique_docs_count, docs}` | ❌ |
| `generate_node.py` | `final_response_` | `{question, response, context_used, reasoning_steps}` | ❌ |
| `generate_node.py` | `sub_answer_` | `{question, answer}` | ❌ |
| `generate_node.py` | `synthesize_response_` | `{question, response}` | ❌ |

**方案**：

**1. 分类处理：有检索价值的存入 MemoryManager，其余降级为日志** --采用

| 存储项 | 处理方式 | 理由 |
|--------|----------|------|
| `final_response_` | ✅ 通过 `MemoryManager.save_conversation_memory(type="qa_pair")` 保存 | 最终问答对，策略 3 的对话记忆检索依赖 |
| `synthesize_response_` | ❌ 移除（与 `final_response_` 内容重叠） | 多跳推理的合成答案已包含在 `final_response_` 中 |
| `sub_answer_` | ❌ 降级为 `monitor_task_status` 日志 | 子问题中间答案已包含在 `final_response_` 的 `reasoning_steps` 中 |
| `need_retrieval_` | ❌ 降级为日志 | 仅布尔值，无检索价值 |
| `query_enhancer_` | ❌ 降级为日志 | 查询增强中间结果，仅调试用 |
| `final_retrieval_` | ❌ 降级为日志 | 检索结果快照数据量大，不会被后续对话引用 |

**2. 统一收敛到 MemoryManager** --采用

将有价值的存储统一通过 `MemoryManager` 管理，消除各节点直接操作 Store 的散落调用：

```python
# generate_node.py 中的 __final 节点
# 替换原来的 store.aput 调用
if memory_manager and user_id:
    await memory_manager.save_conversation_memory(
        user_id=user_id,
        thread_id=thread_id,
        memory_type="qa_pair",
        content={
            "question": query,
            "answer": answer,
        },
    )
```

**3. TTL 过期清理**

当前 `MemoryManager` 的 `ttl` 字段仅存储在 value 中，未实现真正的过期清理。由于 PostgreSQL Store 不支持原生 TTL，在读取时加入过期检查：

```python
async def _is_expired(self, value: dict) -> bool:
    """检查记忆是否已过期"""
    timestamp = value.get("timestamp", 0)
    ttl = value.get("ttl", 0)
    if ttl <= 0:
        return False
    return time.time() - timestamp > ttl
```

在 `get_recent_conversation_memories` 和 `search_related_memories` 中过滤已过期的记忆。

**4. 容量限制与淘汰策略**

每个 namespace 下最多保留 N 条记忆（如 50 条），写入时检查数量，超出则淘汰最旧的记录：

```python
async def _enforce_capacity(self, namespace: tuple, max_items: int = 50):
    """容量限制：超出时淘汰最旧的记录"""
    items = await self.store.asearch(namespace, limit=max_items + 10)
    if len(items) <= max_items:
        return
    # 按时间戳排序，删除最旧的超出部分
    sorted_items = sorted(items, key=lambda x: x.value.get("timestamp", 0))
    for item in sorted_items[:len(items) - max_items]:
        await self.store.adelete(namespace, key=item.key)
```

**预期收益**：
- 消除 6 处散落的 `store.aput`，统一通过 `MemoryManager` 管理
- 减少约 80% 的无效存储写入（仅保留有检索价值的问答对）
- TTL 和容量限制防止长期运行后数据无限增长
- 为策略 3（对话记忆检索）提供高质量的数据源

---

### 三、实施优先级总结

| 优先级 | 优化项 | 涉及文件 | 实施难度 | 预期收益 |
|:------:|--------|----------|:--------:|:--------:|
| 🔴 高 | 跨轮对话历史压缩 | `generate.py`、`route_node.py`、`config.py` | 中 | 高 |
| 🔴 高 | 对话上下文窗口自适应 | `message_util.py` | 低 | 中 |
| 🟡 中 | MemoryManager 对话记忆检索 | `memory_manager.py`、`route_node.py` | 中 | 中 |
| 🟡 中 | 对话主题切换检测 | `route_node.py`、`generate.py` | 中 | 中 |
| 🟡 中 | 工作记忆存储优化 | `route_node.py`、`retrieval_node.py`、`generate_node.py`、`memory_manager.py` | 低 | 中 |
| 🟢 低 | 对话历史分级保留 | `generate.py`、`route_node.py` | 中 | 低 |
| 🟢 低 | 渐进式摘要 | `generate.py`、`graph.py`（State） | 高 | 中 |

### 四、评估验证

每项优化实施后，建议通过以下方式验证效果：

- **多轮对话连贯性**：构造 10-20 轮的长对话测试用例，验证压缩后系统仍能正确引用早期对话中的关键信息
- **Token 消耗对比**：对比压缩前后每轮请求的 token 消耗量，预期压缩后降低 30%-50%
- **主题切换准确率**：构造包含主题切换的对话样本，验证检测准确率 > 90%
- **Faithfulness（RAGAS）**：压缩不应导致答案偏离检索内容，需确认指标无下降
- **Answer Relevancy（RAGAS）**：自适应上下文窗口应提升答案与问题的相关性
- **用户体验指标**：通过人工评估验证长对话场景下的回答质量和响应速度

建议按优先级分阶段实施，每阶段在同一批评估样本上对比优化前后的指标变化，确保改进有据可依。
