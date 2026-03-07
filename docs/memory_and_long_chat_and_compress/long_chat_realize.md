# 多轮对话实现方案

本文档记录 HybridRAG 系统多轮对话的完整实现机制，包括状态持久化、上下文传递、会话管理和长对话增强。

---

## 一、整体架构

```
前端 (frontend/js/chat.js)
  │
  ├─ POST /api/chat/stream  ──→  server.py（会话管理 + SSE 流式）
  │                                   │
  │                                   ├─ conversations 字典（内存缓存消息历史）
  │                                   │
  │                                   └─ Graph.start_stream()
  │                                         │
  │                                         ├─ LangGraph StateGraph（状态机）
  │                                         │     └─ State(MessagesState) ← 自动维护 messages 列表
  │                                         │
  │                                         ├─ AsyncPostgresSaver（checkpointer）
  │                                         │     └─ 按 thread_id 持久化完整 State
  │                                         │
  │                                         └─ AsyncPostgresStore（store）
  │                                               └─ MemoryManager 存储对话记忆
```

多轮对话的核心依赖两个层面：
1. **LangGraph Checkpointer**：按 `thread_id` 自动持久化 `State`（包括 `messages` 列表），同一 `thread_id` 的后续请求自动恢复历史消息
2. **Server 层会话管理**：`conversations` 字典维护前端展示用的消息历史

---

## 二、关键组件

### 2.1 State 定义（`src/graph.py`）

```python
class State(MessagesState):
    original_query: str
    task_characteristics: Optional[TaskCharacteristics]
    need_retrieval: bool
    router_index: Optional[Dict[str, List[str]]]
    search_content: Optional[str]
    retrieved_documents: List[str]
    retrieval_scores: List[float]
    run_count: int
    grade_retry_count: int
    answer: str
    sub_questions: List[str]
    current_sub_question: Optional[str]
    reasoning_context: str
    reasoning_steps: List[dict]
    conversation_summary: str  # 渐进式摘要
```

- 继承 `MessagesState`，自动获得 `messages: List[AnyMessage]` 字段
- `messages` 使用 `add_messages` reducer（追加语义），每轮对话的 HumanMessage 和 AIMessage 自动追加
- `conversation_summary` 用于渐进式摘要的持久化存储

### 2.2 Checkpointer（`src/graph.py` → `_compile_graph`）

```python
async def _compile_graph(self, workflow):
    conn_ctx = AsyncPostgresSaver.from_conn_string(POSTGRESQL_URL)
    store_ctx = AsyncPostgresStore.from_conn_string(POSTGRESQL_URL)
    checkpointer = await conn_ctx.__aenter__()
    store = await store_ctx.__aenter__()
    await checkpointer.setup()
    graph = workflow.compile(checkpointer=checkpointer, store=store)
    return graph, (conn_ctx, store_ctx)
```

- **AsyncPostgresSaver**：LangGraph 的 PostgreSQL checkpointer，按 `thread_id` 存储完整的 State 快照
- **AsyncPostgresStore**：LangGraph 的 KV 存储，供 MemoryManager 使用
- 同一 `thread_id` 的后续调用会自动从 PostgreSQL 恢复上一次的 State，包括完整的 `messages` 列表

### 2.3 Server 层会话管理（`server.py`）

```python
conversations: dict[str, dict] = {}
```

Server 层维护一个内存字典，用于前端展示：

| 操作 | API | 说明 |
|------|-----|------|
| 创建/复用会话 | `POST /api/chat/stream` | 按 `thread_id` 创建或复用会话，追加用户消息 |
| 列出会话 | `GET /api/conversations` | 返回所有会话列表 |
| 查看历史 | `GET /api/conversations/{thread_id}` | 返回单个会话的消息历史 |
| 删除会话 | `DELETE /api/conversations/{thread_id}` | 删除会话 |

**注意**：`conversations` 是内存缓存，服务重启后丢失。真正的对话状态持久化由 LangGraph Checkpointer 负责。

### 2.4 MemoryManager（`src/core/memory_manager.py`）

MemoryManager 基于 `AsyncPostgresStore` 提供结构化的记忆存储：

| 方法 | 用途 |
|------|------|
| `save_conversation_memory` | 存储对话记忆（如 QA 对） |
| `get_recent_conversation_memories` | 获取最近的对话记忆 |
| `save_user_preference` | 存储用户偏好 |
| `save_contextual_memory` | 存储上下文记忆 |
| `search_related_memories` | 基于关键词搜索相关记忆 |

在 `generate_node.py` 的 `__final` 节点中，最终问答对通过 MemoryManager 存储：

```python
await self.memory_manager.save_conversation_memory(
    user_id=user_id,
    thread_id=thread_id,
    memory_type="qa_pair",
    content={"question": question, "answer": answer},
)
```

---

## 三、多轮对话数据流

### 3.1 第一轮对话

```
用户发送: "什么是RAG？"
    │
    ├─ server.py: conversations["thread-1"] = {messages: [{role: "user", content: "什么是RAG？"}]}
    │
    ├─ Graph.start_stream(
    │     {"messages": [HumanMessage("什么是RAG？")]},
    │     {"configurable": {"thread_id": "thread-1", "user_id": "user1"}}
    │   )
    │
    ├─ Checkpointer: thread-1 无历史 → State.messages = [HumanMessage("什么是RAG？")]
    │
    ├─ __retrieve_or_respond → 检索/生成 → __final
    │     └─ 返回 AIMessage("RAG是检索增强生成...")
    │
    ├─ Checkpointer: 保存 State.messages = [HumanMessage("什么是RAG？"), AIMessage("RAG是...")]
    │
    └─ server.py: conversations["thread-1"].messages.append({role: "assistant", content: "RAG是..."})
```

### 3.2 第二轮对话（自动携带历史）

```
用户发送: "它和传统搜索有什么区别？"
    │
    ├─ server.py: conversations["thread-1"].messages.append({role: "user", content: "它和传统搜索..."})
    │
    ├─ Graph.start_stream(
    │     {"messages": [HumanMessage("它和传统搜索有什么区别？")]},
    │     {"configurable": {"thread_id": "thread-1", ...}}
    │   )
    │
    ├─ Checkpointer: thread-1 有历史 → 恢复上次 State
    │   State.messages = [
    │     HumanMessage("什么是RAG？"),      ← 自动恢复
    │     AIMessage("RAG是..."),             ← 自动恢复
    │     HumanMessage("它和传统搜索..."),   ← 本次追加
    │   ]
    │
    ├─ __retrieve_or_respond:
    │   ├─ 策略 1: 检查是否需要压缩（轮数/token 是否超限）
    │   ├─ 策略 6: 检查是否触发渐进式摘要
    │   ├─ 策略 2: get_conversation_context_adaptive 提取上下文
    │   │   └─ 上下文包含: "用户: 什么是RAG？\n助手: RAG是...\n用户: 它和传统搜索..."
    │   └─ 将上下文注入 prompt，LLM 理解"它"指代 RAG
    │
    └─ 生成答案: "RAG与传统搜索的区别在于..."
```

### 3.3 长对话场景（第 N 轮，触发压缩）

```
第 11 轮对话（超过 max_conversation_turns=10）
    │
    ├─ Checkpointer 恢复: State.messages = [20 条消息]
    │
    ├─ __retrieve_or_respond:
    │   ├─ 策略 1: turns_exceeded=True → 触发压缩
    │   │   ├─ 保留最近 3 轮（6 条消息）
    │   │   ├─ 前 14 条消息 → LLM 压缩为 SystemMessage("[对话历史摘要]\n...")
    │   │   └─ 通过 RemoveMessage 机制替换 State.messages
    │   │
    │   ├─ 策略 6: human_count=11, 11%5≠0 → 不触发
    │   │
    │   └─ 策略 2: get_conversation_context_adaptive
    │       └─ 检测到 SystemMessage("[对话历史摘要]") → 直接复用，不二次压缩
    │
    └─ 压缩后 State.messages = [SystemMessage(摘要), 最近 6 条原始消息, 本轮新消息]
```

---

## 四、上下文提取机制

### 4.1 基础版：`get_conversation_context`

```python
def get_conversation_context(messages, num_messages=3) -> str:
```

- 固定取最近 `num_messages` 条消息，拼接为 `"用户: xxx\n助手: yyy"` 格式
- 用于 `route_node.py` 的查询增强等非核心场景

### 4.2 自适应版：`get_conversation_context_adaptive`

```python
async def get_conversation_context_adaptive(messages, llm, max_context_tokens=2000) -> str:
```

两层策略：

| 层级 | 处理对象 | 方式 |
|------|----------|------|
| 第一层 | 最近的消息（token 预算内） | 保留原文 |
| 第二层 | 更早的消息（预算外） | LLM 摘要或复用策略 1 的摘要 |

用于 `route_node.py` 的检索判断和 `generate_node.py` 的答案生成。

---

## 五、使用方式

### 5.1 API 调用

```bash
# 第一轮
curl -X POST http://localhost:8000/api/chat/stream \
  -H "Content-Type: application/json" \
  -d '{"message": "什么是RAG？", "thread_id": "session-123"}'

# 第二轮（同一 thread_id，自动携带历史）
curl -X POST http://localhost:8000/api/chat/stream \
  -H "Content-Type: application/json" \
  -d '{"message": "它和传统搜索有什么区别？", "thread_id": "session-123"}'

# 新会话（不同 thread_id）
curl -X POST http://localhost:8000/api/chat/stream \
  -H "Content-Type: application/json" \
  -d '{"message": "你好", "thread_id": "session-456"}'
```

### 5.2 编程调用

```python
from langchain_core.messages import HumanMessage
from src.graph import Graph

graph = Graph()
config = {"configurable": {"thread_id": "session-123", "user_id": "user1"}}

# 第一轮
async for event in graph.start_stream(
    {"messages": [HumanMessage(content="什么是RAG？")]}, config
):
    print(event)

# 第二轮（自动携带历史）
async for event in graph.start_stream(
    {"messages": [HumanMessage(content="它和传统搜索有什么区别？")]}, config
):
    print(event)
```

---

## 六、配置项

| 配置项 | 默认值 | 说明 |
|--------|--------|------|
| `enable_conversation_compress` | `True` | 是否启用跨轮对话历史压缩 |
| `max_conversation_turns` | `10` | 最大对话轮数（超过触发压缩） |
| `max_conversation_tokens` | `4000` | 最大对话 token 数（超过触发压缩） |
| `keep_recent_turns` | `3` | 压缩时保留最近的轮数 |
| `max_compress_tokens` | `6000` | 单次压缩的最大 token 数（超过分批） |
| `incremental_summary_interval` | `5` | 渐进式摘要触发间隔（每 N 轮） |
| `max_context_tokens` | `2000` | 自适应上下文窗口的 token 预算 |

---

## 七、持久化层级

| 层级 | 存储位置 | 生命周期 | 内容 |
|------|----------|----------|------|
| State.messages | PostgreSQL（Checkpointer） | 永久（按 thread_id） | 完整消息历史（压缩后） |
| State.conversation_summary | PostgreSQL（Checkpointer） | 永久（按 thread_id） | 渐进式摘要 |
| MemoryManager | PostgreSQL（Store） | 按 TTL 过期 | QA 对、用户偏好、上下文记忆 |
| conversations 字典 | 内存 | 服务重启丢失 | 前端展示用的消息历史 |

---

## 八、与压缩策略的协作关系

详见 [compress_realize.md](./compress_realize.md)。

多轮对话是基础设施，压缩策略是长对话增强。三个策略在 `__retrieve_or_respond` 入口处按顺序执行：

```
策略 1（压缩） → 策略 6（渐进式摘要） → 策略 2（自适应上下文提取）
```

- **策略 1** 减少 State.messages 的体积，防止 Checkpointer 存储膨胀和 LLM token 溢出
- **策略 6** 维护跨轮的累积摘要，即使消息被压缩也能保留全局语义
- **策略 2** 在每次生成答案时，智能提取最相关的上下文注入 prompt
