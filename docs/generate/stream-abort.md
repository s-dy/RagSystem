# 流式对话中断机制

本文档描述 HybridRAG 系统中，前端用户中断流式对话时，前后端的协作机制。

---

## 整体流程

```
用户点击"停止"按钮
    ↓
前端 AbortController.abort()
    ↓
浏览器断开 SSE 连接（TCP 关闭）
    ↓
后端 StreamingResponse yield 失败，event_generator 退出
    ↓
finally 块执行：cancel stream_task → 取消 LangGraph 推理
    ↓
Graph.start_stream 捕获 CancelledError，清理数据库连接
```

---

## 前端中断

**相关文件**: `frontend/js/chat.js`

### 创建中断控制器

在发起流式请求时，创建 `AbortController` 并将其 `signal` 绑定到 `fetch` 请求：

```javascript
async function streamChat(message) {
    AppState.isStreaming = true;
    AppState.abortController = new AbortController();

    const response = await fetch(API.chatStream, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ message, thread_id, user_id, enable_web_search }),
        signal: AppState.abortController.signal,  // 绑定中断信号
    });
    // ... 读取 SSE 流
}
```

### 触发中断

用户点击停止按钮时调用 `stopStreaming()`：

```javascript
function stopStreaming() {
    if (AppState.abortController) {
        AppState.abortController.abort();
    }
}
```

### 异常处理

`streamChat` 的 `catch` 块区分用户主动中断和真实错误：

```javascript
} catch (error) {
    if (error.name === 'AbortError') {
        // 用户主动中断 → 保留已渲染的内容，不报错
    } else {
        // 真实网络/服务端错误 → 显示错误提示
    }
} finally {
    AppState.isStreaming = false;
    AppState.abortController = null;
    updateSendButtonState();  // 切换回发送按钮
}
```

---

## 后端中断

### API 层（server.py）

**相关文件**: `server.py` — `chat_stream` 接口

后端使用 FastAPI `StreamingResponse` 返回 SSE 流。核心结构：

```python
stream_task = asyncio.create_task(stream_events())      # LangGraph 推理任务
heartbeat_task = asyncio.create_task(send_heartbeats())  # 心跳保活任务

try:
    while True:
        event = await queue.get()
        if event is None:
            break
        yield f"data: {json.dumps(event)}\n\n"
finally:
    heartbeat_event.set()
    heartbeat_task.cancel()
    stream_task.cancel()          # 取消正在执行的 LangGraph 推理
    for task in (heartbeat_task, stream_task):
        try:
            await task
        except asyncio.CancelledError:
            pass
```

**关键点**：

1. 前端 `abort()` 后，浏览器断开 TCP 连接
2. FastAPI/Uvicorn 在下次 `yield` 时检测到连接已关闭，`event_generator` 退出
3. `finally` 块执行 `stream_task.cancel()`，向 LangGraph 推理任务发送 `CancelledError`
4. 等待 `stream_task` 完成，确保资源被正确清理

### Graph 层（src/graph.py）

**相关文件**: `src/graph.py` — `Graph.start_stream` 方法

`start_stream` 是一个 async generator，通过 `graph.astream_events()` 迭代 LangGraph 事件：

```python
async def start_stream(self, input_data, config):
    workflow = await self._init_graph()
    graph, ctx = await self._compile_graph(workflow)  # ctx 持有 PostgreSQL 连接

    try:
        async for event in graph.astream_events(input_data, config, version="v2"):
            # 检查任务是否已被取消
            current_task = asyncio.current_task()
            if current_task and current_task.cancelled():
                break

            # ... 处理事件并 yield

        yield {"type": "done"}

    except asyncio.CancelledError:
        # 记录取消日志，然后重新抛出
        raise
    finally:
        # 无论正常结束还是被取消，都清理数据库连接
        if ctx:
            conn_ctx, store_ctx = ctx
            await store_ctx.__aexit__(None, None, None)
            await conn_ctx.__aexit__(None, None, None)
```

**关键点**：

1. 每次迭代事件时检查 `current_task().cancelled()`，实现快速响应取消
2. `except asyncio.CancelledError` 捕获取消异常，记录日志后重新抛出
3. `finally` 块确保 PostgreSQL checkpointer 和 store 连接被正确关闭，避免连接泄漏

---

## 资源清理保障

| 资源 | 清理位置 | 清理方式 |
|------|---------|---------|
| **前端 fetch 连接** | `chat.js` finally 块 | `abortController = null` |
| **后端心跳任务** | `server.py` finally 块 | `heartbeat_task.cancel()` |
| **后端推理任务** | `server.py` finally 块 | `stream_task.cancel()` |
| **PostgreSQL checkpointer** | `graph.py` finally 块 | `conn_ctx.__aexit__()` |
| **PostgreSQL store** | `graph.py` finally 块 | `store_ctx.__aexit__()` |

---

## 时序图

```
前端 (chat.js)          后端 (server.py)          Graph (graph.py)
    │                        │                        │
    │── POST /chat/stream ──→│                        │
    │                        │── create stream_task ──→│
    │                        │── create heartbeat ──→  │
    │                        │                        │
    │←── SSE: token ─────────│←── yield token ────────│
    │←── SSE: token ─────────│←── yield token ────────│
    │                        │                        │
    │  [用户点击停止]          │                        │
    │  abort()               │                        │
    │── 断开连接 ────────────→│                        │
    │                        │  [yield 失败，进入 finally]
    │                        │── stream_task.cancel() →│
    │                        │                        │── [CancelledError]
    │                        │                        │── 清理 DB 连接
    │                        │← await stream_task ────│
    │                        │  [清理完成]              │
```

---

## 注意事项

### 已取消任务的部分结果

前端中断后，已经渲染到页面上的内容会被保留。如果 `fullContent` 为空（还没收到任何 token），会显示"已停止生成"提示。

### LangGraph 节点级取消

`asyncio.Task.cancel()` 会在 `astream_events` 的下一个 `await` 点抛出 `CancelledError`。如果某个节点正在执行一个不可中断的同步操作（如 `asyncio.to_thread` 中的 reranker 推理），该操作会执行完毕后才响应取消。这是 Python asyncio 的固有限制。

### 会话记录

如果推理被中途取消，已生成的部分回答**不会**被记录到 `conversations` 中（因为 `done` 事件未被发送）。下次用户发送消息时，会话上下文中不包含被中断的回答。
