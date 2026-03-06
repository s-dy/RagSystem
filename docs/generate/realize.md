## 生成与可解释性 — 实现方案

> 本文档详细记录 [fit.md](./fit.md) 中各优化策略的具体实现方案，包括架构设计、核心代码逻辑、事件协议和前端渲染机制。

---

### 一、整体架构

#### 1.1 数据流全链路

```
用户输入
  │
  ▼
┌─────────────────────────────────────────────────────────────────┐
│  server.py — POST /api/chat/stream                              │
│  ┌───────────────────────────────────────────────────────────┐  │
│  │  event_generator()                                        │  │
│  │  ┌─────────────┐    ┌──────────────┐                      │  │
│  │  │ stream_task  │    │ heartbeat    │                      │  │
│  │  │ (graph事件)  │───▶│ (15s心跳)    │──▶ asyncio.Queue     │  │
│  │  └─────────────┘    └──────────────┘         │            │  │
│  │                                               ▼            │  │
│  │                                    SSE StreamingResponse   │  │
│  └───────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
  │
  ▼ SSE (text/event-stream)
┌─────────────────────────────────────────────────────────────────┐
│  frontend/js/chat.js — streamChat()                             │
│  ┌───────────────────────────────────────────────────────────┐  │
│  │  ReadableStream reader                                    │  │
│  │  ┌──────────────────┐  ┌──────────────────┐               │  │
│  │  │  .reasoning-area │  │  .answer-area    │               │  │
│  │  │  (推理过程折叠块) │  │  (答案 Markdown) │               │  │
│  │  └──────────────────┘  └──────────────────┘               │  │
│  └───────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
```

#### 1.2 Graph 节点执行顺序

```
START
  │
  ▼
retrieve_or_respond ──(不需要检索)──▶ END（直接回复，yield final_answer）
  │
  │ (需要检索)
  ▼
prepare_next_step ──────────────────▶ yield decomposition（子问题分解）
  │
  ▼
enhance_and_route_current
  │
  ▼
fusion_retrieve
  │
  ├──(评分不合格)──▶ enhance_and_route_current（重试）
  │
  ▼ (评分合格)
generate_current_answer ────────────▶ yield sub_answer（子问题中间答案）
  │                                   yield final_answer（最终答案，is_final=True）
  ├──(还有子问题)──▶ prepare_next_step（循环）
  │
  ▼ (子问题处理完毕)
synthesize ─────────────────────────▶ yield token（流式合成答案，多跳时）
  │                                   yield final_answer（合成后的最终答案）
  ▼
final ──────────────────────────────▶ yield done
  │
  ▼
END
```

#### 1.3 SSE 事件协议

后端通过 SSE 向前端推送以下事件类型，每条事件格式为 `data: {JSON}\n\n`：

| 事件类型 | 触发节点 | 数据结构 | 说明 |
|----------|----------|----------|------|
| `token` | `synthesize` / `final` | `{"type":"token","content":"...","node":"synthesize"}` | LLM 逐 token 流式输出，仅允许合成答案和最终输出节点 |
| `decomposition` | `prepare_next_step` | `{"type":"decomposition","sub_questions":["q1","q2",...]}` | 多跳问题的子问题分解结果 |
| `sub_answer` | `generate_current_answer` | `{"type":"sub_answer","sub_question":"...","answer":"..."}` | 子问题的中间答案（非最终答案） |
| `final_answer` | `retrieve_or_respond` / `generate_current_answer` / `synthesize` | `{"type":"final_answer","answer":"..."}` | 最终完整答案 |
| `heartbeat` | 心跳任务 | `{"type":"heartbeat"}` | 每 15 秒发送，防止 SSE 连接超时断开 |
| `error` | 异常捕获 | `{"type":"error","content":"错误信息"}` | 后端处理异常 |
| `done` | 流结束 | `{"type":"done"}` | 标记整个流式响应结束 |

**token 白名单过滤规则**：

`on_chat_model_stream` 事件中，只有 `langgraph_node` 为 `synthesize` 或 `final` 的 token 才会被 yield。其他节点（`retrieve_or_respond`、`prepare_next_step`、`enhance_and_route_current`、`generate_current_answer`）的 LLM token 全部被过滤，原因如下：

- `retrieve_or_respond`：LLM 输出是结构化判断（`NEED_RETRIEVAL[...]` 或直接回复），不应展示中间判断过程
- `prepare_next_step`：LLM 输出是子问题分解的结构化数据
- `enhance_and_route_current`：LLM 输出是查询增强的结构化数据
- `generate_current_answer`：多跳时该节点被多次调用（每个子问题一次），如果放行 token，所有子问题的答案 token 会混在一起拼接到前端的 `fullContent` 中

---

### 二、流式输出实现

#### 2.1 后端：`Graph.start_stream()`

**文件**：`src/graph.py`，第 125 行起

核心方法使用 LangGraph 的 `astream_events(version="v2")` API，该 API 会将 graph 执行过程中的所有事件（LLM token、节点开始/结束等）以异步迭代器的形式返回。

```python
async def start_stream(self, input_data: dict, config: dict):
    workflow = await self._init_graph()
    await self._ensure_initialized()
    graph, ctx = await self._compile_graph(workflow)

    try:
        async for event in graph.astream_events(input_data, config, version="v2"):
            event_kind = event.get("event", "")

            # 1. LLM 逐 token 流式输出（白名单过滤）
            if event_kind == "on_chat_model_stream":
                node_name = event.get("metadata", {}).get("langgraph_node", "")
                if node_name not in ("synthesize", "final"):
                    continue
                chunk = event.get("data", {}).get("chunk")
                if chunk and hasattr(chunk, "content") and chunk.content:
                    yield {"type": "token", "content": chunk.content, "node": node_name}

            # 2. 节点执行完成事件（on_chain_end）
            elif event_kind == "on_chain_end" and event.get("name") in (...):
                # 处理 retrieve_or_respond、prepare_next_step、
                # generate_current_answer、synthesize、final 节点的输出
                ...

        yield {"type": "done"}
    finally:
        # 清理 PostgreSQL 连接上下文
        if ctx:
            conn_ctx, store_ctx = ctx
            await store_ctx.__aexit__(None, None, None)
            await conn_ctx.__aexit__(None, None, None)
```

**`on_chain_end` 事件处理逻辑**：

| 节点名 | 条件 | yield 事件 |
|--------|------|-----------|
| `retrieve_or_respond` | `need_retrieval=False` 且有 `answer` | `final_answer`（直接回复） |
| `prepare_next_step` | 有 `reasoning_steps` | `decomposition`（子问题列表） |
| `generate_current_answer` | 有 `reasoning_steps` 且 `is_final=False` | `sub_answer`（子问题中间答案） |
| `generate_current_answer` | 有 `answer` 且 `is_final=True` | `final_answer`（单跳最终答案） |
| `synthesize` | 有 `answer` | `final_answer`（多跳合成答案） |

**`retrieve_or_respond` 直接回复的特殊处理**：

当 LLM 判断不需要检索时（如问候语、简单聊天），`_retrieve_or_respond` 返回：

```python
return {
    'need_retrieval': False,
    'answer': response,                         # LLM 的直接回复内容
    'messages': [AIMessage(content=response)],   # 写入对话历史
}
```

graph 直接走 `END`，不经过任何答案生成节点。`start_stream` 在 `on_chain_end` 中检测到该节点的输出后，yield `final_answer` 事件。由于 `astream_events` v2 中 `output` 可能是 `{node_name: {...}}` 或直接 `{...}`，代码做了兼容处理：

```python
node_output = output.get("retrieve_or_respond", output)
if not node_output.get("need_retrieval") and node_output.get("answer"):
    yield {"type": "final_answer", "answer": node_output["answer"]}
```

#### 2.2 后端：`server.py` SSE 接口

**文件**：`server.py`，第 87 行起

**接口**：`POST /api/chat/stream`

**请求体**：

```json
{
    "message": "用户消息",
    "thread_id": "会话ID（可选，默认自动生成）",
    "user_id": "用户ID（可选，默认 default）",
    "enable_web_search": false
}
```

**心跳机制**：

使用 `asyncio.Queue` 合并主事件流和心跳任务，避免后端长时间处理（检索、重排序等）期间 SSE 连接被代理或浏览器超时关闭：

```python
async def event_generator():
    queue = asyncio.Queue()

    async def stream_events():
        """主事件流：将 graph.start_stream 的事件放入队列"""
        try:
            async for event in graph.start_stream(input_data, config):
                await queue.put(event)
        except Exception as exc:
            await queue.put({"type": "error", "content": str(exc)})
        finally:
            await queue.put(None)  # 结束标记

    async def send_heartbeats():
        """心跳任务：每 15 秒放入一个 heartbeat 事件"""
        while not heartbeat_event.is_set():
            await asyncio.sleep(15)
            if not heartbeat_event.is_set():
                await queue.put({"type": "heartbeat"})

    stream_task = asyncio.create_task(stream_events())
    heartbeat_task = asyncio.create_task(send_heartbeats())

    try:
        while True:
            event = await queue.get()
            if event is None:
                break
            yield f"data: {json.dumps(event, ensure_ascii=False)}\n\n"
    finally:
        heartbeat_event.set()
        heartbeat_task.cancel()
```

**响应头**：

```python
StreamingResponse(
    event_generator(),
    media_type="text/event-stream",
    headers={
        "Cache-Control": "no-cache",
        "Connection": "keep-alive",
        "X-Accel-Buffering": "no",  # 禁用 Nginx 缓冲
    },
)
```

#### 2.3 前端：`chat.js` 流式渲染

**文件**：`frontend/js/chat.js`，`streamChat()` 函数

**消息体结构**：

```html
<div class="message-body">
    <div class="reasoning-area"></div>   <!-- 推理过程：子问题分解 + 推理步骤 -->
    <div class="answer-area">           <!-- 答案内容：流式 Markdown -->
        <div class="typing-indicator">...</div>
    </div>
</div>
```

两个区域互不干扰：token 渲染只更新 `.answer-area`，推理事件只更新 `.reasoning-area`。

**节流渲染**：

为避免每个 token 都触发完整 Markdown 重新解析（`marked.js` + `highlight.js`），使用 50ms 节流：

```javascript
function throttledRender() {
    if (renderTimer) return;
    pendingRender = true;
    renderTimer = setTimeout(() => {
        if (pendingRender) {
            answerArea.innerHTML = renderMarkdown(fullContent);
            scrollToBottom();
            pendingRender = false;
        }
        renderTimer = null;
    }, 50);
}
```

**事件处理逻辑**：

| 事件类型 | 处理方式 |
|----------|----------|
| `token` | 首次收到时清空打字指示器；拼接到 `fullContent`；调用 `throttledRender()` |
| `decomposition` | 调用 `renderDecomposition()` 生成折叠块 HTML，插入 `reasoningArea`；绑定折叠事件 |
| `sub_answer` | push 到 `reasoningSteps` 数组；调用 `updateReasoningDisplay()` 增量追加 |
| `final_answer` | `fullContent = event.answer`（替换，清除可能混入的子问题 token）；立即渲染到 `answerArea` |
| `heartbeat` | 静默忽略 |
| `error` | 渲染错误提示到 `answerArea` |
| `done` | 无操作（流结束由 `reader.read()` 的 `done=true` 触发） |

**非流式降级**：

当流式接口返回非 200 状态码或发生网络错误时，自动降级到 `POST /api/chat`（非流式接口）：

```javascript
if (!response.ok) {
    bodyElement.innerHTML = '';
    await fallbackChatWithElement(message, bodyElement);
    return;
}
```

`catch` 块中也做了降级处理，网络错误时调用 `fallbackChatWithElement()`，复用已有的消息元素。

**最终渲染保障**：

`finally` 块中确保最后一次渲染完成（清除节流定时器，执行最终渲染）：

```javascript
finally {
    if (renderTimer) {
        clearTimeout(renderTimer);
        renderTimer = null;
    }
    if (fullContent && hasReceivedToken) {
        answerArea.innerHTML = renderMarkdown(fullContent);
    }
    AppState.isStreaming = false;
    AppState.abortController = null;
    updateSendButtonState();
    scrollToBottom();
}
```

---

### 三、推理过程透明化 + 子问题分解可视化

#### 3.1 State 字段

**文件**：`src/graph.py`，第 37 行起

在 `State(MessagesState)` 中新增 `reasoning_steps` 字段：

```python
class State(MessagesState):
    ...
    reasoning_steps: Annotated[list, operator.add]  # 推理步骤记录
```

使用 `operator.add` 作为 reducer，每个节点返回的 `reasoning_steps` 会自动追加到已有列表中。

#### 3.2 后端事件生成

**子问题分解（`decomposition`）**：

`_prepare_next_step()` 节点在多跳问题分解完成后，将子问题列表写入 `reasoning_steps`：

```python
return {
    "sub_questions": sub_questions,
    "current_sub_question": sub_questions[0],
    "reasoning_steps": [{"type": "decomposition", "sub_questions": sub_questions}],
    ...
}
```

`start_stream` 在 `on_chain_end` 中提取并 yield：

```python
if node_name == "prepare_next_step" and output.get("reasoning_steps"):
    yield {
        "type": "decomposition",
        "sub_questions": output["reasoning_steps"][0].get("sub_questions", []),
    }
```

**子问题中间答案（`sub_answer`）**：

`_generate_current_answer()` 节点每完成一个子问题，将答案写入 `reasoning_steps`：

```python
sub_answer_step = {
    "type": "sub_answer",
    "sub_question": current_q,
    "answer": answer[:200],  # 截断避免 state 过大
    "is_final": is_final,
}
return {
    ...
    "reasoning_steps": existing_steps + [sub_answer_step],
}
```

`start_stream` 在 `on_chain_end` 中检测到非最终答案时 yield：

```python
if node_name == "generate_current_answer" and output.get("reasoning_steps"):
    latest_step = output["reasoning_steps"][-1]
    if not latest_step.get("is_final"):
        yield {
            "type": "sub_answer",
            "sub_question": latest_step.get("sub_question", ""),
            "answer": latest_step.get("answer", ""),
        }
```

#### 3.3 前端实时渲染

**子问题分解折叠块**：

`renderDecomposition()` 函数生成默认展开的有序列表：

```javascript
function renderDecomposition(subQuestions) {
    const questionsHtml = subQuestions.map((question, index) => {
        const questionText = typeof question === 'string'
            ? question
            : (question.sub_question || JSON.stringify(question));
        return `<li>${escapeHtml(questionText)}</li>`;
    }).join('');

    return `
        <div class="reasoning-block" style="margin-bottom: 12px;">
            <button class="reasoning-header expanded">
                <span class="arrow" style="transform: rotate(90deg);">▶</span>
                🔍 问题分解
            </button>
            <div class="reasoning-content expanded">
                <ol style="padding-left: 20px;">${questionsHtml}</ol>
            </div>
        </div>
    `;
}
```

**推理步骤增量追加**：

`updateReasoningDisplay()` 函数在收到每个 `sub_answer` 事件时被调用，只追加新的步骤，不重新渲染全部：

```javascript
function updateReasoningDisplay(reasoningArea, steps) {
    let reasoningBlock = reasoningArea.querySelector('.reasoning-steps-block');
    if (!reasoningBlock) {
        // 首次创建推理过程折叠块（默认展开）
        reasoningBlock = document.createElement('div');
        reasoningBlock.className = 'reasoning-block reasoning-steps-block';
        reasoningBlock.innerHTML = `
            <button class="reasoning-header expanded">
                <span class="arrow" style="transform: rotate(90deg);">▶</span>
                🧩 推理过程（<span class="step-count">0</span> 个子问题）
            </button>
            <div class="reasoning-content expanded"></div>
        `;
        // 绑定折叠事件...
        reasoningArea.appendChild(reasoningBlock);
    }

    // 更新步骤计数
    reasoningBlock.querySelector('.step-count').textContent = steps.length;

    // 只追加新的步骤（增量渲染）
    const contentEl = reasoningBlock.querySelector('.reasoning-content');
    const existingCount = contentEl.querySelectorAll('.reasoning-step').length;
    for (let i = existingCount; i < steps.length; i++) {
        const stepDiv = document.createElement('div');
        stepDiv.className = 'reasoning-step';
        stepDiv.innerHTML = `
            <div class="reasoning-step-question">
                📌 子问题 ${i + 1}: ${escapeHtml(steps[i].question)}
            </div>
            <div class="reasoning-step-answer">
                ${renderMarkdown(steps[i].answer)}
            </div>
        `;
        contentEl.appendChild(stepDiv);
    }
}
```

**防重复绑定**：

`bindReasoningToggle()` 使用 `data-bound` 属性防止折叠事件被重复绑定：

```javascript
function bindReasoningToggle(container) {
    container.querySelectorAll('.reasoning-header').forEach(header => {
        if (header.dataset.bound) return;
        header.dataset.bound = 'true';
        header.addEventListener('click', () => {
            header.classList.toggle('expanded');
            const content = header.nextElementSibling;
            content.classList.toggle('expanded');
        });
    });
}
```

#### 3.4 CSS 样式

**文件**：`frontend/css/style.css`

```css
/* 推理过程区域布局 */
.reasoning-area:empty { display: none; }
.answer-area { min-height: 0; }

/* 推理过程折叠块 */
.reasoning-block {
    background: var(--bg-tertiary);
    border: 1px solid var(--border-light);
    border-radius: var(--radius-sm);
    margin: 10px 0;
    overflow: hidden;
}

.reasoning-header {
    display: flex;
    align-items: center;
    gap: 8px;
    padding: 10px 14px;
    cursor: pointer;
    background: none;
    color: var(--text-secondary);
    font-size: 13px;
    border: none;
    width: 100%;
    text-align: left;
}

.reasoning-content {
    display: none;
    padding: 0 14px 12px;
    font-size: 13px;
    color: var(--text-secondary);
    line-height: 1.6;
}

.reasoning-content.expanded { display: block; }
.reasoning-header.expanded .arrow { transform: rotate(90deg); }

.reasoning-step {
    padding: 8px 0;
    border-bottom: 1px solid var(--border-light);
}
.reasoning-step:last-child { border-bottom: none; }
.reasoning-step-question { font-weight: 500; color: var(--accent); margin-bottom: 4px; }
.reasoning-step-answer { color: var(--text-secondary); }
```

---

### 四、reasoning_context 压缩

#### 4.1 实现位置

**文件**：`src/node/generate/generate.py`

#### 4.2 调用链路

```
_generate_current_answer()
  │
  ├── 拼接新的 reasoning_context
  │   new_reasoning_context = state["reasoning_context"] + f"问题：{q}\n答案：{answer}\n\n"
  │
  ├── 调用压缩函数
  │   new_reasoning_context = await compress_reasoning_context(self.llm, new_reasoning_context)
  │
  └── 返回到 state
      return {"reasoning_context": new_reasoning_context, ...}
```

#### 4.3 压缩逻辑

```python
MAX_REASONING_TOKENS = 2000

async def compress_reasoning_context(llm, reasoning_context, max_tokens=MAX_REASONING_TOKENS):
    if estimate_token_count(reasoning_context) <= max_tokens:
        return reasoning_context  # 未超阈值，直接返回

    # 使用 LLM 进行摘要压缩
    compress_prompt = ChatPromptTemplate.from_template("""
    请将以下子问题解答记录压缩为简洁的摘要，保留所有关键事实和结论：
    {reasoning_context}
    输出要求：
    - 保留每个子问题的核心答案
    - 去除冗余描述
    - 保持信息的准确性和完整性
    """)
    chain = compress_prompt | llm | StrOutputParser()
    return await chain.ainvoke({"reasoning_context": reasoning_context})
```

---

### 五、置信度评分

#### 5.1 实现位置

**文件**：`src/graph.py`，`_generate_current_answer()` 方法内

#### 5.2 计算逻辑

当 `is_final=True`（最终答案）时，基于 `state["retrieval_scores"]` 计算置信度：

```python
if is_final:
    scores = state.get("retrieval_scores", [])
    if scores:
        avg_score = sum(scores) / len(scores)
        source_count = len(scores)
        if avg_score >= 0.8:
            confidence_level = "高"
        elif avg_score >= 0.5:
            confidence_level = "中"
        else:
            confidence_level = "低"
        answer += f"\n\n📊 置信度：{confidence_level}（{avg_score:.2f}）| 参考来源：{source_count} 篇文档"
```

#### 5.3 分数来源

`retrieval_scores` 由 `_fusion_retrieve()` 节点在检索和重排序后写入 state，包含所有通过重排序阈值（≥0.8）的文档分数。

---

### 六、各场景行为总结

| 场景 | 推理过程展示 | 最终答案展示 | 流式效果 |
|------|:----------:|:----------:|:-------:|
| **不需要检索**（问候/闲聊） | 无 | `retrieve_or_respond` 完成后通过 `final_answer` 一次性展示 | 非流式 |
| **单跳问题**（简单事实查询） | 无 | `generate_current_answer`（`is_final=True`）完成后通过 `final_answer` 一次性展示 | 非流式 |
| **多跳问题**（复杂推理） | `decomposition` 实时展示子问题分解；`sub_answer` 实时追加推理步骤 | `synthesize` 节点的 token 流式展示合成答案 | 流式 |

---

### 七、涉及文件清单

| 文件 | 修改内容 |
|------|----------|
| `src/graph.py` | `start_stream()` 流式事件分发；`_retrieve_or_respond()` 直接回复返回 answer；token 白名单过滤 |
| `src/node/generate/generate.py` | `generate_answer_for_query_stream()` 流式生成函数；`compress_reasoning_context()` 压缩函数 |
| `server.py` | `POST /api/chat/stream` SSE 接口；`asyncio.Queue` + 心跳机制 |
| `frontend/js/chat.js` | `streamChat()` 流式渲染；`updateReasoningDisplay()` 推理步骤增量追加；`renderDecomposition()` 子问题分解渲染；`throttledRender()` 节流渲染；非流式降级 |
| `frontend/css/style.css` | `.reasoning-area`、`.answer-area` 布局；`.reasoning-block` 折叠块样式 |
