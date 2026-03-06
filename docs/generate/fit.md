## 生成与可解释性优化方案

### 一、现状分析

当前项目的生成模块位于 `src/node/generate/generate.py`，通过 `generate_answer_for_query()` 和 `synthesize_final_subs()` 两个函数完成答案生成。在 `src/graph.py` 的 `_generate_current_answer()` 节点中被调用。

| 模块 | 当前实现 | 说明 |
|------|:--------:|------|
| 最终答案生成 | `final_system_prompt` + `ainvoke()` | 基于检索内容和对话历史生成 |
| 子问题答案生成 | `sub_system_prompt` + `ainvoke()` | 基于 reasoning_context 和检索内容生成 |
| 多跳综合 | `synthesize_system_prompt` + `ainvoke()` | 整合子问题答案生成最终回答 |
| 来源引用 | 无 | Prompt 明确要求"不要提及【根据检索到的信息】" |
| 流式输出 | 未实现 | 全部使用 `ainvoke()` 阻塞式返回 |
| 推理过程展示 | 不可见 | `reasoning_context` 仅内部传递 |

**主要局限性：**

- Prompt 第 8 条规则"不要提及【根据检索到的信息】"导致答案来源完全不可追溯
- `reasoning_context` 每轮追加 `f"问题：{current_q}\n答案：{answer}\n\n"`，无长度限制，可能导致 token 溢出
- 多跳推理的子问题分解过程和中间答案仅写入日志，用户端无感知
- 重排序分数（`CrossEncoderRanker` 阈值 0.8）和检索分数（阈值 0.2）仅用于内部过滤，不对外暴露
- 所有生成均使用 `ainvoke()` 同步返回，用户需等待完整生成后才能看到结果

---

### 二、优化策略

#### 策略 1：答案来源引用（🔴 高优先级）

**问题**：`final_system_prompt` 第 8 条规则要求"不要提及【根据检索到的信息】"，导致用户无法判断答案依据。

**方案**：修改 Prompt，要求模型在答案中自然地标注引用来源：

```python
final_system_prompt = """
你是一个专业、严谨的智能助手，请根据以下检索到的信息和对话历史来回答用户问题。

检索到的信息（每条信息附带来源编号）：
{content}

对话历史：
{conversation_context}

规则：
1. 基于检索到的信息回答问题，用简洁、准确的语言表述。
2. 在引用具体信息时，以自然的方式标注来源，如"根据[来源1]..."或在句末标注[1][2]。
3. 如果信息不足，回答："根据现有资料，无法回答该问题。"
4. 严格基于检索内容回答，不要编造信息。
5. 在答案末尾附加"参考来源"列表，列出引用的文档标题或链接。
"""
```

**配套改动**：

- 在 `_fusion_retrieve()` 中为每条检索结果编号，格式如 `[来源1] 内容... (来自: 文档标题)`
- 在 Document 的 metadata 中保留 `source`（文件路径）和 `section_title`（章节标题），传递到生成阶段

**预期收益**：用户可追溯每条信息的来源，显著提升答案可信度。

---

#### 策略 2：流式输出（🔴 高优先级）

**问题**：`generate_answer_for_query()` 使用 `chain.ainvoke()` 阻塞式返回，用户需等待完整生成后才能看到结果，体验差。

**方案**：将生成函数改为 `AsyncGenerator`，使用 `chain.astream()` 逐 token 返回：

```python
from typing import AsyncGenerator

async def generate_answer_for_query_stream(
        llm: BaseChatModel,
        query: str,
        docs_content: str,
        conversation_context: str = "",
        reasoning_context: str = "",
        is_final: bool = False,
) -> AsyncGenerator[str, None]:
    """流式生成答案，逐 token 返回"""
    system_prompt = final_system_prompt if is_final else sub_system_prompt
    full_context = conversation_context if is_final else reasoning_context

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{query}")
    ])
    chain = prompt | llm | StrOutputParser()

    async for chunk in chain.astream({
        "query": query,
        "content": docs_content,
        "conversation_context": conversation_context,
        "reasoning_context": full_context,
    }):
        yield chunk
```

**配套改动**：

- `_generate_current_answer()` 节点内部仍需收集完整答案（用于 reasoning_context 和 store 存储），但可通过 callback 机制将流式 token 推送到前端
- 前端通过 SSE（Server-Sent Events）或 WebSocket 接收流式 token
- 保留 `ainvoke()` 版本作为非流式降级方案

**预期收益**：用户在生成开始后即可看到逐字输出，首字延迟从数秒降至毫秒级。

---

#### 策略 3：推理过程透明化（🔴 高优先级）

**问题**：多跳推理的 `reasoning_context` 仅在 `_generate_current_answer()` 内部拼接传递，子问题分解和中间答案对用户完全不可见。

**方案**：在多跳推理的关键节点向用户推送结构化的推理过程：

```python
# 在 _prepare_next_step() 中，子问题分解完成后推送
reasoning_steps = {
    "type": "decomposition",
    "original_query": query,
    "sub_questions": sub_questions,
    "message": f"已将问题分解为 {len(sub_questions)} 个子问题"
}

# 在 _generate_current_answer() 中，每个子问题回答后推送
reasoning_steps = {
    "type": "sub_answer",
    "sub_question": current_q,
    "answer": answer,
    "step": current_step,
    "total_steps": total_steps,
    "message": f"正在回答第 {current_step}/{total_steps} 个子问题"
}
```

**实现方式**：

- 定义 `ReasoningStep` 数据类，统一推理过程的结构
- 在 State 中新增 `reasoning_steps: list[dict]` 字段，记录完整推理链路
- 通过 LangGraph 的 `stream_mode="updates"` 将每个节点的状态变更实时推送到前端
- 最终答案中附带完整的推理链路，供用户查看

**预期收益**：用户可看到问题被分解成了哪些子问题、每个子问题的中间答案，理解系统的推理逻辑。

---

#### 策略 4：reasoning_context 压缩（🔴 高优先级）

**问题**：`_generate_current_answer()` 中 `reasoning_context` 每轮直接拼接 `f"问题：{current_q}\n答案：{answer}\n\n"`，无长度限制。当子问题较多或答案较长时，可能超出 LLM 上下文窗口。

**方案**：对 `reasoning_context` 设置最大 token 数，超过时使用 LLM 进行摘要压缩：

```python
MAX_REASONING_TOKENS = 2000

async def compress_reasoning_context(
    llm: BaseChatModel,
    reasoning_context: str,
    max_tokens: int = MAX_REASONING_TOKENS,
) -> str:
    """当 reasoning_context 超过阈值时，使用 LLM 压缩为摘要"""
    if estimate_token_count(reasoning_context) <= max_tokens:
        return reasoning_context

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

**调用位置**：在 `_generate_current_answer()` 拼接新的 reasoning_context 后，调用压缩函数：

```python
new_reasoning_context = state.get("reasoning_context", "") + f"问题：{current_q}\n答案：{answer}\n\n"
new_reasoning_context = await compress_reasoning_context(self.llm, new_reasoning_context)
```

**预期收益**：避免 token 溢出导致的生成失败，同时保留关键推理信息。

---

#### 策略 5：置信度评分（🟡 中优先级）

**问题**：重排序分数（`CrossEncoderRanker` 阈值 0.8）和检索相似度分数（阈值 0.2）仅用于内部过滤，用户无法了解答案的可靠程度。

**方案**：基于检索和重排序分数计算答案置信度，附加到最终响应中：

```python
from dataclasses import dataclass

@dataclass
class AnswerWithConfidence:
    answer: str
    confidence: float          # 0.0 ~ 1.0
    confidence_level: str      # "高" / "中" / "低"
    source_count: int          # 引用的来源数量
    avg_relevance_score: float # 平均检索相关性分数

def calculate_confidence(
    reranked_scores: list[float],
    retrieval_scores: list[float],
    has_sufficient_context: bool,
) -> float:
    """基于多维度信号计算答案置信度"""
    if not reranked_scores:
        return 0.0

    avg_rerank = sum(reranked_scores) / len(reranked_scores)
    avg_retrieval = sum(retrieval_scores) / len(retrieval_scores)
    source_factor = min(len(reranked_scores) / 3, 1.0)
    context_factor = 1.0 if has_sufficient_context else 0.7

    confidence = (avg_rerank * 0.4 + avg_retrieval * 0.3 + source_factor * 0.2 + context_factor * 0.1)
    return round(min(confidence, 1.0), 2)
```

**展示方式**：在最终答案末尾附加置信度信息：

```
答案内容...

📊 置信度：高（0.85）| 参考来源：3 篇文档
```

**预期收益**：用户可根据置信度判断答案可靠性，低置信度时主动寻求其他信息源。

---

#### 策略 6：子问题分解可视化（🟡 中优先级）

**问题**：`_prepare_next_step()` 中子问题分解结果仅通过 `monitor_task_status()` 写入日志，用户端无感知。

**方案**：在子问题分解完成后，通过消息流向用户展示分解结果：

```python
# 在 _prepare_next_step() 返回时，将分解信息写入 messages
decomposition_message = AIMessage(
    content=f"🔍 为了更好地回答您的问题，我将其分解为以下子问题：\n"
            + "\n".join(f"  {i+1}. {q}" for i, q in enumerate(sub_questions))
            + "\n\n正在逐一检索和回答...",
    additional_kwargs={"type": "decomposition"}
)

return {
    "sub_questions": sub_questions,
    "current_sub_question": sub_questions[0],
    "messages": [decomposition_message],
}
```

**预期收益**：用户可看到问题被分解成了哪些子问题，理解系统的处理逻辑，增强信任感。

---

### 三、实施优先级总结

| 优先级 | 优化项 | 涉及文件 | 实施难度 | 预期收益 |
|:------:|--------|----------|:--------:|:--------:|
| 🔴 高 | 答案来源引用 | `generate.py`、`graph.py`、`fusion_retrieve.py` | 中 | 高 |
| 🔴 高 | 流式输出 | `generate.py`、`graph.py`、前端接入层 | 中 | 高 |
| 🔴 高 | 推理过程透明化 | `graph.py`（`_prepare_next_step`、`_generate_current_answer`） | 中 | 高 |
| 🔴 高 | reasoning_context 压缩 | `generate.py`、`graph.py` | 低 | 高 |
| 🟡 中 | 置信度评分 | `graph.py`、`cross_encoder_ranker.py`、`fusion_retrieve.py` | 中 | 中 |
| 🟡 中 | 子问题分解可视化 | `graph.py`（`_prepare_next_step`） | 低 | 中 |

### 四、评估验证

每项优化实施后，建议通过项目的 RAG 评估体系（`src/eval/ragas_eval.py`）验证效果：

- **Faithfulness**：来源引用优化后，衡量答案是否更忠实于检索内容
- **Answer Relevancy**：流式输出不应影响答案质量，需确认指标无下降
- **Context Relevance**：推理过程透明化后，衡量上下文利用是否更充分
- **用户体验指标**：流式输出的首字延迟（TTFT）、子问题分解的展示效果需通过人工评估

建议在同一批评估样本上对比优化前后的指标变化，确保改进有据可依。
