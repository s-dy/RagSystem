from utils.environ import set_huggingface_hf_env

set_huggingface_hf_env()

import json
import random
import uuid
import sys
from pathlib import Path
from dataclasses import asdict

from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.runnables import RunnableConfig

from src.graph import Graph
from src.eval.ragas_eval import RagEvaluator, EvalSample
from utils.async_task import async_run

# ---------------------------------------------------------------------------
# 数据加载
# ---------------------------------------------------------------------------

TEST_DATA_PATH = Path(__file__).parent / "test_data" / "test_row_data.json"


def load_test_data(limit: int = 1) -> list[dict]:
    with open(TEST_DATA_PATH, encoding="utf-8") as file:
        dataset = json.load(file)
    return random.choices(dataset, k=limit)


def make_config(recursion_limit: int = 25) -> RunnableConfig:
    return {
        "configurable": {
            "thread_id": str(uuid.uuid4()),
            "user_id": str(random.randint(1, 10000)),
        },
        "recursion_limit": recursion_limit,
    }


# ---------------------------------------------------------------------------
# 单跳测试（ainvoke）
# ---------------------------------------------------------------------------


async def test_single_hop():
    """单跳问答测试：逐条问题调用 graph.start()"""
    data = load_test_data(1)
    graph = Graph()

    for idx, item in enumerate(data, 1):
        print(f"【{idx}】单跳测试开始" + "=" * 50)
        reference_text = item["text"]
        config = make_config()
        print(
            f"user_id={config['configurable']['user_id']}  "
            f"thread_id={config['configurable']['thread_id']}"
        )

        for question_answer in item["qas"]:
            question = question_answer["question"]
            expected_answer = question_answer["answer"]
            inputs = {"messages": [{"role": "user", "content": question}]}

            response = await graph.start(inputs, config=config)
            llm_answer = _extract_answer(response)

            print("llm答案", llm_answer)
            print("预期答案", expected_answer)

            await _evaluate_and_log(
                question=question,
                llm_answer=llm_answer,
                retrieved_contexts=response.get("retrieved_documents", []),
                reference=reference_text,
            )


# ---------------------------------------------------------------------------
# 多跳测试（ainvoke）
# ---------------------------------------------------------------------------


async def test_multi_hop():
    """多跳问答测试：将多个子问题合并为一个复合问题"""
    data = load_test_data(1)
    graph = Graph()

    for idx, item in enumerate(data, 1):
        print(f"【{idx}】多跳测试开始" + "=" * 50)
        config = make_config()
        print(
            f"user_id={config['configurable']['user_id']}  "
            f"thread_id={config['configurable']['thread_id']}"
        )

        combined_question = "；".join(qa["question"] for qa in item["qas"])
        combined_answer = "\n".join(qa["answer"] for qa in item["qas"])

        inputs = {"messages": [{"role": "user", "content": combined_question}]}
        response = await graph.start(inputs, config=config)
        llm_answer = _extract_answer(response)

        print("llm答案", llm_answer)
        print("预期答案", combined_answer)

        await _evaluate_and_log(
            question=combined_question,
            llm_answer=llm_answer,
            retrieved_contexts=response.get("retrieved_documents", []),
            reference=item["text"],
        )


# ---------------------------------------------------------------------------
# 流式输出测试（start_stream）
# ---------------------------------------------------------------------------


async def test_stream():
    """流式输出测试：验证 start_stream 逐 token 返回"""
    graph = Graph()
    config = make_config()
    question = "水腺毛草的学名是什么？它通常被称之为什么？"
    inputs = {"messages": [HumanMessage(content=question)]}

    print("流式测试开始" + "=" * 50)
    print("问题", question)

    token_count = 0
    final_answer = ""
    sub_questions_received = []

    async for event in graph.start_stream(inputs, config):
        event_type = event.get("type", "")

        if event_type == "token":
            token_count += 1
            if token_count % 20 == 0:
                print(
                    f"已接收 {token_count} 个 token",
                    f"当前节点: {event.get('node', '')}",
                )

        elif event_type == "decomposition":
            sub_questions_received = event.get("sub_questions", [])
            print("子问题分解", sub_questions_received)

        elif event_type == "sub_answer":
            print(
                f"子问题中间答案: {event.get('sub_question', '')}",
                event.get("answer", "")[:100],
            )

        elif event_type == "final_answer":
            final_answer = event.get("answer", "")
            print("最终答案", final_answer[:200])

        elif event_type == "done":
            print("流式输出完成")

    print(f"流式测试结果: 共接收 {token_count} 个 token")
    assert token_count > 0, "未接收到任何 token"
    assert final_answer, "未接收到最终答案"
    print("流式测试通过 ✅")


# ---------------------------------------------------------------------------
# 辅助函数
# ---------------------------------------------------------------------------


def _extract_answer(response: dict) -> str:
    """从 graph 响应中提取最终答案文本"""
    if response.get("answer"):
        return response["answer"]

    messages = response.get("messages", [])
    if not messages:
        return ""

    last_message = messages[-1]
    if isinstance(last_message, AIMessage):
        return last_message.content
    elif isinstance(last_message, dict):
        return last_message.get("content", "")
    return str(last_message)


async def _evaluate_and_log(
    question: str,
    llm_answer: str,
    retrieved_contexts: list[str],
    reference: str,
):
    """使用 RagEvaluator 评估并记录结果"""
    try:
        evaluator = RagEvaluator()
        sample = EvalSample(
            user_input=question,
            response=llm_answer,
            retrieved_contexts=retrieved_contexts,
            reference=reference,
        )
        scores = await evaluator.evaluate_sample(sample)
        scores_dict = asdict(scores)

        print("评估结果", scores_dict)

        for metric_name, metric_value in scores_dict.items():
            display = f"{metric_value:.4f}" if metric_value is not None else "N/A"
            print(f"  {metric_name}", display)

    except Exception as eval_error:
        print(f"评估失败: {eval_error}", level="WARNING")


# ---------------------------------------------------------------------------
# 入口
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    test_name = "stream"

    test_map = {
        "single": test_single_hop,
        "multi": test_multi_hop,
        "stream": test_stream,
    }

    selected_test = test_map.get(test_name)
    if selected_test is None:
        print(f"未知测试: {test_name}，可选: {', '.join(test_map.keys())}")
        sys.exit(1)

    print(f"运行测试: {test_name}")
    async_run(selected_test())
