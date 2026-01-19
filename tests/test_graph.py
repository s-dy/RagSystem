import json
import random
import uuid

from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableConfig
from langfuse import get_client, propagate_attributes

from src.graph import Graph
from src.monitoring.logger import monitor_task_status
from src.services.GradeModel import DocumentGrader
from utils.async_task import async_run


def get_data(limit: int = 1):
    data_path = 'eval/test_row_data.json'
    with open(data_path) as f:
        dataset = json.load(f)

    return random.choices(dataset, k=limit)


langfuse = get_client()


async def test_graph():
    data = get_data(1)
    for idx, item in enumerate(data, 1):
        monitor_task_status(f"【{idx}】Start" + '=' * 50)
        user_id = str(random.randint(1, 10000))
        config: RunnableConfig = {'configurable': {'thread_id': str(uuid.uuid4()), 'user_id': user_id},'recursion_limit':15}
        monitor_task_status(
            f'开始单跳测试 ==> user_id={config["configurable"]["user_id"]} ==> thread_id={config["configurable"]["thread_id"]}')
        for question_answer in item['qas']:
            question = question_answer['question']
            answers = question_answer['answer']
            inputs = {
                "messages": [{"role": "user", "content": question}],
            }
            await invoke(inputs, config, answers)


async def test_multi_graph():
    data = get_data(1)
    for idx, item in enumerate(data, 1):
        monitor_task_status(f"【{idx}】Start" + '=' * 50)
        user_id = str(random.randint(1, 10000))
        config: RunnableConfig = {'configurable': {'thread_id': str(uuid.uuid4()), 'user_id': user_id},'recursion_limit':25}
        monitor_task_status(
            f'开始多跳测试 ==> user_id={config["configurable"]["user_id"]} ==> thread_id={config["configurable"]["thread_id"]}')

        content,ans = '',''
        for question_answer in item['qas']:
            question = question_answer['question']
            answers = question_answer['answer']
            content += question
            ans = ans + '\n' + answers
        inputs = {
            "messages": [{"role": "user", "content": '李呈瑞于哪一年参加红军？他获得过哪些勋章？他于哪一年逝世？他在抗战中担任过哪些职位？'}],
        }
        await invoke(inputs, config, ans)


def get_similarity(llm_answer,answer) -> float:
    calculate = len(set(llm_answer) & set(answer)) / len(answer)
    return calculate

# 初始化评估器
doc_grader = DocumentGrader(threshold=0.7)
async def invoke(inputs, config, answers: str):
    graph = Graph()
    monitor_task_status("invoke question" + "=" * 50)

    session_id = config["configurable"]["thread_id"]

    with langfuse.start_as_current_observation(as_type="span", name="hybridRagTest") as span:
        with propagate_attributes(session_id=session_id, user_id=config["configurable"]["user_id"], tags=['test']):
            span.update_trace(input=inputs['messages'])

            # 获取响应
            response = await graph.start(inputs, config=config)
            msg = response['messages'][-1]
            if isinstance(msg,AIMessage):
                llm_answer = response['messages'][-1].content
            elif isinstance(msg,dict):
                llm_answer = msg['messages'][-1]['content']
            else:
                raise NotImplementedError
            monitor_task_status('llm答案: ',llm_answer)
            monitor_task_status('预期答案: ',answers)

            # 计算llm答案相似度分数
            question = inputs['messages'][0]['content']
            # similarity = doc_grader.get_similarity(question, llm_answer)

            # 计算与期望答案的相似度
            # expected_similarity = doc_grader.get_similarity(llm_answer, answers)
            expected_similarity = get_similarity(llm_answer, answers)
            # 二值化分类（基于阈值）
            # is_relevant_pred = similarity >= doc_grader.threshold
            # is_relevant_true = expected_similarity >= 0.5

            span.update_trace(output={
                "response": llm_answer,
                # "similarity_score": similarity,
                "expected_similarity": expected_similarity
            })

            # 评估分数
            evaluate_score(span, {
                # 'similarity_score': similarity,
                'similarity_score': expected_similarity,
                # 'is_relevant_pred': is_relevant_pred,
                # 'is_relevant_true': is_relevant_true
            })


def evaluate_score(span, evaluate_data):
    """评估分数并记录到Langfuse"""
    span.score(
        name="llm Correlation Metrics",
        value=evaluate_data['similarity_score'],
        data_type="NUMERIC",
    )

    # 3. 记录详细的评估结果
    monitor_task_status("\n" + "=" * 50)
    monitor_task_status("评估结果:")
    monitor_task_status(f"实际答案相似度: {evaluate_data['similarity_score']:.4f}")
    monitor_task_status("=" * 50)


if __name__ == '__main__':
    async_run(test_graph())