import asyncio
import json
import os
from dataclasses import dataclass, field, asdict
from typing import Optional

from utils.environ import set_huggingface_hf_env

set_huggingface_hf_env()

from dotenv import load_dotenv
from openai import AsyncOpenAI
from ragas.llms import llm_factory
from ragas.embeddings import HuggingFaceEmbeddings
from ragas.metrics.collections import (
    Faithfulness,
    AnswerRelevancy,
    ContextRelevance,
    ContextRecall,
)

from src.observability.logger import get_logger

logger = get_logger(__name__)

load_dotenv()


@dataclass
class EvalSample:
    """单条评估样本"""

    user_input: str  # 用户输入
    response: str  # LLM 回答
    retrieved_contexts: list[str]  # 检索到的上下文
    reference: Optional[str] = None  # 参考答案


@dataclass
class EvalScores:
    """单条样本的评估分数"""

    faithfulness: Optional[float] = None  # 忠实度
    answer_relevancy: Optional[float] = None  # 答案相关性
    context_relevance: Optional[float] = None  # 上下文相关性
    context_recall: Optional[float] = None  # 上下文召回率


@dataclass
class EvalReport:
    """批量评估报告"""

    sample_count: int = 0  # 样本数量
    avg_faithfulness: Optional[float] = None
    avg_answer_relevancy: Optional[float] = None
    avg_context_relevance: Optional[float] = None
    avg_context_recall: Optional[float] = None
    details: list[dict] = field(default_factory=list)


class RagEvaluator:
    """RAG 评估器，基于 ragas 框架的核心指标评估"""

    def __init__(self):
        client = AsyncOpenAI(
            api_key=os.getenv("QWEN_API_KEY"),
            base_url=os.getenv("QWEN_BASE_URL"),
        )
        self.llm = llm_factory(model=os.getenv("QWEN_MODEL_NAME"), client=client)
        self.embeddings = HuggingFaceEmbeddings(
            model="BAAI/bge-large-zh-v1.5",
        )

        self._faithfulness = Faithfulness(llm=self.llm)
        self._answer_relevancy = AnswerRelevancy(
            llm=self.llm, embeddings=self.embeddings
        )
        self._context_relevance = ContextRelevance(llm=self.llm)
        self._context_recall = ContextRecall(llm=self.llm)

    async def evaluate_faithfulness(self, sample: EvalSample) -> float:
        """
        忠实度：回答中的主张是否能从检索上下文中推断出来。
        分数越高表示回答越忠实于检索到的文档。
        """
        result = await self._faithfulness.ascore(
            user_input=sample.user_input,
            response=sample.response,
            retrieved_contexts=sample.retrieved_contexts,
        )
        return result.value

    async def evaluate_answer_relevancy(self, sample: EvalSample) -> float:
        """
        答案相关性：回答与用户问题的匹配程度。
        通过生成反向问题并计算与原始问题的余弦相似度来衡量。
        """
        result = await self._answer_relevancy.ascore(
            user_input=sample.user_input,
            response=sample.response,
        )
        return result.value

    async def evaluate_context_relevance(self, sample: EvalSample) -> float:
        """
        上下文相关性：检索到的上下文是否与用户问题相关。
        评估检索质量，分数越高表示检索越精准。
        """
        result = await self._context_relevance.ascore(
            user_input=sample.user_input,
            retrieved_contexts=sample.retrieved_contexts,
        )
        return result.value

    async def evaluate_context_recall(self, sample: EvalSample) -> Optional[float]:
        """
        上下文召回率：检索到的上下文是否覆盖了参考答案中的关键信息。
        需要提供 reference（参考答案），否则跳过。
        """
        if not sample.reference:
            return None
        result = await self._context_recall.ascore(
            user_input=sample.user_input,
            retrieved_contexts=sample.retrieved_contexts,
            reference=sample.reference,
        )
        return result.value

    async def evaluate_sample(self, sample: EvalSample) -> EvalScores:
        """对单条样本运行所有指标评估"""
        tasks = [
            self.evaluate_faithfulness(sample),
            self.evaluate_answer_relevancy(sample),
            self.evaluate_context_relevance(sample),
        ]

        has_reference = sample.reference is not None
        if has_reference:
            tasks.append(self.evaluate_context_recall(sample))

        results = await asyncio.gather(*tasks, return_exceptions=True)

        scores = EvalScores()
        metric_names = ["faithfulness", "answer_relevancy", "context_relevance"]
        if has_reference:
            metric_names.append("context_recall")

        for name, result in zip(metric_names, results):
            if isinstance(result, Exception):
                logger.warning(f"[RagEvaluator] 评估指标 {name} 失败: {result}")
                setattr(scores, name, None)
            else:
                setattr(scores, name, result)

        return scores

    async def evaluate_batch(self, samples: list[EvalSample]) -> EvalReport:
        """批量评估多条样本，生成汇总报告"""
        report = EvalReport(sample_count=len(samples))
        all_scores: list[EvalScores] = []

        for index, sample in enumerate(samples):
            logger.info(
                f"[RagEvaluator] 评估样本 {index + 1}/{len(samples)}: {sample.user_input[:50]}..."
            )
            scores = await self.evaluate_sample(sample)
            all_scores.append(scores)
            report.details.append(
                {
                    "user_input": sample.user_input,
                    "scores": asdict(scores),
                }
            )

        valid_faithfulness = [
            s.faithfulness for s in all_scores if s.faithfulness is not None
        ]
        valid_answer_relevancy = [
            s.answer_relevancy for s in all_scores if s.answer_relevancy is not None
        ]
        valid_context_relevance = [
            s.context_relevance for s in all_scores if s.context_relevance is not None
        ]
        valid_context_recall = [
            s.context_recall for s in all_scores if s.context_recall is not None
        ]

        if valid_faithfulness:
            report.avg_faithfulness = sum(valid_faithfulness) / len(valid_faithfulness)
        if valid_answer_relevancy:
            report.avg_answer_relevancy = sum(valid_answer_relevancy) / len(
                valid_answer_relevancy
            )
        if valid_context_relevance:
            report.avg_context_relevance = sum(valid_context_relevance) / len(
                valid_context_relevance
            )
        if valid_context_recall:
            report.avg_context_recall = sum(valid_context_recall) / len(
                valid_context_recall
            )

        return report

    @staticmethod
    def print_report(report: EvalReport):
        """打印评估报告"""
        print("\n" + "=" * 60)
        print("RAG 评估报告")
        print("=" * 60)
        print(f"样本数量: {report.sample_count}")
        print("-" * 40)

        metrics = [
            ("忠实度 (Faithfulness)", report.avg_faithfulness),
            ("答案相关性 (Answer Relevancy)", report.avg_answer_relevancy),
            ("上下文相关性 (Context Relevance)", report.avg_context_relevance),
            ("上下文召回率 (Context Recall)", report.avg_context_recall),
        ]
        for name, value in metrics:
            display_value = f"{value:.4f}" if value is not None else "N/A"
            print(f"  {name}: {display_value}")

        print("-" * 40)
        print("各样本详情:")
        for index, detail in enumerate(report.details):
            print(f"\n  [{index + 1}] {detail['user_input'][:60]}")
            for metric_key, metric_value in detail["scores"].items():
                display = f"{metric_value:.4f}" if metric_value is not None else "N/A"
                print(f"      {metric_key}: {display}")
        print("=" * 60)

    @staticmethod
    def save_report(report: EvalReport, output_path: str):
        """将评估报告保存为 JSON 文件"""
        with open(output_path, "w", encoding="utf-8") as file:
            json.dump(asdict(report), file, ensure_ascii=False, indent=2)
        print(f"评估报告已保存到: {output_path}")
