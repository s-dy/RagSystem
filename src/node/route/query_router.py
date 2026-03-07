from typing import Dict, List, Any, Tuple
from langchain_core.language_models import BaseChatModel

from src.observability.logger import monitor_task_status
from utils.ParallelChain import ParallelChain


class QueryRouter(ParallelChain):
    def __init__(self, llm: BaseChatModel, temperature: float = 0.1):
        super().__init__(llm)
        self.llm = llm
        self.temperature = temperature

    def _format_index_dict(self, index_dict: List[Dict]) -> str:
        """格式化知识库字典为可读字符串"""
        try:
            formatted = []
            for info in index_dict:
                index_name = info.get("index", "unknown")
                description = info.get("description", "无描述")
                # domain = info.get("domain", "general")
                prompt_str = f"- {index_name}    描述: {description}"
                # prompt_str += f"    领域: {domain}"
                # if "keywords" in info:
                #     prompt_str += f"    关键词: {', '.join(info.get('keywords', []))}"
                formatted.append(prompt_str)
            return "\n".join(formatted)
        except Exception as e:
            monitor_task_status(f"格式化知识库字典失败: {e}")
            return str(index_dict)

    async def multi_queries_index_router(
            self,
            queries: List[str],
            index_dict: List[Dict],
            external_tools:list=None,
            top_k: int = 3
    ) -> List[Tuple[str, List[Dict]]]:
        """
        并行路由所有查询到内部知识库或外部工具
        """
        monitor_task_status(f"multi_index_router_queries", queries)

        # 构建路由判断链
        for query in queries:
            messages = [
                ('system', f"""你是一个智能路由助手，需判断用户查询应使用内部知识库还是外部工具。

                可用的内部知识库：
                {{formatted_indices}}

                可用的外部工具：
                - web_search: 用于获取最新信息、社交媒体内容、新闻、无法在知识库中找到的内容
                {{external_tools}}

                判断规则：
                1. 如果查询涉及以下内容，优先选择调用外部工具：
                   - 社交媒体
                   - 最新事件/实时数据
                   - 获取/抓取/下载具体文章或内容
                   - 知识库中明显没有覆盖的领域
                2. 否则，从内部知识库中选择最相关的（最多{top_k}个）
                3. 可以同时返回 internal 和 external 建议（按相关性排序）

                输出格式（JSON数组）：
                [
                  {{{{
                    "type": "internal" or "external",
                    "index": "knowledge_base_name",  // type=internal 时
                    "tool": "web_search",             // type=external 时
                    "reason": "简要原因",
                    "score": 0.0-1.0
                  }}}}
                ]

                如果完全不相关，返回空数组 []。
                """),
                ('human', f'用户查询：{query}')
            ]
            self.task_map[query] = self.create_chain(messages,parse='json',config={'llm_temperature': self.temperature})

        formatted_indices = self._format_index_dict(index_dict)
        responses = await self.runnable_parallel({'formatted_indices': formatted_indices,'external_tools': external_tools})
        result = self.parse_parallel_response(responses)
        return result

    def parse_parallel_response(
            self,
            responses: Dict[str, Any]
    ) -> List[Tuple[str, List[Dict]]]:
        results = []
        for key, response in responses.items():
            if not response:
                monitor_task_status(f"multi_index_router_response_empty for query: {key}")
                results.append((key, []))
                continue

            validated_recommendations = []
            for rec in response:
                if not isinstance(rec, dict):
                    continue

                score = rec.get("score", 0.5)
                if "index" not in rec:
                    continue
                validated_rec = {
                    "index": rec["index"],
                    "score": score
                }
                validated_recommendations.append(validated_rec)

            # 按评分排序
            validated_recommendations.sort(key=lambda x: x["score"], reverse=True)
            monitor_task_status(f"multi_index_router_completed", {
                "recommendations_count": len(validated_recommendations),
                "recommendations": validated_recommendations
            })
            results.append((key, validated_recommendations))

        return results

    def _build_fallback_routes(self, index_dict: List[Dict]) -> List[Dict]:
        """构建回退路由：返回所有知识库，用于 LLM 路由失败时兜底"""
        fallback = [
            {"index": info.get("index", ""), "score": 0.5}
            for info in index_dict
            if info.get("index")
        ]
        monitor_task_status("query_router_fallback", {
            "reason": "LLM 路由失败或返回空结果，回退到全部知识库",
            "fallback_indices": [r["index"] for r in fallback],
        })
        return fallback

    async def multi_all_queries_index_router(self, queries: List[str], index_dict: List[Dict], top_k: int = 3) -> List[Dict]:
        """将所有query一次性使用一个llm完成路由，失败时回退到全部知识库"""
        try:
            unique_queries = list(set(queries))
            combined_query = "；\n\n".join(unique_queries)
            messages = [
                ('system', f"""
                你是一个智能路由助手，需判断用户查询应使用内部知识库还是外部工具。

                可用的内部知识库：
                {{formatted_indices}}

                判断规则：
                1. 只返回最终结果，不反回其他无关内容。
                2. 从内部知识库中选择最相关的（最多{top_k}个）
                3. 严格按照可用的知识库进行选择，不要编造。

                输出格式（JSON数组）：
                [
                  {{{{
                    "index": "knowledge_base_name",
                    "score": 0.0-1.0
                  }}}}
                ]

                如果完全不相关，返回空数组 []。
                """),
                ('human', f'用户有多个相关查询：\n\n {combined_query} \n\n请为每个子查询分别路由')
            ]
            self.task_map['all_queries'] = self.create_chain(messages, parse='json', config={'llm_temperature': 0})
            formatted_indices = self._format_index_dict(index_dict)
            responses = await self.runnable_parallel(
                {'formatted_indices': formatted_indices})
            result = self.parse_parallel_response(responses)
            if result:
                result = result[0][1]

            if not result:
                monitor_task_status("LLM 路由返回空结果，触发回退", level="WARNING")
                return self._build_fallback_routes(index_dict)

            return result

        except Exception as error:
            monitor_task_status(f"LLM 路由调用异常: {error}，触发回退", level="WARNING")
            return self._build_fallback_routes(index_dict)