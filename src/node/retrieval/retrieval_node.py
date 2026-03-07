"""检索节点 Mixin

包含：
- _retrieve_internal: 内部向量检索
- _retrieve_external: 外部搜索（MCP/Bing）
- _fusion_retrieve: 融合检索 + 重排序
- _grade_documents: 检索结果评分
"""

import asyncio
from typing import List

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableConfig
from langgraph.store.base import BaseStore
from langgraph.types import StreamWriter

from src.core.exceptions import InternalRetrievalError, ExternalSearchError
from src.observability.logger import get_logger
from src.services.task_analyzer import TaskCharacteristics, TaskType
from .fusion_retrieve import FusionRetrieve, RetrievedDoc

logger = get_logger(__name__)


class RetrievalNodeMixin:
    """检索节点方法集合，通过 Mixin 注入到 Graph 类"""

    async def __retrieve_internal(self, router_index) -> List[RetrievedDoc]:
        """处理内部检索，返回 RetrievedDoc 列表"""
        doc_result: List[RetrievedDoc] = []
        if not router_index:
            return doc_result
        logger.info(
            f"[RetrievalNode] 开始内部检索: collections={list(router_index.keys())}"
        )
        search_model = FusionRetrieve(
            use_parent_child=self.config.enable_parent_child_retrieval
        )
        for collection_index, queries in router_index.items():
            try:
                result = await search_model.search_queries(
                    queries, collection_name=collection_index
                )
                if not result:
                    continue
                for doc_list in result:
                    if doc_list:
                        doc_result.extend(doc_list)
            except ConnectionError as conn_err:
                raise InternalRetrievalError(collection_index, cause=conn_err)
            except TimeoutError as timeout_err:
                logger.warning(
                    f"[RetrievalNode] 检索超时: collection={collection_index}, error={timeout_err}"
                )
            except Exception as exc:
                logger.warning(
                    f"[RetrievalNode] 检索出错: collection={collection_index}, error={exc}"
                )
        logger.info(f"[RetrievalNode] 内部检索完成: total_docs={len(doc_result)}")
        return doc_result

    async def __retrieve_external(self, query: str, max_retries: int = 2) -> list:
        """处理外部检索，支持重试"""
        external_docs = []
        if not query:
            return external_docs

        for attempt in range(max_retries + 1):
            try:
                logger.info(
                    f"[RetrievalNode] 开始外部搜索: attempt={attempt + 1}/{max_retries + 1}"
                )

                prompt = ChatPromptTemplate.from_template("""
                你是一个优化在搜索引擎中搜索查询的助手，需要重写用户问题，便于使用搜索引擎搜索。

                以下是需要重写的用户问题：
                【{question}】

                要求：
                1. 只返回最终的答案，不用返回其他无关内容。
                """)
                chain = prompt | self.llm | StrOutputParser()
                search_query = await chain.ainvoke({"question": query})
                logger.debug(
                    f"[RetrievalNode] 搜索查询重写: original={query[:50]}..., rewritten={search_query[:50]}..."
                )

                search_result = self.tools_pool.get_response(
                    await self.tools_pool.call_tool(
                        "bing_search", {"query": search_query}
                    )
                )
                uids, uid_map = [], {}
                if len(search_result):
                    results = search_result[0].get("results")
                else:
                    results = []
                for item in results:
                    uids.append(item["uuid"])
                    uid_map[item["uuid"]] = item["url"]

                pages_results = self.tools_pool.get_response(
                    await self.tools_pool.call_tool(
                        "crawl_webpage", {"uuids": uids, "url_map": uid_map}
                    )
                )
                if pages_results:
                    pages_results = pages_results[0]
                for item in pages_results:
                    if item.get("content"):
                        external_docs.append(item["content"])

                logger.info(
                    f"[RetrievalNode] 外部搜索完成: docs_count={len(external_docs)}"
                )
                return external_docs

            except ConnectionError as conn_err:
                logger.warning(
                    f"[RetrievalNode] 外部搜索连接失败: attempt={attempt + 1}, error={conn_err}"
                )
                if attempt == max_retries:
                    raise ExternalSearchError(
                        "外部搜索连接失败",
                        attempt=attempt + 1,
                        max_retries=max_retries + 1,
                        cause=conn_err,
                    )
                await asyncio.sleep(1 * (attempt + 1))
            except Exception as exc:
                logger.warning(
                    f"[RetrievalNode] 外部检索失败: attempt={attempt + 1}, error={exc}"
                )
                if attempt == max_retries:
                    logger.warning(
                        "[RetrievalNode] 外部检索已达最大重试次数，返回空结果"
                    )
                    return external_docs
                await asyncio.sleep(1 * (attempt + 1))

        return external_docs

    async def __fusion_retrieve(
        self, state, config: RunnableConfig, store: BaseStore, writer: StreamWriter
    ) -> dict:
        """融合检索节点：支持来源编号和置信度分数"""
        logger.info("[RetrievalNode] 开始融合检索")
        task_characteristics = state.get("task_characteristics") or TaskCharacteristics(
            task_type=TaskType.FACT_RETRIEVAL, requires_external_tools=False
        )

        # 发送检索开始事件
        collection_names = (
            list(state["router_index"].keys()) if state.get("router_index") else []
        )
        writer(
            {
                "type": "retrieval_progress",
                "stage": "start",
                "message": f"正在检索 {len(collection_names)} 个知识库...",
                "collections": collection_names,
            }
        )

        # 内部检索（返回 RetrievedDoc 列表）
        internal_docs = await self.__retrieve_internal(state["router_index"])

        # 外部检索（仅在前端开启联网搜索时才调用）
        query = state.get("current_sub_question") or state["original_query"]
        enable_web_search = config.get("configurable", {}).get(
            "enable_web_search", False
        )
        external_retrieved_docs: List[RetrievedDoc] = []
        if enable_web_search and (
            not internal_docs or task_characteristics.requires_external_tools
        ):
            external_texts = await self.__retrieve_external(query)
            for text in external_texts:
                external_retrieved_docs.append(
                    RetrievedDoc(content=text, source="外部搜索", score=0.5)
                )

        # 融合 + 近似去重（基于 embedding 余弦相似度）
        all_retrieved = internal_docs + external_retrieved_docs
        unique_retrieved: List[RetrievedDoc] = []
        if all_retrieved:
            doc_texts = [d.content for d in all_retrieved]
            embeddings = await asyncio.to_thread(
                self.document_grader.model.encode,
                doc_texts,
                convert_to_tensor=True,
            )
            from sentence_transformers import util as st_util

            dedup_similarity_threshold = 0.92
            for idx, retrieved_doc in enumerate(all_retrieved):
                is_duplicate = False
                for kept_idx in range(len(unique_retrieved)):
                    original_idx = doc_texts.index(unique_retrieved[kept_idx].content)
                    similarity = st_util.pytorch_cos_sim(
                        embeddings[idx],
                        embeddings[original_idx],
                    ).item()
                    if similarity >= dedup_similarity_threshold:
                        is_duplicate = True
                        break
                if not is_duplicate:
                    unique_retrieved.append(retrieved_doc)
            logger.debug(
                f"[RetrievalNode] 文档去重: before={len(all_retrieved)}, after={len(unique_retrieved)}"
            )

        # 交叉编码器重排序
        rerank_scores: List[float] = []
        if unique_retrieved:
            rerank_query = state.get("current_sub_question") or state["original_query"]
            doc_texts = [d.content for d in unique_retrieved]
            reranked = await asyncio.to_thread(
                self.cross_encoder_ranker.reranker,
                rerank_query,
                doc_texts,
                threshold=self.config.reranker_threshold,
            )

            # 用重排序结果重建有序列表，保留来源信息
            text_to_source = {d.content: d.source for d in unique_retrieved}

            ordered_docs: List[RetrievedDoc] = []
            for doc_text, score in reranked:
                ordered_docs.append(
                    RetrievedDoc(
                        content=doc_text,
                        source=text_to_source.get(doc_text, "未知来源"),
                        score=score,
                    )
                )
                rerank_scores.append(score)

            # 兜底机制：如果重排序过滤掉了所有文档，保留原始排序的 top-1
            if not ordered_docs:
                all_scores = self.cross_encoder_ranker.model.predict(
                    [(rerank_query, d.content) for d in unique_retrieved]
                )
                best_idx = int(max(range(len(all_scores)), key=lambda i: all_scores[i]))
                best_doc = unique_retrieved[best_idx]
                ordered_docs.append(
                    RetrievedDoc(
                        content=best_doc.content,
                        source=best_doc.source,
                        score=float(all_scores[best_idx]),
                    )
                )
                rerank_scores.append(float(all_scores[best_idx]))
                logger.warning(
                    f"[RetrievalNode] 重排序兜底: all docs below threshold, keeping top-1, score={all_scores[best_idx]:.3f}"
                )

            # 构建带来源编号的 content，供 Prompt 引用 [1][2]
            numbered_parts = []
            for idx, doc in enumerate(ordered_docs, 1):
                numbered_parts.append(
                    f"[来源{idx}]（来自: {doc.source}）\n{doc.content}"
                )
            content = "\n\n".join(numbered_parts)
            retrieved_docs_text = [d.content for d in ordered_docs]
        else:
            content = ""
            retrieved_docs_text = []
            ordered_docs = []

        # 发送检索完成事件
        avg_score = sum(rerank_scores) / len(rerank_scores) if rerank_scores else 0
        writer(
            {
                "type": "retrieval_progress",
                "stage": "done",
                "message": f"检索完成，找到 {len(ordered_docs)} 条文档",
                "doc_count": len(ordered_docs),
                "avg_score": round(avg_score, 3),
            }
        )

        logger.info(
            f"[RetrievalNode] 融合检索完成: unique_docs={len(unique_retrieved)}, reranked_docs={len(ordered_docs)}, avg_score={avg_score:.3f}"
        )

        return {
            "search_content": content,
            "retrieved_documents": retrieved_docs_text,
            "retrieval_scores": rerank_scores,
        }

    async def __grade_documents(self, state, config: RunnableConfig, store: BaseStore):
        """检索结果评分节点：使用纯文本文档列表评分，避免来源编号干扰

        通过 grade_retry_count 防止无限循环：每个子问题最多重试 2 次，
        超过后强制通过评分，避免子问题场景下 run_count 不递增导致死循环。
        """
        question = state.get("current_sub_question") or state["original_query"]
        retrieved_docs = state.get("retrieved_documents", [])
        grade_retry_count = state.get("grade_retry_count", 0)

        if not retrieved_docs or grade_retry_count >= 2:
            logger.debug(
                f"[RetrievalNode] 跳过评分: docs_empty={not retrieved_docs}, retry_count={grade_retry_count}"
            )
            return "good"

        relevant_docs = await asyncio.to_thread(
            self.document_grader.grade, question, retrieved_docs
        )
        if relevant_docs:
            logger.info(
                f"[RetrievalNode] 文档评分通过: relevant={len(relevant_docs)}/{len(retrieved_docs)}"
            )
            return "good"
        else:
            logger.warning(
                f"[RetrievalNode] 文档评分未通过: all {len(retrieved_docs)} docs filtered"
            )
            return "bad"
