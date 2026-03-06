"""检索节点 Mixin

包含：
- _retrieve_internal: 内部向量检索
- _retrieve_external: 外部搜索（MCP/Bing）
- _fusion_retrieve: 融合检索 + 重排序
- _grade_documents: 检索结果评分
"""

import asyncio
import re
from typing import List

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableConfig
from langgraph.store.base import BaseStore

from .fusion_retrieve import FusionRetrieve, RetrievedDoc
from src.services.task_analyzer import TaskCharacteristics, TaskType
from src.core.exceptions import InternalRetrievalError, ExternalSearchError
from src.observability.logger import monitor_task_status


class RetrievalNodeMixin:
    """检索节点方法集合，通过 Mixin 注入到 Graph 类"""

    async def __retrieve_internal(self, router_index) -> List[RetrievedDoc]:
        """处理内部检索，返回 RetrievedDoc 列表"""
        doc_result: List[RetrievedDoc] = []
        if not router_index:
            return doc_result
        monitor_task_status('开始处理内部检索')
        search_model = FusionRetrieve(use_parent_child=self.config.enable_parent_child_retrieval)
        for collection_index, queries in router_index.items():
            try:
                result = await search_model.search_queries(queries, collection_name=collection_index)
                if not result:
                    continue
                for doc_list in result:
                    if doc_list:
                        doc_result.extend(doc_list)
            except ConnectionError as conn_err:
                raise InternalRetrievalError(collection_index, cause=conn_err)
            except TimeoutError as timeout_err:
                monitor_task_status(
                    f"检索超时 [collection={collection_index}]: {timeout_err}", level="WARNING"
                )
            except Exception as exc:
                monitor_task_status(
                    f"检索出错 [collection={collection_index}]: {exc}", level="WARNING"
                )
        monitor_task_status('内部检索完成')
        return doc_result

    async def __retrieve_external(self, query: str, max_retries: int = 2) -> list:
        """处理外部检索，支持重试"""
        external_docs = []
        if not query:
            return external_docs

        for attempt in range(max_retries + 1):
            try:
                monitor_task_status(f'开始调用外部工具 (attempt {attempt + 1}/{max_retries + 1})')

                prompt = ChatPromptTemplate.from_template("""
                你是一个优化在搜索引擎中搜索查询的助手，需要重写用户问题，便于使用搜索引擎搜索。

                以下是需要重写的用户问题：
                【{question}】

                要求：
                1. 只返回最终的答案，不用返回其他无关内容。
                """)
                chain = prompt | self.llm | StrOutputParser()
                search_query = await chain.ainvoke({'question': query})
                monitor_task_status('rewrite search query', search_query)

                search_result = self.tools_pool.get_response(
                    await self.tools_pool.call_tool('bing_search', {'query': search_query})
                )
                uids, uid_map = [], {}
                if len(search_result):
                    results = search_result[0].get('results')
                else:
                    results = []
                for item in results:
                    uids.append(item['uuid'])
                    uid_map[item['uuid']] = item['url']

                pages_results = self.tools_pool.get_response(
                    await self.tools_pool.call_tool('crawl_webpage', {'uuids': uids, 'url_map': uid_map})
                )
                if pages_results:
                    pages_results = pages_results[0]
                for item in pages_results:
                    if item.get('content'):
                        external_docs.append(item['content'])

                monitor_task_status('外部工具调用完成', external_docs)
                return external_docs

            except ConnectionError as conn_err:
                monitor_task_status(
                    f"外部搜索连接失败 (attempt {attempt + 1}): {conn_err}", level='WARNING'
                )
                if attempt == max_retries:
                    raise ExternalSearchError(
                        "外部搜索连接失败", attempt=attempt + 1,
                        max_retries=max_retries + 1, cause=conn_err,
                    )
                await asyncio.sleep(1 * (attempt + 1))
            except Exception as exc:
                monitor_task_status(
                    f"外部检索失败 (attempt {attempt + 1}): {exc}", level='WARNING'
                )
                if attempt == max_retries:
                    monitor_task_status("外部检索已达最大重试次数，返回空结果", level='WARNING')
                    return external_docs
                await asyncio.sleep(1 * (attempt + 1))

        return external_docs

    async def __fusion_retrieve(self, state, config: RunnableConfig, store: BaseStore) -> dict:
        """融合检索节点：支持来源编号和置信度分数"""
        monitor_task_status("---FUSION RETRIEVAL---")
        task_characteristics = state.get('task_characteristics') or TaskCharacteristics(
            task_type=TaskType.FACT_RETRIEVAL, requires_external_tools=False
        )

        # 内部检索（返回 RetrievedDoc 列表）
        internal_docs = await self.__retrieve_internal(state['router_index'])

        # 外部检索（仅在前端开启联网搜索时才调用）
        query = state.get("current_sub_question") or state["original_query"]
        enable_web_search = config.get("configurable", {}).get("enable_web_search", False)
        external_retrieved_docs: List[RetrievedDoc] = []
        if enable_web_search and (not internal_docs or task_characteristics.requires_external_tools):
            external_texts = await self.__retrieve_external(query)
            for text in external_texts:
                external_retrieved_docs.append(RetrievedDoc(content=text, source="外部搜索", score=0.5))

        # 融合 + 去重
        all_retrieved = internal_docs + external_retrieved_docs
        seen = set()
        unique_retrieved: List[RetrievedDoc] = []
        for retrieved_doc in all_retrieved:
            normalized = re.sub(r'[^\w\s]', '', retrieved_doc.content.lower().strip())
            if normalized and normalized not in seen:
                seen.add(normalized)
                unique_retrieved.append(retrieved_doc)

        # 交叉编码器重排序
        rerank_scores: List[float] = []
        if unique_retrieved:
            rerank_query = state.get("current_sub_question") or state["original_query"]
            doc_texts = [d.content for d in unique_retrieved]
            reranked = await asyncio.to_thread(
                self.cross_encoder_ranker.reranker, rerank_query, doc_texts,
                threshold=self.config.reranker_threshold,
            )

            # 用重排序结果重建有序列表，保留来源信息
            text_to_source = {d.content: d.source for d in unique_retrieved}

            ordered_docs: List[RetrievedDoc] = []
            for doc_text, score in reranked:
                ordered_docs.append(RetrievedDoc(
                    content=doc_text,
                    source=text_to_source.get(doc_text, "未知来源"),
                    score=score,
                ))
                rerank_scores.append(score)

            # 构建带来源编号的 content，供 Prompt 引用 [1][2]
            numbered_parts = []
            for idx, doc in enumerate(ordered_docs, 1):
                numbered_parts.append(f"[来源{idx}]（来自: {doc.source}）\n{doc.content}")
            content = "\n\n".join(numbered_parts)
            retrieved_docs_text = [d.content for d in ordered_docs]
        else:
            content = ""
            retrieved_docs_text = []
            ordered_docs = []

        monitor_task_status("final_retrieval", {
            "unique_docs_count": len(unique_retrieved),
            "reranked_count": len(ordered_docs),
        })

        return {
            'search_content': content,
            'retrieved_documents': retrieved_docs_text,
            'retrieval_scores': rerank_scores,
        }

    async def __grade_documents(self, state, config: RunnableConfig, store: BaseStore):
        """检索结果评分节点：使用纯文本文档列表评分，避免来源编号干扰

        通过 grade_retry_count 防止无限循环：每个子问题最多重试 2 次，
        超过后强制通过评分，避免子问题场景下 run_count 不递增导致死循环。
        """
        question = state.get("current_sub_question") or state["original_query"]
        retrieved_docs = state.get('retrieved_documents', [])
        grade_retry_count = state.get('grade_retry_count', 0)

        if not retrieved_docs or grade_retry_count >= 2:
            return 'good'

        grade = await asyncio.to_thread(self.document_grader.grade, question, retrieved_docs)
        if grade:
            return 'good'
        else:
            return 'bad'
