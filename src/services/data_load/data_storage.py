from dataclasses import dataclass
from typing import Optional

from src.observability.logger import get_logger
from src.services.storage import MilvusExecutor, MilvusConfig, PostgreSQLConnector
from src.core.exceptions import IngestError, StorageError
from .chunk import ChunkHandler
from .file_tool import load_document

logger = get_logger(__name__)


@dataclass
class IngestConfig:
    """文档入库配置"""

    collection_name: str = "default"
    chunk_size: int = 1024
    chunk_overlap: int = 128
    use_parent_child: bool = False


class DataDBStorage:
    def __init__(self, collection_name: str = "default"):
        self.chunk_handler = ChunkHandler()
        self.collection_name = collection_name

    def _get_vector_store(
        self, collection_name: Optional[str] = None
    ) -> MilvusExecutor:
        """获取指定 collection 的 MilvusExecutor（单例池复用）"""
        target = collection_name or self.collection_name
        return MilvusExecutor(MilvusConfig(collection_name=target))

    async def load_data_and_chunk(
        self, data_path: str, chunk_size: int = 1024, chunk_overlap: int = 128
    ):
        """加载文档并分块"""
        import time
        start_time = time.time()
        
        logger.info(f"[DataStorage] 开始加载文档: path={data_path}")
        documents = load_document(data_path)
        load_elapsed = time.time() - start_time
        logger.info(
            f"[DataStorage] 文档加载完成: docs_count={len(documents)}, path={data_path}, elapsed={load_elapsed:.2f}s"
        )

        markdown_docs = [
            doc
            for doc in documents
            if doc.metadata.get("file_type") in ("md", "markdown")
        ]
        other_docs = [
            doc
            for doc in documents
            if doc.metadata.get("file_type") not in ("md", "markdown")
        ]

        logger.debug(
            f"[DataStorage] 文档类型分布: markdown={len(markdown_docs)}, other={len(other_docs)}"
        )

        chunks = []
        if markdown_docs:
            md_start = time.time()
            md_chunks = self.chunk_handler.markdown_chunk(
                markdown_docs, chunk_size=chunk_size, chunk_overlap=chunk_overlap
            )
            md_elapsed = time.time() - md_start
            logger.info(
                f"[DataStorage] Markdown分块完成: chunks_count={len(md_chunks)}, elapsed={md_elapsed:.2f}s"
            )
            chunks.extend(md_chunks)
        if other_docs:
            other_start = time.time()
            other_chunks = self.chunk_handler.recursive_chunk(
                other_docs, chunk_size=chunk_size, chunk_overlap=chunk_overlap
            )
            other_elapsed = time.time() - other_start
            logger.info(
                f"[DataStorage] 其他文档分块完成: chunks_count={len(other_chunks)}, elapsed={other_elapsed:.2f}s"
            )
            chunks.extend(other_chunks)

        total_elapsed = time.time() - start_time
        logger.info(
            f"[DataStorage] 总分块完成: total_chunks={len(chunks)}, total_elapsed={total_elapsed:.2f}s"
        )
        return chunks

    async def load_and_chunk_parent_child(
        self, data_path: str, chunk_size: int = 1024, chunk_overlap: int = 128
    ):
        """使用父子文档策略切分：子文档用于向量检索，父文档存入 PostgreSQL。

        Markdown 文件先按标题层级切分出父文档，再切分子文档；
        其他文件（PDF、DOCX）直接按大小切分父子文档。
        """
        import time
        start_time = time.time()
        
        logger.info(f"[DataStorage] 开始加载文档(父子模式): path={data_path}")
        documents = load_document(data_path)
        load_elapsed = time.time() - start_time
        logger.info(
            f"[DataStorage] 文档加载完成(父子模式): docs_count={len(documents)}, path={data_path}, elapsed={load_elapsed:.2f}s"
        )

        markdown_docs = [
            doc
            for doc in documents
            if doc.metadata.get("file_type") in ("md", "markdown")
        ]
        other_docs = [
            doc
            for doc in documents
            if doc.metadata.get("file_type") not in ("md", "markdown")
        ]

        logger.debug(
            f"[DataStorage] 父子模式文档类型分布: markdown={len(markdown_docs)}, other={len(other_docs)}"
        )

        parent_store: dict = {}
        child_docs: list = []

        if markdown_docs:
            md_start = time.time()
            md_parent_store, md_child_docs = (
                self.chunk_handler.markdown_parent_child_chunk(
                    markdown_docs,
                    chunk_size=chunk_size,
                    chunk_overlap=chunk_overlap,
                )
            )
            md_elapsed = time.time() - md_start
            parent_store.update(md_parent_store)
            child_docs.extend(md_child_docs)
            logger.info(
                f"[DataStorage] Markdown父子分块: parents={len(md_parent_store)}, children={len(md_child_docs)}, elapsed={md_elapsed:.2f}s"
            )

        if other_docs:
            other_start = time.time()
            other_parent_store, other_child_docs = (
                self.chunk_handler.parent_child_chunk(other_docs)
            )
            other_elapsed = time.time() - other_start
            parent_store.update(other_parent_store)
            child_docs.extend(other_child_docs)
            logger.info(
                f"[DataStorage] 其他文档父子分块: parents={len(other_parent_store)}, children={len(other_child_docs)}, elapsed={other_elapsed:.2f}s"
            )

        total_elapsed = time.time() - start_time
        logger.info(
            f"[DataStorage] 父子分块完成: total_parents={len(parent_store)}, total_children={len(child_docs)}, total_elapsed={total_elapsed:.2f}s"
        )
        return parent_store, child_docs

    async def ingest(self, config: IngestConfig, data_path: str) -> dict:
        """完整的文档入库流程：加载 → 分块 → 向量化 → 写入 Milvus

        Args:
            config: 入库配置（collection_name、chunk 参数等）
            data_path: 文件或目录路径

        Returns:
            入库结果摘要
        """
        import time
        start_time = time.time()
        
        logger.info(
            f"[DataStorage] 开始入库: collection={config.collection_name}, path={data_path}, use_parent_child={config.use_parent_child}, chunk_size={config.chunk_size}, chunk_overlap={config.chunk_overlap}"
        )
        self.collection_name = config.collection_name
        vector = self._get_vector_store()

        async def _batch_add_documents(vector_client, documents, batch_size=500, progress_callback=None):
            total = len(documents)
            if total == 0:
                return 0
            batches = (total + batch_size - 1) // batch_size
            inserted = 0
            logger.info(f"[DataStorage] 开始写入Milvus: total={total}, batches={batches}, batch_size={batch_size}")
            for i in range(batches):
                start = i * batch_size
                end = min(total, (i + 1) * batch_size)
                try:
                    await vector_client.aadd_documents(documents=documents[start:end])
                    inserted += end - start
                    progress = int((inserted / total) * 100)
                    logger.info(f"[DataStorage] 已写入 Milvus 批次: {i+1}/{batches}, cumulative={inserted}, progress={progress}%")
                    
                    # 调用进度回调（如果提供）
                    if progress_callback:
                        try:
                            progress_callback(inserted, total, progress)
                        except Exception as cb_err:
                            logger.warning(f"[DataStorage] 进度回调失败: {cb_err}")
                            
                except Exception as exc:
                    logger.error(f"[DataStorage] 写入 Milvus 失败 at batch {i+1}/{batches}: {exc}")
                    raise StorageError(f"写入 Milvus 失败", exc)
            return inserted

        if config.use_parent_child:
            parent_store, child_docs = await self.load_and_chunk_parent_child(
                data_path,
                chunk_size=config.chunk_size,
                chunk_overlap=config.chunk_overlap,
            )
            if parent_store:
                pg_start = time.time()
                PostgreSQLConnector().batch_insert_parent_documents(parent_store)
                pg_elapsed = time.time() - pg_start
                logger.info(
                    f"[DataStorage] 父文档写入PostgreSQL: count={len(parent_store)}, elapsed={pg_elapsed:.2f}s"
                )
            if child_docs:
                try:
                    milvus_start = time.time()
                    
                    # 定义进度回调函数，更新 ingest_status_store
                    def update_progress(inserted, total, progress_percent):
                        from server import ingest_status_store
                        try:
                            ingest_status_store[config.collection_name] = {
                                "status": "processing",
                                "message": f"正在写入向量数据库... {progress_percent}% ({inserted}/{total})",
                                "progress": progress_percent,
                                "inserted": inserted,
                                "total": total,
                                "result": None,
                            }
                        except Exception as store_err:
                            logger.warning(f"[DataStorage] 更新进度状态失败: {store_err}")
                    
                    inserted = await _batch_add_documents(
                        vector.client, 
                        child_docs, 
                        progress_callback=update_progress
                    )
                    
                    # Flush 确保数据持久化并立即可查询
                    flush_start = time.time()
                    try:
                        # langchain_milvus.Milvus 对象可能有 collection 属性或方法
                        if hasattr(vector.client, 'col'):
                            # 某些版本的 langchain_milvus 使用 col 属性
                            vector.client.col.flush()
                        elif hasattr(vector.client, '_collection'):
                            vector.client._collection.flush()
                        else:
                            # 尝试通过 pymilvus 直接连接并 flush
                            from pymilvus import connections, Collection
                            from config import MilvusConfig
                            milvus_config = MilvusConfig()
                            uri = f"http://{milvus_config.host}:{milvus_config.port}"
                            conn_alias = f"flush_{config.collection_name}"
                            try:
                                connections.connect(
                                    alias=conn_alias,
                                    uri=uri,
                                    db_name=milvus_config.db_name,
                                    token=milvus_config.token,
                                )
                                coll = Collection(config.collection_name, using=conn_alias)
                                coll.flush()
                                connections.disconnect(conn_alias)
                            except Exception as conn_err:
                                logger.warning(f"[DataStorage] 备用 flush 方案失败: {conn_err}")
                        
                        flush_elapsed = time.time() - flush_start
                        logger.info(f"[DataStorage] Milvus flush完成: collection={config.collection_name}, elapsed={flush_elapsed:.2f}s")
                    except Exception as flush_err:
                        logger.warning(f"[DataStorage] Milvus flush 失败（不影响入库）: {flush_err}")
                    
                    milvus_elapsed = time.time() - milvus_start
                    logger.info(
                        f"[DataStorage] 子文档写入Milvus: count={inserted}, elapsed={milvus_elapsed:.2f}s"
                    )
                except StorageError as se:
                    raise IngestError(config.collection_name, stage="vector_write", cause=se)

            total_elapsed = time.time() - start_time
            logger.info(
                f"[DataStorage] 入库完成(父子模式): collection={config.collection_name}, parents={len(parent_store)}, children={len(child_docs)}, total_elapsed={total_elapsed:.2f}s"
            )
            return {
                "collection_name": config.collection_name,
                "parent_chunks": len(parent_store),
                "child_chunks": len(child_docs),
                "total_chunks": len(child_docs),
            }
        else:
            chunks = await self.load_data_and_chunk(
                data_path,
                chunk_size=config.chunk_size,
                chunk_overlap=config.chunk_overlap,
            )
            if chunks:
                try:
                    milvus_start = time.time()
                    
                    # 定义进度回调函数，更新 ingest_status_store
                    def update_progress(inserted, total, progress_percent):
                        from server import ingest_status_store
                        try:
                            ingest_status_store[config.collection_name] = {
                                "status": "processing",
                                "message": f"正在写入向量数据库... {progress_percent}% ({inserted}/{total})",
                                "progress": progress_percent,
                                "inserted": inserted,
                                "total": total,
                                "result": None,
                            }
                        except Exception as store_err:
                            logger.warning(f"[DataStorage] 更新进度状态失败: {store_err}")
                    
                    inserted = await _batch_add_documents(
                        vector.client, 
                        chunks, 
                        progress_callback=update_progress
                    )
                    
                    # Flush 确保数据持久化并立即可查询
                    flush_start = time.time()
                    try:
                        # langchain_milvus.Milvus 对象可能有 collection 属性或方法
                        if hasattr(vector.client, 'col'):
                            # 某些版本的 langchain_milvus 使用 col 属性
                            vector.client.col.flush()
                        elif hasattr(vector.client, '_collection'):
                            vector.client._collection.flush()
                        else:
                            # 尝试通过 pymilvus 直接连接并 flush
                            from pymilvus import connections, Collection
                            from config import MilvusConfig
                            milvus_config = MilvusConfig()
                            uri = f"http://{milvus_config.host}:{milvus_config.port}"
                            conn_alias = f"flush_{config.collection_name}"
                            try:
                                connections.connect(
                                    alias=conn_alias,
                                    uri=uri,
                                    db_name=milvus_config.db_name,
                                    token=milvus_config.token,
                                )
                                coll = Collection(config.collection_name, using=conn_alias)
                                coll.flush()
                                connections.disconnect(conn_alias)
                            except Exception as conn_err:
                                logger.warning(f"[DataStorage] 备用 flush 方案失败: {conn_err}")
                        
                        flush_elapsed = time.time() - flush_start
                        logger.info(f"[DataStorage] Milvus flush完成: collection={config.collection_name}, elapsed={flush_elapsed:.2f}s")
                    except Exception as flush_err:
                        logger.warning(f"[DataStorage] Milvus flush 失败（不影响入库）: {flush_err}")
                    
                    milvus_elapsed = time.time() - milvus_start
                    logger.info(
                        f"[DataStorage] 文档写入Milvus: count={inserted}, elapsed={milvus_elapsed:.2f}s"
                    )
                except StorageError as se:
                    raise IngestError(config.collection_name, stage="vector_write", cause=se)

            total_elapsed = time.time() - start_time
            logger.info(
                f"[DataStorage] 入库完成(普通模式): collection={config.collection_name}, chunks={len(chunks)}, total_elapsed={total_elapsed:.2f}s"
            )
            return {
                "collection_name": config.collection_name,
                "total_chunks": len(chunks),
            }

    async def _generate_image_caption(self, image_bytes: bytes) -> str:
        """调用 VLM 为图片生成文字描述（Caption），用于增强图片的文字检索召回率。

        通过 MultimodalConfig.caption_model_name 配置 VLM 模型，为空则跳过 Caption 生成。
        Caption 存入 Milvus 图片 Collection 的 caption 字段，可参与文字检索。

        Args:
            image_bytes: 图片原始字节

        Returns:
            图片描述文字，生成失败或未配置模型时返回空字符串
        """
        try:
            from config import MultimodalConfig

            caption_model_name = MultimodalConfig().caption_model_name
            if not caption_model_name:
                return ""

            import base64
            from langchain_openai import ChatOpenAI
            from langchain_core.messages import HumanMessage

            image_b64 = base64.b64encode(image_bytes).decode("utf-8")
            image_format = "png" if image_b64.startswith("iVBOR") else "jpeg"

            vlm = ChatOpenAI(model=caption_model_name)
            message = HumanMessage(
                content=[
                    {
                        "type": "text",
                        "text": "请用一句话简洁描述这张图片的主要内容，不超过100字。",
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/{image_format};base64,{image_b64}"
                        },
                    },
                ]
            )
            response = await vlm.ainvoke([message])
            caption = response.content if hasattr(response, "content") else ""
            logger.debug(f"[DataStorage] VLM Caption 生成完成: caption={caption[:50]}")
            return caption
        except Exception as exc:
            logger.debug(f"[DataStorage] VLM Caption 生成失败，跳过: {exc}")
            return ""

    async def _ingest_tables_from_path(
        self, data_path: str, collection_name: str
    ) -> int:
        """从 PDF 中提取表格并生成摘要 chunk 写入文字 Collection，增强表格内容的文字检索精度。

        表格转 Markdown 后，同时生成一份"逐行摘要"（第N行第M列的值是X），
        两种格式都写入文字 Collection，提升表格内容的检索召回率。

        Args:
            data_path: 文件或目录路径
            collection_name: 文字 Collection 名称

        Returns:
            成功写入的表格 chunk 数量
        """
        from pathlib import Path

        try:
            from src.services.data_load.parser import PaddleOCRParser
            from langchain_core.documents import Document

            parser = PaddleOCRParser()
            vector = self._get_vector_store(collection_name)

            target_path = Path(data_path)
            pdf_files = (
                list(target_path.rglob("*.pdf"))
                if target_path.is_dir()
                else [target_path] if target_path.suffix.lower() == ".pdf" else []
            )
            if not pdf_files:
                return 0

            table_docs: list = []
            for pdf_path in pdf_files:
                try:
                    content_list = parser.parse_pdf(
                        pdf_path,
                        extract_images=False,
                        extract_tables=True,
                    )
                    for item in content_list:
                        if item.get("type") != "table":
                            continue
                        markdown_text = item.get("text", "")
                        if not markdown_text:
                            continue
                        source = str(pdf_path)
                        page_idx = item.get("page_idx", 0)

                        # Markdown 格式 chunk：保留原始表格结构
                        table_docs.append(
                            Document(
                                page_content=markdown_text,
                                metadata={
                                    "source": source,
                                    "page": page_idx,
                                    "content_type": "table_markdown",
                                },
                            )
                        )

                        # 逐行摘要 chunk：将表格每行转为自然语言描述，提升检索精度
                        row_summary = self._table_markdown_to_row_summary(
                            markdown_text, source, page_idx
                        )
                        if row_summary:
                            table_docs.append(
                                Document(
                                    page_content=row_summary,
                                    metadata={
                                        "source": source,
                                        "page": page_idx,
                                        "content_type": "table_summary",
                                    },
                                )
                            )
                except Exception as pdf_error:
                    logger.warning(
                        f"[DataStorage] 表格提取失败，跳过: path={pdf_path}, error={pdf_error}"
                    )

            if not table_docs:
                return 0

            await vector.client.aadd_documents(documents=table_docs)
            logger.info(
                f"[DataStorage] 表格 chunk 写入完成: collection={collection_name}, count={len(table_docs)}"
            )
            return len(table_docs)
        except Exception as exc:
            logger.warning(f"[DataStorage] 表格入库失败，跳过: {exc}")
            return 0

    @staticmethod
    def _table_markdown_to_row_summary(
        markdown_text: str, source: str, page_idx: int
    ) -> str:
        """将 Markdown 表格转为逐行自然语言摘要，提升表格内容的文字检索精度。

        例如：
            | 姓名 | 年龄 |
            |---|---|
            | 张三 | 25 |
        转为：
            表格内容（来源：xxx.pdf 第1页）
            第1行：姓名=张三，年龄=25

        Args:
            markdown_text: Markdown 格式的表格文本
            source: 来源文件路径
            page_idx: 来源页码（从 0 开始）

        Returns:
            逐行摘要字符串，无有效数据时返回空字符串
        """
        lines = [
            line.strip() for line in markdown_text.strip().splitlines() if line.strip()
        ]
        # 过滤分隔行（如 |---|---|）
        data_lines = [line for line in lines if not all(c in "|-: " for c in line)]
        if len(data_lines) < 2:
            return ""

        headers = [cell.strip() for cell in data_lines[0].strip("|").split("|")]
        summary_parts = []
        for row_idx, row_line in enumerate(data_lines[1:], 1):
            cells = [cell.strip() for cell in row_line.strip("|").split("|")]
            cell_descriptions = [
                f"{header}={cell}"
                for header, cell in zip(headers, cells)
                if header and cell
            ]
            if cell_descriptions:
                summary_parts.append(f"第{row_idx}行：{'，'.join(cell_descriptions)}")

        if not summary_parts:
            return ""
        return f"表格内容（来源：{source} 第{page_idx + 1}页）\n" + "\n".join(
            summary_parts
        )

    async def _ingest_images_from_path(
        self, data_path: str, collection_name: str
    ) -> int:
        """从文件路径中提取图片并写入 Milvus 图片 Collection。

        使用 PaddleOCRParser 解析 PDF 中的嵌入图片，调用 VLM 生成 Caption（可选），
        通过 MilvusImageClient 将 CLIP 向量写入图片专用 Collection。
        同时并行调用 _ingest_tables_from_path 将表格摘要写入文字 Collection。

        Args:
            data_path: 文件或目录路径
            collection_name: 对应的文本 Collection 名称（图片 Collection 自动加后缀）

        Returns:
            成功入库的图片数量
        """
        import hashlib
        import asyncio as _asyncio
        from pathlib import Path

        # 表格摘要与图片入库并行执行，互不阻塞
        table_task = _asyncio.create_task(
            self._ingest_tables_from_path(data_path, collection_name)
        )

        try:
            from src.services.data_load.parser import PaddleOCRParser
            from src.services.storage.milvus_image_client import MilvusImageClient
            from config import MilvusConfig

            parser = PaddleOCRParser()
            image_client = MilvusImageClient(
                MilvusConfig(collection_name=collection_name)
            )

            target_path = Path(data_path)
            pdf_files = (
                list(target_path.rglob("*.pdf"))
                if target_path.is_dir()
                else [target_path] if target_path.suffix.lower() == ".pdf" else []
            )

            if not pdf_files:
                await table_task
                return 0

            all_image_records = []
            for pdf_path in pdf_files:
                try:
                    content_list = parser.parse_pdf(
                        pdf_path,
                        extract_images=True,
                        extract_tables=False,  # 表格由 _ingest_tables_from_path 单独处理
                    )
                    for item in content_list:
                        if item.get("type") != "image":
                            continue
                        image_bytes = item.get("data")
                        if not image_bytes:
                            continue
                        # 用文件路径 + 图片名生成唯一 ID
                        unique_id = hashlib.md5(
                            f"{pdf_path}:{item.get('name', '')}:{item.get('page_idx', 0)}".encode()
                        ).hexdigest()
                        # 调用 VLM 生成 Caption（配置了 CAPTION_MODEL_NAME 时才生效，否则返回空字符串）
                        caption = await self._generate_image_caption(image_bytes)
                        all_image_records.append(
                            {
                                "image_id": unique_id,
                                "image_bytes": image_bytes,
                                "caption": caption,
                                "source": str(pdf_path),
                                "page_idx": item.get("page_idx", 0),
                            }
                        )
                except Exception as pdf_error:
                    logger.warning(
                        f"[DataStorage] 图片提取失败，跳过: path={pdf_path}, error={pdf_error}"
                    )

            if not all_image_records:
                await table_task
                return 0

            # CLIP 推理为 CPU 密集型，通过 insert_images_async 在线程池中执行
            inserted = await image_client.insert_images_async(all_image_records)
            logger.info(
                f"[DataStorage] 图片向量写入完成: collection={collection_name}, count={inserted}"
            )
            await table_task
            return inserted

        except Exception as exc:
            logger.warning(f"[DataStorage] 图片入库失败，跳过: {exc}")
            try:
                await table_task
            except Exception:
                pass
            return 0
