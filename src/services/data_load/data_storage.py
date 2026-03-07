from dataclasses import dataclass
from typing import Optional

from src.observability.logger import get_logger
from src.services.storage import MilvusExecutor, MilvusConfig, PostgreSQLConnector
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
        documents = load_document(data_path)
        logger.info(
            f"[DataStorage] 文档加载完成: docs_count={len(documents)}, path={data_path}"
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

        chunks = []
        if markdown_docs:
            md_chunks = self.chunk_handler.markdown_chunk(
                markdown_docs, chunk_size=chunk_size, chunk_overlap=chunk_overlap
            )
            logger.debug(
                f"[DataStorage] Markdown分块完成: chunks_count={len(md_chunks)}"
            )
            chunks.extend(md_chunks)
        if other_docs:
            other_chunks = self.chunk_handler.recursive_chunk(
                other_docs, chunk_size=chunk_size, chunk_overlap=chunk_overlap
            )
            logger.debug(
                f"[DataStorage] 其他文档分块完成: chunks_count={len(other_chunks)}"
            )
            chunks.extend(other_chunks)

        logger.info(f"[DataStorage] 总分块完成: total_chunks={len(chunks)}")
        return chunks

    async def load_and_chunk_parent_child(
        self, data_path: str, chunk_size: int = 1024, chunk_overlap: int = 128
    ):
        """使用父子文档策略切分：子文档用于向量检索，父文档存入 PostgreSQL。

        Markdown 文件先按标题层级切分出父文档，再切分子文档；
        其他文件（PDF、DOCX）直接按大小切分父子文档。
        """
        documents = load_document(data_path)
        logger.info(
            f"[DataStorage] 文档加载完成(父子模式): docs_count={len(documents)}, path={data_path}"
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

        parent_store: dict = {}
        child_docs: list = []

        if markdown_docs:
            md_parent_store, md_child_docs = (
                self.chunk_handler.markdown_parent_child_chunk(
                    markdown_docs,
                    chunk_size=chunk_size,
                    chunk_overlap=chunk_overlap,
                )
            )
            parent_store.update(md_parent_store)
            child_docs.extend(md_child_docs)
            logger.debug(
                f"[DataStorage] Markdown父子分块: parents={len(md_parent_store)}, children={len(md_child_docs)}"
            )

        if other_docs:
            other_parent_store, other_child_docs = (
                self.chunk_handler.parent_child_chunk(other_docs)
            )
            parent_store.update(other_parent_store)
            child_docs.extend(other_child_docs)
            logger.debug(
                f"[DataStorage] 其他文档父子分块: parents={len(other_parent_store)}, children={len(other_child_docs)}"
            )

        logger.info(
            f"[DataStorage] 父子分块完成: total_parents={len(parent_store)}, total_children={len(child_docs)}"
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
        logger.info(
            f"[DataStorage] 开始入库: collection={config.collection_name}, path={data_path}"
        )
        self.collection_name = config.collection_name
        vector = self._get_vector_store()

        if config.use_parent_child:
            parent_store, child_docs = await self.load_and_chunk_parent_child(
                data_path,
                chunk_size=config.chunk_size,
                chunk_overlap=config.chunk_overlap,
            )
            if parent_store:
                PostgreSQLConnector().batch_insert_parent_documents(parent_store)
                logger.info(
                    f"[DataStorage] 父文档写入PostgreSQL: count={len(parent_store)}"
                )
            if child_docs:
                await vector.client.aadd_documents(documents=child_docs)
                logger.info(f"[DataStorage] 子文档写入Milvus: count={len(child_docs)}")
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
                await vector.client.aadd_documents(documents=chunks)
                logger.info(f"[DataStorage] 文档写入Milvus: count={len(chunks)}")
            return {
                "collection_name": config.collection_name,
                "total_chunks": len(chunks),
            }
