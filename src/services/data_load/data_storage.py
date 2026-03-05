from src.monitoring.logger import monitor_task_status
from src.services.llm.models import get_embedding_model
from src.services.storage import MilvusExecutor, MilvusConfig, PostgreSQLConnector

from .file_tool import load_document
from .chunk import ChunkHandler


class DataDBStorage:
    def __init__(self):
        self.chunk_handler = ChunkHandler()
        self.vector = MilvusExecutor(MilvusConfig(collection_name="hybridRag_news"))
        self.embedding = get_embedding_model("qwen")

    async def load_data_and_chunk(self, data_path):
        documents = load_document(data_path)
        monitor_task_status("load documents numbers", len(documents))

        markdown_docs = [doc for doc in documents if doc.metadata.get("file_type") in ("md", "markdown")]
        other_docs = [doc for doc in documents if doc.metadata.get("file_type") not in ("md", "markdown")]

        chunks = []
        if markdown_docs:
            md_chunks = self.chunk_handler.markdown_chunk(markdown_docs)
            monitor_task_status("markdown chunks numbers", len(md_chunks))
            chunks.extend(md_chunks)
        if other_docs:
            other_chunks = self.chunk_handler.recursive_chunk(other_docs)
            monitor_task_status("recursive chunks numbers", len(other_chunks))
            chunks.extend(other_chunks)

        monitor_task_status("total chunks numbers", len(chunks))
        return chunks

    async def load_and_chunk_parent_child(self, data_path: str):
        """使用父子文档策略切分：子文档用于向量检索，父文档存入 PostgreSQL。

        Markdown 文件先按标题层级切分出父文档，再切分子文档；
        其他文件（PDF、DOCX）直接按大小切分父子文档。
        """
        documents = load_document(data_path)
        monitor_task_status("load documents numbers", len(documents))

        markdown_docs = [doc for doc in documents if doc.metadata.get("file_type") in ("md", "markdown")]
        other_docs = [doc for doc in documents if doc.metadata.get("file_type") not in ("md", "markdown")]

        parent_store: dict = {}
        child_docs: list = []

        if markdown_docs:
            md_parent_store, md_child_docs = self.chunk_handler.markdown_parent_child_chunk(markdown_docs)
            parent_store.update(md_parent_store)
            child_docs.extend(md_child_docs)
            monitor_task_status("markdown parent documents", len(md_parent_store))
            monitor_task_status("markdown child documents", len(md_child_docs))

        if other_docs:
            other_parent_store, other_child_docs = self.chunk_handler.parent_child_chunk(other_docs)
            parent_store.update(other_parent_store)
            child_docs.extend(other_child_docs)
            monitor_task_status("other parent documents", len(other_parent_store))
            monitor_task_status("other child documents", len(other_child_docs))

        monitor_task_status("total parent documents", len(parent_store))
        monitor_task_status("total child documents", len(child_docs))
        return parent_store, child_docs

    async def save_to_vector(self, data_path: str, use_parent_child: bool = False):
        if use_parent_child:
            parent_store, child_docs = await self.load_and_chunk_parent_child(data_path)
            PostgreSQLConnector().batch_insert_parent_documents(parent_store)
            monitor_task_status("parent documents saved to PostgreSQL", len(parent_store))
            await self.vector.client.aadd_documents(documents=child_docs)
            monitor_task_status("child documents saved to Milvus", len(child_docs))
        else:
            documents = await self.load_data_and_chunk(data_path)
            await self.vector.client.aadd_documents(documents=documents)
