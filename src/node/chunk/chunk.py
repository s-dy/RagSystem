from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Dict, Any

from langchain_text_splitters import RecursiveCharacterTextSplitter, MarkdownHeaderTextSplitter
from langchain_core.documents import Document


@dataclass
class DocumentChunk:
    """文档分块"""
    id: str
    content: str
    embedding: List[float]
    metadata: Dict[str, Any]
    document_id: str
    chunk_index: int
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    tags: List[str] = field(default_factory=list)


# 中文优先的分隔符列表，按优先级从高到低排列
CHINESE_SEPARATORS = [
    "\n\n",
    "\n",
    "。",
    "！", "？",
    "；",
    "，",
    " ",
    "",
]

# Markdown 标题层级切分配置
MARKDOWN_HEADERS_TO_SPLIT = [
    ("#", "h1"),
    ("##", "h2"),
    ("###", "h3"),
    ("####", "h4"),
]


class ChunkHandler:
    def __init__(self):
        self._markdown_splitter = MarkdownHeaderTextSplitter(
            headers_to_split_on=MARKDOWN_HEADERS_TO_SPLIT,
            strip_headers=False,
        )

    def recursive_chunk(self, contents: list[Document | str], chunk_size=1024, chunk_overlap=128) -> list[Document | str]:
        """使用中文优化分隔符的递归切分，适用于 PDF、DOCX 等非结构化文档。"""
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=CHINESE_SEPARATORS,
        )
        if len(contents) == 0:
            return []
        doc_contents = []
        for content in contents:
            if not isinstance(content, Document):
                doc_contents.append(Document(page_content=content))

        return splitter.split_documents(contents)

    def parent_child_chunk(
        self,
        contents: list[Document],
        parent_size=1500,
        parent_overlap=200,
        child_size=400,
        child_overlap=64,
    ) -> tuple[dict[str, Document], list[Document]]:
        """父子文档切分：大 chunk 作为上下文，小 chunk 用于向量检索。

        Returns:
            (parent_store, child_docs)
            - parent_store: {parent_id: Document} 父文档映射
            - child_docs: 子文档列表，metadata 中包含 parent_id
        """
        parent_splitter = RecursiveCharacterTextSplitter(
            chunk_size=parent_size,
            chunk_overlap=parent_overlap,
            separators=CHINESE_SEPARATORS,
        )
        child_splitter = RecursiveCharacterTextSplitter(
            chunk_size=child_size,
            chunk_overlap=child_overlap,
            separators=CHINESE_SEPARATORS,
        )

        parent_docs = parent_splitter.split_documents(contents)
        parent_store: dict[str, Document] = {}
        child_docs: list[Document] = []

        for idx, parent_doc in enumerate(parent_docs):
            source = parent_doc.metadata.get("source", "unknown")
            parent_id = f"{source}_{idx}"
            parent_doc.metadata["parent_id"] = parent_id
            parent_store[parent_id] = parent_doc

            children = child_splitter.split_text(parent_doc.page_content)
            for child_text in children:
                child_doc = Document(
                    page_content=child_text,
                    metadata={**parent_doc.metadata, "parent_id": parent_id},
                )
                child_docs.append(child_doc)

        return parent_store, child_docs

    def markdown_parent_child_chunk(
        self,
        contents: list[Document],
        child_size=400,
        child_overlap=64,
        chunk_size=1024,
        chunk_overlap=128,
    ) -> tuple[dict[str, Document], list[Document]]:
        """Markdown 父子文档切分：先按标题层级切分出父文档，再对每个父文档切分出子文档。

        Returns:
            (parent_store, child_docs)
            - parent_store: {parent_id: Document} 父文档映射
            - child_docs: 子文档列表，metadata 中包含 parent_id
        """
        parent_docs = self.markdown_chunk(contents, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        child_splitter = RecursiveCharacterTextSplitter(
            chunk_size=child_size,
            chunk_overlap=child_overlap,
            separators=CHINESE_SEPARATORS,
        )

        parent_store: dict[str, Document] = {}
        child_docs: list[Document] = []

        for idx, parent_doc in enumerate(parent_docs):
            source = parent_doc.metadata.get("source", "unknown")
            parent_id = f"{source}_md_{idx}"
            parent_doc.metadata["parent_id"] = parent_id
            parent_store[parent_id] = parent_doc

            if len(parent_doc.page_content) <= child_size:
                child_doc = Document(
                    page_content=parent_doc.page_content,
                    metadata={**parent_doc.metadata, "parent_id": parent_id},
                )
                child_docs.append(child_doc)
            else:
                children = child_splitter.split_text(parent_doc.page_content)
                for child_text in children:
                    child_doc = Document(
                        page_content=child_text,
                        metadata={**parent_doc.metadata, "parent_id": parent_id},
                    )
                    child_docs.append(child_doc)

        return parent_store, child_docs

    def markdown_chunk(self, contents: list[Document], chunk_size=1024, chunk_overlap=128) -> list[Document]:
        """Markdown 结构化切分：先按标题层级切分，超长章节再用递归切分兜底。"""
        secondary_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=CHINESE_SEPARATORS,
        )
        result = []
        for doc in contents:
            original_metadata = doc.metadata
            header_chunks = self._markdown_splitter.split_text(doc.page_content)

            for header_chunk in header_chunks:
                merged_metadata = {**original_metadata, **header_chunk.metadata}
                if len(header_chunk.page_content) > chunk_size:
                    sub_chunks = secondary_splitter.split_text(header_chunk.page_content)
                    for sub_chunk in sub_chunks:
                        result.append(Document(page_content=sub_chunk, metadata=merged_metadata))
                else:
                    result.append(Document(page_content=header_chunk.page_content, metadata=merged_metadata))
        return result
