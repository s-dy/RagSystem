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
