import os
from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Dict, Any

from langchain_text_splitters import RecursiveCharacterTextSplitter,CharacterTextSplitter
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


class ChunkHandler:
    def __init__(self):
        pass

    def recursive_chunk(self,contents:list[Document|str],chunk_size=1024,chunk_overlap=128)-> list[Document|str]:
        """
        极易破坏语义完整性：非常可能在句子中间、单词中间（如果按字符切）、代码行中间等不恰当的地方断开，导致上下文严重割裂。
        忽略文本结构：完全无视段落、标题、列表等任何文本固有结构。固定大小对于信息密度不同、语言不同的文本效果可能差异巨大。同样的 500 字符，在信息密集的文本中可能只包含半个观点，在稀疏文本中可能包含好几个。

        适用场景：
        对文本结构要求不高的简单场景。
        数据量极大，需要快速进行初步处理时。
        作为更复杂分块策略（如递归分块）的最后“兜底”手段。
        对上下文完整性要求不高的检索任务。
        """
        splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        if isinstance(contents[0], Document):
            return splitter.split_documents(contents)
        elif isinstance(contents[0], str):
            res = []
            for text in contents:
               res.extend(splitter.split_text(text))
            return res
        else:
            return []