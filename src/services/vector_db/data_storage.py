from typing import Callable

from src.core.shared.chunk import ChunkHandler
from src.monitoring.logger import monitor_task_status
from src.services.llm.models import get_embedding_model
from src.services.vector_db.client import MilvusExecutor, MilvusConfig
from utils.file_tool import load_document
from utils.async_task import async_run


def news_chunk_adapter(chunks):...

def short_chunk_adapter(chunks):
    result = []
    for chunk in chunks:
        if len(chunk.page_content) < 48:
            monitor_task_status('chunk too short ==> len',f'{len(chunk.page_content)} , content : {chunk.page_content}')
            continue
        result.append(chunk)
    return result

class DataDBStorage:
    def __init__(self,adapters:list[Callable] = None):
        # 数据切分工具
        self.chunk_handler = ChunkHandler()
        # 向量数据库
        self.vector = MilvusExecutor(MilvusConfig(collection_name='hybridRag_news'))
        self.embedding = get_embedding_model('qwen')
        # 切分数据检测适配器
        self.adapters = adapters or []

    async def load_data_and_chunk(self,data_path):
        documents = load_document(data_path)
        monitor_task_status("load documents numbers",len(documents))
        chunks = self.chunk_handler.recursive_chunk(documents)
        monitor_task_status("origin chunks numbers",len(chunks))
        # 按序执行适配器
        for adapter in self.adapters:
            chunks = adapter(chunks)
            monitor_task_status("processed adapter chunks numbers",len(chunks))
        return chunks

    async def save_to_vector(self):
        data_path = '/Users/sdy/hybridRag/src/crawler/data'
        documents = await self.load_data_and_chunk(data_path)
        await self.vector.client.aadd_documents(documents=documents)


if __name__ == '__main__':
    async_run(DataDBStorage([short_chunk_adapter]).save_to_vector())