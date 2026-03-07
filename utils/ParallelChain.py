from typing import List, Dict, Any
from langchain.chat_models import BaseChatModel
from langchain_core.output_parsers import JsonOutputParser, StrOutputParser
from langchain_core.runnables import RunnableParallel, RunnableSerializable, RunnableConfig
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables.utils import Input

from src.observability.logger import get_logger


logger = get_logger(__name__)


class ParallelChain:
    """请求并发器"""
    def __init__(self, llm_client: BaseChatModel):
        self.llm_client = llm_client
        self.task_map:Dict[str,RunnableSerializable] = {}

    async def runnable_parallel(self,input_:Input,config:RunnableConfig=None,**kwargs) -> Dict[str, Any]:
        """并行执行所有增强任务"""
        logger.debug(f"[ParallelChain] 开始并行执行: tasks_count={len(self.task_map)}")
        map_chain = RunnableParallel(self.task_map)
        responses = await map_chain.ainvoke(input_,config,**kwargs)
        logger.debug(f"[ParallelChain] 并行执行完成: responses_count={len(responses)}")
        return responses

    def parse_parallel_response(self,responses:Dict[str, Any]) -> List:
        result = []
        for part, response in responses.items():
            logger.debug(f"[ParallelChain] 任务响应: task={part}, response_type={type(response).__name__}")
            if isinstance(response, str):
                result.append(response)
            elif isinstance(response, list):
                result.extend(response)
            else:
                logger.warning(f"[ParallelChain] 无法解析响应类型: task={part}, type={type(response).__name__}")

        return result

    def create_chain(self, prompt: str | list, parse='str', config: dict = None) -> RunnableSerializable:
        if not config:
            config = {}
        if isinstance(prompt, str):
            chain_model = ChatPromptTemplate.from_template(prompt)
        elif isinstance(prompt,list):
            chain_model = ChatPromptTemplate.from_messages(prompt)
        else:
            raise ValueError('不支持的模版创建方式')
        if config and isinstance(config, dict):
            chain_model = chain_model | self.llm_client.with_config(configurable=config)
        else:
            chain_model = chain_model | self.llm_client
        if parse == 'json':
            chain_model = chain_model | JsonOutputParser()
        elif parse == 'str':
            chain_model = chain_model | StrOutputParser()
        return chain_model