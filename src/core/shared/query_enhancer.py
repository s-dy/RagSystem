from typing import List, Dict, Any, Optional
import re
from langchain.chat_models import BaseChatModel
from langchain_core.runnables import RunnableSerializable

from src.core.shared.time_transforme import TimeParseTool
from src.monitoring.logger import monitor_task_status
from config.Config import QueryEnhancementConfig
from utils.ParallelChain import ParallelChain


class QueryEnhancer(ParallelChain):
    """查询增强器"""
    def __init__(self, llm_client: BaseChatModel, config: QueryEnhancementConfig = None):
        super().__init__(llm_client)
        if not config:
            config = QueryEnhancementConfig()
        self.config = config
        self.llm_client = llm_client
        self.time_parse_tool = TimeParseTool(config={'include_source': True, 'strict_mode': False})

    async def enhance(self, query: str, user_context: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        """增强查询"""
        monitor_task_status('query enhancer starting...')
        enhanced_queries = [query]  # 始终包含原始查询
        if not user_context:
            user_context = {}

        # 基于专业水平改写
        user_expertise = user_context.get("user_expertise_level", "beginner")
        if self.config.formalize and user_expertise != "beginner" and (task := self._formalize_rewrite_query(query)):
            self.task_map['formalize'] = task

        # 扩展改写
        if self.config.expand and (task := self._expand_rewrite_query(query)):
            self.task_map['expand'] = task

        # 同义改写
        if self.config.paraphrase and (task := self._paraphrase_rewrite_query(query)):
            self.task_map['paraphrase'] = task

        if self.config.enable_query_decomposition and (task := self._decompose_query(query)):
            self.task_map['decomposition'] = task

        if self.config.hyde_predict and (task := self._predict_query(query)):
            self.task_map['predict'] = task

        # 并行执行所有增强任务
        responses = await self.runnable_parallel({'query': query})
        enhanced_queries = self.parse_parallel_response(responses)
        # 去重
        enhanced_queries = self._deduplicate_queries(enhanced_queries)
        # 解析时间
        enhanced_timer_queries = self.parse_query_time(enhanced_queries)
        # 限制数量
        enhanced_result = enhanced_timer_queries[:self.config.max_enhanced_queries]
        monitor_task_status(f"Enhanced queries result: {enhanced_result}")
        monitor_task_status('query enhancer ended...')
        return enhanced_result

    def _formalize_rewrite_query(self,query:str) -> RunnableSerializable:
        """基于专业水平调整改写策略"""
        formalize_prompt = """
            您是查询优化专家，负责将用户查询改写为更正式、专业的表述。请将以下查询改写为更正式、专业的表述：

            原始查询：{query}

            严格遵循以下要求：
            1. 只输出答案即可，不用返回其他无关内容
            2. 使用更专业的术语，句子结构更严谨，适合学术或专业场景
            3. 如果问题中带有时间，你不需要提供准确的时间,使用问题中提供的时间即可。
        """
        return self.create_chain(formalize_prompt,config={"llm_temperature": 0.2})

    def _expand_rewrite_query(self,query:str) -> RunnableSerializable:
        expand_prompt = """
        您是查询优化专家，负责将用户查询扩展为更具体、完整、详细的版本，以提高检索精度。请将以下查询扩展为更具体、详细、更完整的表述：
        
        原始查询：{query}
        
        严格遵循以下要求：
        1. 只输出答案即可，不用返回其他无关内容
        2. 明确隐含的需求，使查询更具体
        3. 如果问题中带有时间，你不需要提供准确的时间,使用问题中提供的时间即可。
        """
        return self.create_chain(expand_prompt,config={"llm_temperature": 0.2})

    def _paraphrase_rewrite_query(self,query:str) -> RunnableSerializable:
        paraphrase_prompt = """
        您是查询优化专家，负责将用户查询用不同的方式重新表述,生成同义词或相关表述,以提高检索精度。请将以下查询进行优化：

        原始查询：{query}

        严格遵循以下要求：
        1. 只输出答案即可，不用返回其他无关内容
        2. 保持完全相同的含义，不添加额外信息。
        3. 使用不同的词汇或句式
        4. 补充有助于检索的细节和术语。
        5. 如果问题中带有时间，你不需要提供准确的时间,使用问题中提供的时间即可。
        6. 最多返回 2 条结果。
        
        请用JSON数组格式回复改写后的查询，如：["查询1", "查询2"]
        """
        return self.create_chain(paraphrase_prompt,parse='json',config={"llm_temperature": 0.2})

    def _predict_query(self,query:str) -> RunnableSerializable:
        """HyDE predict"""
        prompt = """
        回答以下问题：
        
        严格遵循以下要求：
        1. 只输出答案即可，不用返回其他无关内容。
        2. 如果问题中带有时间，你不需要提供准确的时间,使用问题中提供的时间即可。

        问题：{query}
        """
        return self.create_chain(prompt,parse='str',config={"llm_temperature": 0.2})

    def _decompose_query(self, query: str) -> Optional[RunnableSerializable]:
        """查询分解"""
        # 检查是否需要分解
        decomposition_indicators = [
            "和", "以及", "并且", "同时", "还有", "对比", "比较"
        ]

        needs_decomposition = any(indicator in query for indicator in decomposition_indicators)
        if not needs_decomposition:
            return None

        prompt = """
        请将以下复杂查询分解为多个独立的子查询：

        原始查询：{query}

        分解要求：
        1. 保持原意不变，不添加额外信息。
        2. 将复杂查询拆解为多个子问题，逐步获取信息。
        3. 输出拆解后的查询链，按顺序输出为json数组格式。
        4. 识别查询中的多个独立问题或要求，将每个问题或要求分解为独立的查询

        请用JSON数组格式回复子查询，如：["子查询1", "子查询2"]
        """
        return self.create_chain(prompt,parse='json',config={"llm_temperature": 0.2})

    def _deduplicate_queries(self, queries: List[str]) -> List[str]:
        """去重查询"""
        seen = set()
        unique_queries = []
        for query in queries:
            normalized = query.lower().strip()
            # 移除标点符号
            normalized = re.sub(r'[^\w\s\u4e00-\u9fa5]', '', normalized)
            if normalized and normalized not in seen:
                seen.add(normalized)
                unique_queries.append(query)

        return unique_queries

    def parse_query_time(self,queries:List[str]) -> List[Dict[str, Any]]:
        result = []
        for query in queries:
            time_span = self.time_parse_tool(query).get('time',[None,None])
            result.append({
                'query':query,
                'start_time':time_span[0],
                'end_time':time_span[1],
            })
        return result

if __name__ == '__main__':
    from utils.async_task import async_run
    from src.services.llm.models import get_ollama_deepseek_model,get_qwen_model
    model = get_qwen_model()
    query_enhancer = QueryEnhancer(model)
    async_run(query_enhancer.enhance('对比兔子警官和最美护士',{"user_expertise_level":"expert"}))