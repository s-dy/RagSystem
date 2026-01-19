from typing import List, Dict, Any, Optional
import re
from langchain.chat_models import BaseChatModel
from langchain_core.runnables import RunnableSerializable

from src.services.time_transforme import TimeParseTool
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

    async def enhance(self, query: str, user_context: Dict[str, Any] = None, conversation_context: str = "") -> List[Dict[str, Any]]:
        """增强查询"""
        monitor_task_status('query enhancer starting...')
        # enhanced_queries = [query]  # 始终包含原始查询
        if not user_context:
            user_context = {}

        # 基于专业水平改写
        user_expertise = user_context.get("user_expertise_level", "beginner")
        if self.config.formalize and user_expertise != "beginner" and (task := self._formalize_rewrite_query_with_coref(query,conversation_context)):
            self.task_map['formalize'] = task

        # 扩展改写
        if self.config.expand and (task := self._expand_rewrite_query_with_coref(query,conversation_context)):
            self.task_map['expand'] = task

        # 同义改写
        if self.config.paraphrase and (task := self._paraphrase_rewrite_query_with_coref(query,conversation_context)):
            self.task_map['paraphrase'] = task

        if self.config.enable_query_decomposition and (task := self._decompose_query_with_coref(query,conversation_context)):
            self.task_map['decomposition'] = task

        if self.config.hyde_predict and (task := self._predict_query_with_context_enhanced(query,conversation_context)):
            self.task_map['predict'] = task

        # 并行执行所有增强任务
        responses = await self.runnable_parallel({'query': query,'conversation_context': conversation_context})
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

    def _paraphrase_rewrite_query_with_coref(self, query: str, conversation_context: str = "") -> RunnableSerializable:
        """基于对话上下文进行指代消解的同义改写"""

        if conversation_context:
            prompt = """
            您是查询优化专家，负责基于对话历史进行指代消解和同义改写。请根据对话历史理解和优化当前查询：

            对话历史：
            {conversation_context}

            当前查询：
            {query}

            严格遵循以下要求：
            1. 首先分析对话历史，识别并解析所有代词和指代（如"他"、"它"、"这个"、"上述"等）
            2. 将当前查询中的代词替换为明确的实体名称
            3. 基于上下文补充必要的背景信息，使查询更加完整和明确
            4. 生成2-3个不同表述的查询版本，确保每个版本都能独立被理解
            5. 保持原始查询的核心意图不变
            6. 如果问题涉及时间，请保持原问题中的时间描述，不需要修改

            输出格式（JSON数组）：
            [
              "完整且明确的查询1",
              "同义但不同表述的查询2",
              "简化的查询3（可选）"
            ]

            示例：
            对话历史：用户之前问过"李呈瑞是谁？"
            当前查询："他有哪些成就？"
            输出：["李呈瑞有哪些成就？", "李呈瑞的主要成就是什么？", "李呈瑞取得了哪些重要成就？"]

            现在请处理以下内容：
            """
        else:
            prompt = """
            您是查询优化专家，负责将用户查询用不同的方式重新表述，生成同义词或相关表述，以提高检索精度。

            原始查询：{query}

            严格遵循以下要求：
            1. 只输出答案即可，不用返回其他无关内容
            2. 保持完全相同的含义，不添加额外信息
            3. 使用不同的词汇或句式
            4. 补充有助于检索的细节和术语
            5. 如果问题中带有时间，你不需要提供准确的时间，使用问题中提供的时间即可
            6. 最多返回2条结果

            请用JSON数组格式回复改写后的查询，如：["查询1", "查询2"]
            """

        return self.create_chain(prompt, parse='json', config={"llm_temperature": 0.2})

    def _expand_rewrite_query_with_coref(self, query: str, conversation_context: str = "") -> RunnableSerializable:
        """基于对话上下文进行指代消解的扩展改写"""

        if conversation_context:
            prompt = """
            您是查询优化专家，负责基于对话历史进行指代消解和查询扩展。请根据对话历史理解和扩展当前查询：

            对话历史：
            {conversation_context}

            当前查询：
            {query}

            严格遵循以下要求：
            1. 首先解析当前查询中的所有代词和指代，基于对话历史确定其指代的实体
            2. 将代词替换为明确的实体名称
            3. 扩展查询时考虑对话历史中的背景信息
            4. 使查询更具体、详细、完整，明确隐含的需求
            5. 保持原始查询的核心意图不变
            6. 如果问题涉及时间，请保持原问题中的时间描述，不需要修改
            7. 只输出最终的扩展查询，不要解释过程

            输出格式（单个扩展后的查询字符串）：
            "扩展后的完整查询"

            示例：
            对话历史：用户之前问过"介绍一下Python语言"
            当前查询："它的应用场景有哪些？"
            输出："Python语言的应用场景有哪些？包括但不限于：Web开发、数据分析、人工智能等具体领域。"

            现在请处理以下内容：
            """
        else:
            prompt = """
            您是查询优化专家，负责将用户查询扩展为更具体、完整、详细的版本，以提高检索精度。

            原始查询：{query}

            严格遵循以下要求：
            1. 只输出答案即可，不用返回其他无关内容
            2. 明确隐含的需求，使查询更具体
            3. 如果问题中带有时间，你不需要提供准确的时间，使用问题中提供的时间即可

            输出格式（单个扩展后的查询字符串）：
            "扩展后的完整查询"
            """

        return self.create_chain(prompt, parse='str', config={"llm_temperature": 0.2})

    def _decompose_query_with_coref(self, query: str, conversation_context: str = "") -> Optional[RunnableSerializable]:
        """基于对话上下文进行指代消解的查询分解"""

        if conversation_context:
            prompt = """
            请基于对话历史，将以下复杂查询分解为一系列按顺序执行的子问题。每个子问题应能独立回答，且后续问题可引用前序答案：

            对话历史：
            {conversation_context}

            当前查询：
            {query}

            分解要求：
            1. 首先分析对话历史，识别并解析所有代词和指代
            2. 将当前查询中的代词替换为明确的实体名称
            3. 基于对话历史的背景信息，将复杂查询拆解为多个逻辑相关的子问题
            4. 保持原意不变，不添加额外信息
            5. 将复杂查询拆解为多个子问题，逐步获取信息
            6. 输出拆解后的查询链，按顺序输出为JSON数组格式
            7. 每个子查询都应该是完整且独立的

            输出格式（JSON数组）：
            ["明确后的子查询1", "明确后的子查询2", ...]

            示例：
            对话历史：用户之前问过"介绍一下阿里巴巴公司"
            当前查询："它的创始人是谁？还有主要业务有哪些？"
            输出：["阿里巴巴公司的创始人是谁？", "阿里巴巴公司的主要业务有哪些？"]

            现在请处理以下内容：
            """
        else:
            prompt = """
            请将以下复杂查询分解为一系列按顺序执行的子问题。每个子问题应能独立回答，且后续问题可引用前序答案：

            原始查询：{query}

            分解要求：
            1. 保持原意不变，不添加额外信息
            2. 将复杂查询拆解为多个子问题，逐步获取信息
            3. 输出拆解后的查询链，按顺序输出为JSON数组格式
            4. 识别查询中的多个独立问题或要求，将每个问题或要求分解为独立的查询

            输出格式（JSON数组）：
            ["子查询1", "子查询2"]
            """

        return self.create_chain(prompt, parse='json')

    def _formalize_rewrite_query_with_coref(self, query: str, conversation_context: str = "") -> RunnableSerializable:
        """基于对话上下文进行指代消解的专业化改写"""

        if conversation_context:
            prompt = """
            您是查询优化专家，负责基于对话历史进行指代消解和专业术语改写。请根据对话历史将查询改写为更正式、专业的表述：

            对话历史：
            {conversation_context}

            当前查询：
            {query}

            严格遵循以下要求：
            1. 首先解析当前查询中的代词和指代，基于对话历史确定其指代的实体
            2. 将代词替换为明确的实体名称
            3. 使用更专业的术语，句子结构更严谨，适合学术或专业场景
            4. 保持原始查询的核心意图不变
            5. 如果问题中带有时间，你不需要提供准确的时间，使用问题中提供的时间即可
            6. 只输出最终的改写查询，不要解释过程

            输出格式（单个改写后的查询字符串）：
            "改写后的专业查询"

            示例：
            对话历史：用户之前问过"李呈瑞是谁？"
            当前查询："他什么时候参军的？"
            输出："李呈瑞同志是在哪一年加入中国人民解放军的？"

            现在请处理以下内容：
            """
        else:
            prompt = """
            您是查询优化专家，负责将用户查询改写为更正式、专业的表述。

            原始查询：{query}

            严格遵循以下要求：
            1. 只输出答案即可，不用返回其他无关内容
            2. 使用更专业的术语，句子结构更严谨，适合学术或专业场景
            3. 如果问题中带有时间，你不需要提供准确的时间，使用问题中提供的时间即可

            输出格式（单个改写后的查询字符串）：
            "改写后的专业查询"
            """

        return self.create_chain(prompt, parse='str', config={"llm_temperature": 0.2})

    def _predict_query_with_context_enhanced(self, query: str, conversation_context: str = "") -> RunnableSerializable:
        """增强版HyDE predict（考虑对话历史和指代消解）"""

        if conversation_context:
            prompt = """
            根据对话历史和当前问题，生成一个详细的、基于上下文的回答草案。这个回答草案将用于指导文档检索：

            对话历史：
            {conversation_context}

            当前问题：
            {query}

            生成回答草案的要求：
            1. 首先分析对话历史，理解上下文背景
            2. 解析当前问题中的代词和指代，将其明确化
            3. 基于对话历史中的信息和当前问题的需求，生成一个详细的、假设性的回答
            4. 这个回答应该包含可能的答案要点和相关信息
            5. 保持回答的专业性和准确性
            6. 如果问题涉及时间，请保持原问题中的时间范围
            7. 生成的回答应该能很好地指导后续的文档检索

            输出格式（单个回答草案字符串）：
            "详细的回答草案..."

            示例：
            对话历史：用户之前问过"李呈瑞是谁？"
            当前问题："他获得过哪些勋章？"
            输出："李呈瑞同志可能获得的勋章包括但不限于：八一勋章、独立自由勋章、解放勋章等。作为一位资深革命军人，他可能在抗日战争、解放战争等不同时期获得相应的荣誉勋章。"

            现在请处理以下内容：
            """
        else:
            prompt = """
            根据以下问题，生成一个详细的、假设性的回答草案。这个回答草案将用于指导文档检索：

            问题：{query}

            生成要求：
            1. 生成一个包含可能答案要点和相关信息的详细回答
            2. 保持回答的专业性和准确性
            3. 如果问题涉及时间，请保持原问题中的时间范围
            4. 这个回答应该能很好地指导后续的文档检索

            输出格式（单个回答草案字符串）：
            "详细的回答草案..."
            """

        return self.create_chain(prompt, parse='str', config={"llm_temperature": 0.2})

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
    from src.services.llm.models import get_qwen_model
    model = get_qwen_model()
    query_enhancer = QueryEnhancer(model,QueryEnhancementConfig(paraphrase=False,expand=False,enable_query_decomposition=False,hyde_predict=False,decompose_to_subquestions=True))
    async_run(query_enhancer.enhance('A 的 CEO 是谁？他在哪所大学获得博士学位？',{"user_expertise_level":"expert"}))