import re
import jieba
import jieba.posseg as pseg
from collections import defaultdict
from typing import Dict, Any, List, Tuple
from dataclasses import dataclass, field
from enum import Enum


class TaskType(Enum):
    """任务类型分类"""
    FACT_RETRIEVAL = "fact_retrieval"  # 事实检索
    PROCEDURAL_QUERY = "procedural_query"  # 流程/操作查询
    ANALYTICAL_COMPARISON = "analytical_comparison"  # 分析对比
    CREATIVE_GENERATION = "creative_generation"  # 创造性生成
    COMPLEX_PLANNING = "complex_planning"  # 复杂规划
    MULTI_STEP_EXECUTION = "multi_step_execution"  # 多步骤执行
    REAL_TIME_INTERACTION = "real_time_interaction"  # 实时交互
    VALIDATION_VERIFICATION = "validation_verification"  # 验证核查


@dataclass
class TaskCharacteristics:
    """任务特征分析"""
    # 基础特征
    task_type: TaskType = TaskType.FACT_RETRIEVAL
    confidence: float = 0.0  # 分类置信度

    # 执行特征
    requires_external_tools: bool = False  # 是否需要调用外部工具/API

    # 结构特征
    action_verbs: List[str] = field(default_factory=list)  # 动作动词列表
    question_count: int = 0  # 问句数量
    has_conjunctions: bool = False  # 是否有连接词
    entity_count: int = 0  # 实体数量

    # 多跳特定特征
    is_multi_hop: bool = False  # 是否为多跳问题
    hop_count: int = 0  # 预估跳数
    requires_reasoning: bool = False  # 是否需要推理

    def __repr__(self) -> str:
        # 任务类型中文映射
        type_names = {
            TaskType.FACT_RETRIEVAL: "事实检索",
            TaskType.PROCEDURAL_QUERY: "流程操作查询",
            TaskType.ANALYTICAL_COMPARISON: "分析对比",
            TaskType.CREATIVE_GENERATION: "创造性生成",
            TaskType.COMPLEX_PLANNING: "复杂规划",
            TaskType.MULTI_STEP_EXECUTION: "多步骤执行",
            TaskType.REAL_TIME_INTERACTION: "实时交互",
            TaskType.VALIDATION_VERIFICATION: "验证核查"
        }

        # 格式化列表
        def fmt_list(lst: list, max_items: int = 5) -> str:
            if not lst:
                return "无"
            display = lst[:max_items]
            suffix = f" 等 {len(lst)} 项" if len(lst) > max_items else ""
            return "、".join(str(item) for item in display) + suffix

        return (
            f"【任务特征分析结果】\n"
            f"- 任务类型：{type_names.get(self.task_type, self.task_type.value)} (置信度: {self.confidence:.2f})\n"
            f"- 是否为多跳：{'是' if self.is_multi_hop else '否'} (预估跳数: {self.hop_count})\n"
            f"- 动作动词：{fmt_list(self.action_verbs)}\n"
            f"- 需要外部工具：{'是' if self.requires_external_tools else '否'}\n"
            f"- 需要推理：{'是' if self.requires_reasoning else '否'}\n"
            f"- 查询特征：问句数={self.question_count}, 实体数={self.entity_count}"
        )


class TaskAnalyzer:
    """任务分析器：识别任务特征和需求"""

    def __init__(self):
        # 初始化jieba
        jieba.initialize()

        # 任务类型正则模式
        self.task_patterns = {
            TaskType.FACT_RETRIEVAL: [
                r'什么是.*[？?]', r'.*是什么[？?]', r'.*的定义', r'.*解释',
                r'谁.*[？?]', r'何时.*[？?]', r'哪里.*[？?]', r'何地.*[？?]',
                r'多少.*[？?]', r'是否.*[？?]', r'有没有.*[？?]', r'存在.*吗[？?]',
                r'哪一年.*[？?]', r'哪些.*[？?]', r'哪个.*[？?]', r'什么.*[？?]',
                r'简述.*', r'介绍.*'
            ],
            TaskType.PROCEDURAL_QUERY: [
                r'如何.*[？?]', r'怎样.*[？?]', r'怎么.*[？?]',
                r'(?:步骤|流程|方法|操作|指南).*[？?]',
                r'(?:安装|配置|设置|使用|运行|连接).*[？?]',
                r'(?:解决|修复|处理).*问题[？?]'
            ],
            TaskType.ANALYTICAL_COMPARISON: [
                r'(?:对比|比较|对照).*[和与跟及以及].*',
                r'.*(?:优缺点|优势劣势|好坏|强弱|异同|区别|不同)',
                r'(?:分析|评估|评价|评测).*数据',
                r'(?:有什么区别|有什么不同|差异.*)',
                r'哪个.*更好', r'哪个.*更合适', r'哪个.*更优'
            ],
            TaskType.CREATIVE_GENERATION: [
                r'(?:生成|创作|编写|设计|起草|草拟|构思|想象).*',
                r'(?:建议|推荐|提供).*方案',
                r'(?:写一首|画一个|做一个).*',
                r'(?:帮我|请).*(?:写|创作|设计|生成).*',
                r'(?:创意|创新|新颖).*'
            ],
            TaskType.COMPLEX_PLANNING: [
                r'(?:制定|规划|部署|安排|组织|实施|执行|开展).*计划',
                r'(?:项目|活动|日程|策略|架构|路线图).*设计',
                r'(?:如何规划|如何制定|如何安排).*',
                r'(?:方案|蓝图).*制定'
            ],
            TaskType.MULTI_STEP_EXECUTION: [
                r'(?:首先|第一步).*(?:然后|接着|第二步).*(?:最后|第三步)',
                r'(?:分步骤|多阶段|迭代|循环).*',
                r'(?:如果|当|假如).*时?.*(?:那么|就|则).*',
                r'(?:先.*再.*)|(?:先.*然后.*)|(?:先.*接着.*)',
                r'(?:第一步.*第二步.*第三步)'
            ],
            TaskType.REAL_TIME_INTERACTION: [
                r'(?:实时|当前|现在|最新|刚刚|立即|马上|此刻|此时|今日|今天).*',
                r'(?:最新情况|最新消息|最新进展|最新动态)',
                r'(?:现在.*情况|当前.*状态)',
                r'现在几点了', r'今天.*天气'
            ],
            TaskType.VALIDATION_VERIFICATION: [
                r'(?:验证|检查|确认|核对|审核|审查|确保|保证|确定|核实).*',
                r'.*是否正确', r'.*是否准确', r'.*是否有效',
                r'请检查.*', r'请确认.*', r'请核实.*'
            ]
        }

        # 动作动词关键词
        self.action_verbs_keywords = {
            "计算": ["计算", "统计", "求和", "平均", "百分比", "总计", "算出", "算一下"],
            "搜索": ["搜索", "查找", "查询", "检索", "探索", "挖掘", "找", "查找", "检索"],
            "分析": ["分析", "建模", "预测", "趋势", "关联", "诊断", "解析", "剖析", "解读"],
            "生成": ["生成", "创建", "编写", "设计", "绘制", "输出", "制作", "创作", "构建"],
            "验证": ["验证", "检查", "测试", "调试", "审核", "校验", "确认", "核对"],
            "推理": ["推理", "推断", "推测", "猜想", "假设", "因果", "因为", "所以", "因此", "故而"],
            "比较": ["对比", "比较", "对照", "相比", "相较于", "比起"],
            "评估": ["评估", "评价", "评定", "评判", "估价", "估值"],
            "解释": ["解释", "说明", "阐述", "诠释", "解读", "阐明"],
            "总结": ["总结", "概括", "归纳", "综述", "汇总"]
        }

        # 连接词（用于检测多跳）
        self.conjunction_words = {
            "并且", "而且", "以及", "还有", "同时", "此外", "另外", "加之", "况且",
            "然后", "接着", "之后", "随后", "接下来", "继而",
            "首先", "其次", "再次", "最后", "第一", "第二", "第三",
            "如果", "假如", "倘若", "要是", "那么", "则", "就",
            "因为", "由于", "所以", "因此", "因而", "故而",
            "虽然", "但是", "然而", "可是", "不过", "却",
            "不仅", "而且", "既", "又", "也", "还"
        }

        # 疑问词（用于计数问句）
        self.question_words = ["什么", "哪些", "哪", "谁", "何时", "何地", "为什么", "如何", "怎样", "多少", "是否",
                               "有没有"]

        # 编译正则表达式
        self.compiled_patterns = {}
        for task_type, patterns in self.task_patterns.items():
            self.compiled_patterns[task_type] = [
                re.compile(pattern, re.IGNORECASE) for pattern in patterns
            ]

    def analyze_task(self, query: str) -> TaskCharacteristics:
        """分析任务特征"""
        # 基础统计
        query_lower = query.lower()

        # 1. 识别任务类型（带置信度）
        task_type_scores = self._identify_task_type(query_lower)

        # 2. 提取语言特征
        language_features = self._extract_language_features(query)

        # 3. 判断是否为多跳问题
        multi_hop_features = self._analyze_multi_hop(query, query_lower, language_features)

        # 4. 提取动作动词
        action_verbs = self._extract_action_verbs(query)

        # 5. 分析执行特征
        execution_features = self._analyze_execution_features(
            query_lower, task_type_scores, action_verbs, language_features
        )

        # 7. 获取主要任务类型和备选类型
        primary_type, confidence = self._get_primary_task_type(task_type_scores)

        # 9. 组合所有特征
        characteristics = TaskCharacteristics(
            task_type=primary_type,
            confidence=confidence,
            action_verbs=action_verbs,
            question_count=language_features["question_count"],
            has_conjunctions=language_features["has_conjunctions"],
            entity_count=language_features["entity_count"],
            is_multi_hop=multi_hop_features["is_multi_hop"],
            hop_count=multi_hop_features["hop_count"],
            requires_reasoning=execution_features["requires_reasoning"],
            requires_external_tools=execution_features["requires_external_tools"],
        )

        return characteristics

    def _identify_task_type(self, query: str) -> Dict[TaskType, float]:
        """识别任务类型（带分数）"""
        scores = defaultdict(float)

        # 1. 正则匹配分数
        for task_type, patterns in self.compiled_patterns.items():
            match_count = 0
            for pattern in patterns:
                if pattern.search(query):
                    match_count += 1
            if match_count > 0:
                # 基础分 + 匹配数加成
                scores[task_type] = min(1.0, 0.4 + 0.1 * match_count)

        # 2. 如果没有匹配，基于启发式规则
        if not scores:
            # 检查是否有明显的问题特征
            has_question_mark = '？' in query or '?' in query
            has_question_word = any(word in query for word in self.question_words)

            if has_question_mark or has_question_word:
                scores[TaskType.FACT_RETRIEVAL] = 0.6
            else:
                scores[TaskType.CREATIVE_GENERATION] = 0.5

        # 3. 防止分数过高
        for task_type in scores:
            scores[task_type] = min(1.0, scores[task_type])

        return dict(scores)

    def _extract_language_features(self, query: str) -> Dict[str, Any]:
        """提取语言特征"""
        features = {
            "question_count": 0,
            "has_conjunctions": False,
            "entity_count": 0,
            "word_count": 0
        }

        # 统计问句数量
        features["question_count"] = query.count('？') + query.count('?')

        # 检查连接词
        for conj in self.conjunction_words:
            if conj in query:
                features["has_conjunctions"] = True
                break

        # 分词和词性标注
        words = list(pseg.cut(query))
        word_list = []
        entity_count = 0
        for pair in words:
            word = pair.word
            flag = pair.flag
            word_list.append(word)
            if flag in ['nr', 'ns', 'nt', 'nz']:
                entity_count += 1

        features["entity_count"] = entity_count
        features["word_count"] = len(word_list)

        return features

    def _analyze_multi_hop(self, query: str, query_lower: str, language_features: Dict) -> Dict[str, Any]:
        """分析是否为多跳问题"""
        result = {
            "is_multi_hop": False,
            "hop_count": 0
        }

        # 多跳判断规则（加权计算）
        multi_hop_score = 0.0

        # 规则1：多个问句（权重0.4）
        if language_features["question_count"] >= 2:
            multi_hop_score += 0.4
            result["hop_count"] = min(5, language_features["question_count"])

        # 规则2：包含连接词（权重0.3）
        if language_features["has_conjunctions"]:
            multi_hop_score += 0.3
            result["hop_count"] = max(result["hop_count"], 2)

        # 规则3：查询长度（权重0.2）
        if len(query) > 30:  # 长查询更可能是多跳
            multi_hop_score += 0.2
            result["hop_count"] = max(result["hop_count"], 2)

        # 规则4：特定多跳模式（权重0.4）
        multi_hop_patterns = [
            r'.*[和与及以及].*[？?]',  # "A和B的问题"
            r'分别.*[？?]',  # "分别..."
            r'列举.*[？?]',  # "列举..."
            r'多个.*[？?]',  # "多个..."
            r'各.*[？?]',  # "各个..."
            r'所有.*[？?]',  # "所有..."
            r'还有.*[？?]',  # "还有..."
            r'另外.*[？?]',  # "另外..."
        ]

        for pattern in multi_hop_patterns:
            if re.search(pattern, query):
                multi_hop_score += 0.4
                result["hop_count"] = max(result["hop_count"], 3)
                break

        # 规则5：实体数量（权重0.2）
        if language_features["entity_count"] >= 2:
            multi_hop_score += 0.2
            result["hop_count"] = max(result["hop_count"], 2)

        # 综合判断
        result["is_multi_hop"] = multi_hop_score >= 0.5

        # 如果没有明确跳数，根据分数估算
        if result["is_multi_hop"] and result["hop_count"] == 0:
            result["hop_count"] = int(2 + multi_hop_score * 3)

        return result

    def _extract_action_verbs(self, query: str) -> List[str]:
        """提取动作动词"""
        verbs = set()

        # 从关键词表中匹配
        for category, keywords in self.action_verbs_keywords.items():
            for keyword in keywords:
                if keyword in query:
                    verbs.add(keyword)

        # 使用jieba分词补充识别
        words = list(pseg.cut(query))
        for pair in words:
            word = pair.word
            flag = pair.flag
            if flag.startswith('v') and len(word) > 1:  # 动词且长度>1
                verbs.add(word)

        return sorted(list(verbs))

    def _analyze_execution_features(
            self,
            query: str,
            task_type_scores: Dict[TaskType, float],
            action_verbs: List[str],
            language_features: Dict
    ) -> Dict[str, Any]:
        """分析执行特征"""
        features = {
            "requires_external_tools": False,
            "requires_reasoning": False
        }

        # 外部工具需求判断
        tool_verbs = ["计算", "搜索", "分析", "生成", "验证", "比较", "评估"]
        tool_verb_list = [v for cat in tool_verbs
                          for v in self.action_verbs_keywords.get(cat, [])]

        if any(verb in action_verbs for verb in tool_verb_list):
            features["requires_external_tools"] = True

        # 推理需求判断
        reasoning_keywords = ["因为", "所以", "因此", "因而", "故而", "推理", "推断", "推测",
                              "因果", "关系", "原因", "结果", "影响", "导致"]

        if any(keyword in query for keyword in reasoning_keywords):
            features["requires_reasoning"] = True

        # 分析对比任务通常需要推理
        if TaskType.ANALYTICAL_COMPARISON in task_type_scores:
            if task_type_scores[TaskType.ANALYTICAL_COMPARISON] > 0.6:
                features["requires_reasoning"] = True

        return features

    def _calculate_complexity_score(
            self,
            language_features: Dict,
            task_type_scores: Dict[TaskType, float],
            multi_hop_features: Dict
    ) -> float:
        """计算复杂度评分"""
        complexity = 0.0

        # 1. 查询长度（权重0.2）
        if language_features["word_count"] > 50:
            complexity += 0.2
        elif language_features["word_count"] > 20:
            complexity += 0.1

        # 2. 问句数量（权重0.3）
        if language_features["question_count"] >= 3:
            complexity += 0.3
        elif language_features["question_count"] >= 2:
            complexity += 0.2

        # 3. 是否为多跳（权重0.4）
        if multi_hop_features["is_multi_hop"]:
            complexity += 0.4
            # 跳数加成
            complexity += min(0.2, multi_hop_features["hop_count"] * 0.05)

        # 4. 任务类型复杂度（权重0.3）
        complex_task_types = [
            TaskType.ANALYTICAL_COMPARISON,
            TaskType.COMPLEX_PLANNING,
            TaskType.MULTI_STEP_EXECUTION
        ]

        for task_type in complex_task_types:
            if task_type in task_type_scores and task_type_scores[task_type] > 0.5:
                complexity += 0.3
                break

        # 5. 实体数量（权重0.2）
        if language_features["entity_count"] >= 3:
            complexity += 0.2
        elif language_features["entity_count"] >= 2:
            complexity += 0.1

        return min(1.0, complexity)

    def _get_primary_task_type(self, task_type_scores: Dict[TaskType, float]) -> Tuple[TaskType, float]:
        """获取主要任务类型和备选类型"""
        if not task_type_scores:
            return TaskType.FACT_RETRIEVAL, 0.5

        # 按分数排序
        sorted_types = sorted(task_type_scores.items(), key=lambda x: x[1], reverse=True)

        # 主要类型
        primary_type, confidence = sorted_types[0]

        return primary_type, confidence

if __name__ == '__main__':
    analyzer = TaskAnalyzer()

    # 测试用例
    test_cases = [
        "李呈瑞于哪一年参加红军？他获得过哪些勋章？他于哪一年逝世？他在抗战中担任过哪些职位？",
        "如何安装Python并配置环境变量？",
        "比较Python和Java在Web开发中的优缺点",
        "帮我写一首关于春天的诗",
        "首先收集用户需求，然后进行市场分析，最后制定产品计划",
        "现在北京的天气怎么样？",
        "验证这个数学公式是否正确：E=mc²"
    ]

    for test_query in test_cases:
        print(f"\n{'=' * 60}")
        print(f"测试查询: {test_query}")
        print('-' * 60)
        result = analyzer.analyze_task(test_query)
        print(result)