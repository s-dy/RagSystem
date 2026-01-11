import re
from collections import defaultdict
from typing import Dict, Any, List
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

    # 语义特征
    entities: List[str] = field(default_factory=list)  # 提取的实体列表（如人名、地名、机构名等）

    # 执行特征
    steps_required: int = 1
    requires_external_tools: bool = False  # 是否需要调用外部工具/API
    requires_real_time_data: bool = False # 实时数据需求

    # 结构特征
    comparison_count: int = 0  # 比较关系的数量（如"A和B哪个更好"）
    action_verbs: List[str] = field(default_factory=list)  # 动作动词列表（如"计算"、"比较"等）
    numeric_values: List[float] = field(default_factory=list)  # 数值列表（用于量化查询）

    def __repr__(self) -> str:
        # 格式化列表（中文友好）
        def fmt_list(lst: list, max_items: int = 5) -> str:
            if not lst:
                return "无"
            display = lst[:max_items]
            suffix = f" 等 {len(lst)} 项" if len(lst) > max_items else ""
            return "、".join(str(item) for item in display) + suffix

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

        return (
            f"【任务特征分析结果】\n"
            f"- 任务类型：{type_names.get(self.task_type, self.task_type.value)}\n"
            f"- 涉及实体：{fmt_list(self.entities)}\n"
            f"- 动作动词：{fmt_list(self.action_verbs)}\n"
            f"- 数值参数：{fmt_list([f'{v:g}' for v in self.numeric_values])}\n"
            f"- 比较对象数量：{self.comparison_count}\n"
            f"- 所需步骤数：{self.steps_required}\n"
            f"- 需要外部工具：{'是' if self.requires_external_tools else '否'}\n"
            f"- 需要实时数据：{'是' if self.requires_real_time_data else '否'}"
        )


class TaskAnalyzer:
    """任务分析器：识别任务特征和需求"""

    def __init__(self):
        self.task_patterns = {
            TaskType.FACT_RETRIEVAL: [
                r'什么是.*[？?]', r'.*是什么[？?]', r'.*的定义', r'.*解释',
                r'谁.*[？?]', r'何时.*[？?]', r'哪里.*[？?]', r'何地.*[？?]',
                r'多少.*[？?]', r'是否.*[？?]', r'有没有.*[？?]', r'存在.*吗[？?]'
            ],
            TaskType.PROCEDURAL_QUERY: [
                r'如何.*[？?]', r'怎样.*[？?]', r'怎么.*[？?]',
                r'(?:步骤|流程|方法|操作|指南).*[？?]',
                r'(?:安装|配置|设置|使用|运行).*[？?]'
            ],
            TaskType.ANALYTICAL_COMPARISON: [
                r'(?:对比|比较|对照).*[和与跟].*',
                r'.*(?:优缺点|优势劣势|好坏|强弱|异同)',
                r'(?:分析|评估|评价|评测).*数据',
                r'(?:有什么区别|有什么不同|差异.*)'
            ],
            TaskType.CREATIVE_GENERATION: [
                r'(?:生成|创作|编写|设计|起草|草拟|构思|想象).*',
                r'(?:建议|推荐|提供).*方案',
                r'(?:写一首|画一个|做一个).*'
            ],
            TaskType.COMPLEX_PLANNING: [
                r'(?:制定|规划|部署|安排|组织|实施|执行|开展).*计划',
                r'(?:项目|活动|日程|策略|架构).*设计'
            ],
            TaskType.MULTI_STEP_EXECUTION: [
                r'(?:首先|第一步).*(?:然后|接着|第二步).*(?:最后|第三步)',
                r'(?:分步骤|多阶段|迭代|循环).*',
                r'(?:如果|当).*时?.*(?:那么|就).*'
            ],
            TaskType.REAL_TIME_INTERACTION: [
                r'(?:实时|当前|现在|最新|刚刚|立即|马上|此刻|此时|今日|今天).*',
                r'(?:最新情况|最新消息|最新进展)'
            ],
            TaskType.VALIDATION_VERIFICATION: [
                r'(?:验证|检查|确认|核对|审核|审查|确保|保证|确定).*'
            ]
        }

        self.action_verbs_keywords = {
            "计算": ["计算", "统计", "求和", "平均", "百分比", "总计"],
            "搜索": ["搜索", "查找", "查询", "检索", "探索", "挖掘"],
            "分析": ["分析", "建模", "预测", "趋势", "关联", "诊断"],
            "生成": ["生成", "创建", "编写", "设计", "绘制", "输出"],
            "验证": ["验证", "检查", "测试", "调试", "审核", "校验"],
            "推理": ["推理", "推断", "推测", "猜想", "假设", "因果", "因为", "所以", "因此"]
        }

        # 编译正则表达式
        self.compiled_patterns = {}
        for task_type, patterns in self.task_patterns.items():
            self.compiled_patterns[task_type] = [
                re.compile(pattern, re.IGNORECASE) for pattern in patterns
            ]

        # 实体提取模式
        self.entity_patterns = {
            "money": r'[¥$€]\d+(?:\.\d+)?|\d+(?:\.\d+)?[元美元欧元]',
        }

    def analyze_task(self, query: str) -> TaskCharacteristics:
        """分析任务特征"""
        query_lower = query.lower()

        # 1. 识别任务类型
        task_type_scores = self._identify_task_type(query_lower)
        primary_task_type = max(task_type_scores.items(), key=lambda x: x[1])[0]

        # 2. 提取基础特征
        entities = self.extract_entities(query)
        action_verbs = self._extract_action_verbs(query_lower)
        numeric_values = self._extract_numeric_values(query)
        comparison_count = self._count_comparisons(query_lower)

        # 3. 分析执行特征
        execution_features = self._analyze_execution_features(
            query_lower, primary_task_type, action_verbs
        )

        # 4. 组合所有特征
        characteristics = TaskCharacteristics(
            task_type=primary_task_type,
            entities=entities,
            action_verbs=action_verbs,
            numeric_values=numeric_values,
            comparison_count=comparison_count,
            **execution_features
        )

        return characteristics

    def _identify_task_type(self, query: str) -> Dict[TaskType, float]:
        """识别任务类型"""
        scores = defaultdict(float)

        for task_type, patterns in self.compiled_patterns.items():
            match_count = 0
            for pattern in patterns:
                if pattern.search(query):
                    match_count += 1
            if match_count > 0:
                # 分数 = min(1.0, 0.3 * 匹配模式数)
                scores[task_type] = min(1.0, 0.3 * match_count)

        # 如果没有匹配，默认事实检索
        if not scores:
            scores[TaskType.FACT_RETRIEVAL] = 0.5

        return dict(scores)

    def _extract_action_verbs(self, query: str) -> List[str]:
        """提取动作动词"""
        verbs = set()
        for category, keywords in self.action_verbs_keywords.items():
            for keyword in keywords:
                if keyword in query:
                    verbs.add(keyword)
        return sorted(list(verbs))

    def _extract_numeric_values(self, query: str) -> List[float]:
        """提取数值"""
        numbers = []
        # 匹配数字（包括带单位的）
        number_pattern = r'(\d+(?:\.\d+)?)(?:[万亿%]?)'
        matches = re.findall(number_pattern, query)
        for match in matches:
            try:
                numbers.append(float(match))
            except ValueError:
                continue
        return numbers

    def _count_comparisons(self, query: str) -> int:
        """计算比较关系数量"""
        # 检测 "A和B"、"X与Y" 等结构
        comparison_patterns = [
            r'.*[和与跟].*',
            r'(?:对比|比较|对照)',
            r'(?:vs|VS| versus )'
        ]
        count = 0
        for pattern in comparison_patterns:
            if re.search(pattern, query, re.IGNORECASE):
                count += 1
        return count

    def _analyze_execution_features(
            self,
            query: str,
            task_type: TaskType,
            action_verbs: List[str]
    ) -> Dict[str, Any]:
        """分析执行特征"""
        features = {
            "steps_required": 1,
            "requires_external_tools": False,
            "requires_real_time_data": False
        }

        # 步骤数量判断
        step_indicators = ["首先", "然后", "接着", "最后", "第一步", "第二步", "分步", "多步", "逐步"]
        step_count = sum(1 for indicator in step_indicators if indicator in query)
        if step_count > 0:
            features["steps_required"] = min(step_count + 1, 10)
        elif task_type in [TaskType.MULTI_STEP_EXECUTION, TaskType.COMPLEX_PLANNING]:
            features["steps_required"] = 3  # 默认多步骤

        # 外部工具需求
        tool_categories = ["计算", "搜索", "分析", "生成", "验证"]
        if any(verb in action_verbs for verb in
               [v for cat in tool_categories for v in self.action_verbs_keywords[cat]]):
            features["requires_external_tools"] = True

        # 实时数据需求
        if task_type == TaskType.REAL_TIME_INTERACTION:
            features["requires_real_time_data"] = True

        return features

    def extract_entities(self, query: str) -> List[str]:
        """提取命名实体"""
        entities = []
        for entity_type, pattern in self.entity_patterns.items():
            try:
                matches = re.findall(pattern, query)
                for match in matches:
                    if isinstance(match, tuple):
                        match = match[0] if match else ""
                    if match and len(str(match)) > 1:  # 过滤单字符
                        entities.append(f"{entity_type}:{match}")
            except re.error:
                continue
        return list(set(entities))  # 去重


if __name__ == '__main__':
    analyzer = TaskAnalyzer()
    result = analyzer.analyze_task('获取小红书美食方面的一片文章，并进行解析')
    print(result)
