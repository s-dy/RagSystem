import asyncio

from src.core.router.classification.model import ProcessingRequirements, RouteDecision
from src.core.router.classification.router import ClassifierRouter


async def test_intelligent_router_extended():
    """扩展的智能路由器测试 - 覆盖多个领域和复杂场景"""

    config = {
        "max_history_size": 2000,
        "enable_learning": True,
        "cache_size": 500
    }

    router = ClassifierRouter(config)

    # 分类别的测试用例
    test_categories = {
        "医疗健康领域": [
            {
                "query": "感冒的症状有哪些？",
                "description": "基础医学知识查询",
                "user_context": {"domain": "medical", "user_expertise_level": "beginner"}
            },
            {
                "query": "如何诊断糖尿病？需要做哪些检查？",
                "description": "医学诊断流程查询",
                "user_context": {"domain": "medical", "user_expertise_level": "intermediate"}
            },
            {
                "query": "对比阿司匹林和对乙酰氨基酚在镇痛效果和副作用方面的差异",
                "description": "药物对比分析",
                "user_context": {"domain": "medical", "user_expertise_level": "expert", "requires_citations": True}
            },
            {
                "query": "设计一个针对高血压患者的个性化治疗方案",
                "description": "医疗方案设计",
                "user_context": {"domain": "medical", "user_expertise_level": "professional"}
            },
            {
                "query": "根据患者血压实时数据调整用药剂量",
                "description": "实时医疗决策",
                "user_context": {"domain": "medical", "user_expertise_level": "expert", "requires_real_time": True}
            }
        ],

        "金融投资领域": [
            {
                "query": "什么是市盈率？",
                "description": "金融术语查询",
                "user_context": {"domain": "finance", "user_expertise_level": "beginner"}
            },
            {
                "query": "如何开通美股账户？具体步骤是什么？",
                "description": "金融流程查询",
                "user_context": {"domain": "finance", "user_expertise_level": "intermediate"}
            },
            {
                "query": "对比价值投资和成长投资策略在不同经济周期的表现",
                "description": "投资策略对比分析",
                "user_context": {"domain": "finance", "user_expertise_level": "expert"}
            },
            {
                "query": "为中等风险承受能力的客户设计一个资产配置组合",
                "description": "金融规划",
                "user_context": {"domain": "finance", "user_expertise_level": "professional", "client_risk": "medium"}
            },
            {
                "query": "当前特斯拉股价是多少？实时买入信号分析",
                "description": "实时金融数据查询",
                "user_context": {"domain": "finance", "user_expertise_level": "intermediate",
                                 "requires_real_time": True}
            },
            # 测试HYBRID模式触发
            {
                "query": "设计一个智能家居系统并分析其能耗效率",
                "description": "混合任务测试（设计+分析）",
                "user_context": {"domain": "technology", "user_expertise_level": "expert"}
            },
            # 测试步骤数量分配
            {
                "query": "详细分析机器学习在医疗诊断中的应用，包括技术原理、实施步骤、风险控制和未来展望",
                "description": "多步骤复杂分析",
                "user_context": {"domain": "medical", "user_expertise_level": "expert"}
            },
            # 测试实时性分级
            {
                "query": "当前北京交通拥堵情况",
                "description": "低风险实时查询",
                "user_context": {"domain": "general", "user_expertise_level": "beginner"}
            }
        ],

        "法律咨询领域": [
            {
                "query": "什么是正当防卫？",
                "description": "法律概念查询",
                "user_context": {"domain": "legal", "user_expertise_level": "beginner"}
            },
            {
                "query": "劳动合同纠纷的诉讼流程是怎样的？",
                "description": "法律程序查询",
                "user_context": {"domain": "legal", "user_expertise_level": "intermediate"}
            },
            {
                "query": "比较《民法典》和原《合同法》在违约责任规定上的异同",
                "description": "法律条文对比分析",
                "user_context": {"domain": "legal", "user_expertise_level": "expert", "requires_citations": True}
            },
            {
                "query": "为一个初创公司设计完整的法律风险防范体系",
                "description": "法律方案设计",
                "user_context": {"domain": "legal", "user_expertise_level": "professional", "company_type": "startup"}
            },
            {
                "query": "审核这份采购合同的潜在法律风险",
                "description": "法律文件审查",
                "user_context": {"domain": "legal", "user_expertise_level": "expert", "document_type": "contract"}
            }
        ],

        "教育教学领域": [
            {
                "query": "勾股定理是什么？",
                "description": "数学知识查询",
                "user_context": {"domain": "education", "user_expertise_level": "beginner",
                                 "grade_level": "middle_school"}
            },
            {
                "query": "如何教授小学生理解分数概念？教学步骤是什么？",
                "description": "教学方法查询",
                "user_context": {"domain": "education", "user_expertise_level": "intermediate",
                                 "subject": "mathematics"}
            },
            {
                "query": "对比项目式学习和传统讲授式学习在STEM教育中的效果",
                "description": "教育方法对比分析",
                "user_context": {"domain": "education", "user_expertise_level": "expert", "research_purpose": True}
            },
            {
                "query": "为高中生物课程设计一个学期的探究式学习方案",
                "description": "课程设计",
                "user_context": {"domain": "education", "user_expertise_level": "professional", "subject": "biology"}
            },
            {
                "query": "根据学生实时测试数据调整教学策略",
                "description": "个性化教学调整",
                "user_context": {"domain": "education", "user_expertise_level": "expert", "adaptive_teaching": True}
            }
        ],

        "科技编程领域": [
            {
                "query": "Python中的列表和元组有什么区别？",
                "description": "编程概念查询",
                "user_context": {"domain": "technology", "user_expertise_level": "beginner", "language": "python"}
            },
            {
                "query": "如何使用Docker部署一个Web应用？具体步骤是什么？",
                "description": "技术流程查询",
                "user_context": {"domain": "technology", "user_expertise_level": "intermediate", "tech_stack": "docker"}
            },
            {
                "query": "比较React、Vue和Angular在大型企业应用中的优劣",
                "description": "技术栈对比分析",
                "user_context": {"domain": "technology", "user_expertise_level": "expert",
                                 "project_scale": "enterprise"}
            },
            {
                "query": "设计一个支持百万并发的微服务架构",
                "description": "系统架构设计",
                "user_context": {"domain": "technology", "user_expertise_level": "expert", "concurrent_users": 1000000}
            },
            {
                "query": "实时监控服务器状态并自动扩缩容",
                "description": "实时系统运维",
                "user_context": {"domain": "technology", "user_expertise_level": "intermediate", "monitoring": True}
            }
        ],

        "商务管理领域": [
            {
                "query": "什么是SWOT分析？",
                "description": "商业概念查询",
                "user_context": {"domain": "business", "user_expertise_level": "beginner"}
            },
            {
                "query": "如何进行市场调研？具体方法和步骤是什么？",
                "description": "商业流程查询",
                "user_context": {"domain": "business", "user_expertise_level": "intermediate"}
            },
            {
                "query": "对比传统零售和电商模式在成本结构和客户获取方面的差异",
                "description": "商业模式对比分析",
                "user_context": {"domain": "business", "user_expertise_level": "expert"}
            },
            {
                "query": "为公司制定未来三年的数字化转型战略",
                "description": "商业战略规划",
                "user_context": {"domain": "business", "user_expertise_level": "professional", "company_size": "large"}
            },
            {
                "query": "根据实时销售数据调整营销策略",
                "description": "实时商业决策",
                "user_context": {"domain": "business", "user_expertise_level": "expert", "real_time_analytics": True}
            }
        ],

        "边缘案例和异常场景": [
            {
                "query": "",
                "description": "空查询",
                "user_context": {"domain": "general", "user_expertise_level": "beginner"}
            },
            {
                "query": "这是一个非常非常长的查询，包含了很多不必要的重复信息和细节描述，目的是测试系统对长文本的处理能力，以及是否能够准确提取关键信息进行分类和路由决策。",
                "description": "超长文本处理",
                "user_context": {"domain": "general", "user_expertise_level": "intermediate"}
            },
            {
                "query": "$$$ @@@ 特殊符号测试 ***",
                "description": "特殊字符处理",
                "user_context": {"domain": "general", "user_expertise_level": "beginner"}
            },
            {
                "query": "查询1 对比 分析2 设计3 实时4",
                "description": "混合类型查询",
                "user_context": {"domain": "general", "user_expertise_level": "expert", "mixed_requirements": True}
            },
            {
                "query": "紧急！服务器宕机！需要立即处理！",
                "description": "紧急情况处理",
                "user_context": {"domain": "technology", "user_expertise_level": "expert", "urgent": True,
                                 "emergency": True}
            }
        ],

        "多语言支持测试": [
            {
                "query": "What is machine learning?",
                "description": "英文基础查询",
                "user_context": {"domain": "technology", "user_expertise_level": "beginner", "language": "en"}
            },
            {
                "query": "機械学習とは何ですか？",
                "description": "日文查询",
                "user_context": {"domain": "technology", "user_expertise_level": "beginner", "language": "ja"}
            },
            {
                "query": "Qu'est-ce que l'apprentissage automatique ?",
                "description": "法文查询",
                "user_context": {"domain": "technology", "user_expertise_level": "beginner", "language": "fr"}
            },
            {
                "query": "¿Qué es el aprendizaje automático?",
                "description": "西班牙文查询",
                "user_context": {"domain": "technology", "user_expertise_level": "beginner", "language": "es"}
            }
        ],

        "专业领域深入测试": [
            {
                "query": "量子纠缠的基本原理及其在量子计算中的应用",
                "description": "量子物理专业查询",
                "user_context": {"domain": "physics", "user_expertise_level": "expert",
                                 "specialization": "quantum_physics"}
            },
            {
                "query": "CRISPR-Cas9基因编辑技术的机制和伦理考量",
                "description": "生物技术专业查询",
                "user_context": {"domain": "biology", "user_expertise_level": "expert", "specialization": "genetics"}
            },
            {
                "query": "区块链共识算法在分布式系统中的应用与优化",
                "description": "区块链专业查询",
                "user_context": {"domain": "technology", "user_expertise_level": "expert",
                                 "specialization": "blockchain"}
            },
            {
                "query": "气候变化对农业生产力的长期影响评估模型",
                "description": "气候科学专业查询",
                "user_context": {"domain": "environmental", "user_expertise_level": "expert",
                                 "specialization": "climate_science"}
            }
        ]
    }

    print("=" * 100)
    print("智能路由器扩展测试")
    print("=" * 100)

    # 执行测试
    total_tests = 0
    passed_tests = 0

    for category, test_cases in test_categories.items():
        print(f"\n{'=' * 50}")
        print(f"测试类别: {category}")
        print(f"{'=' * 50}")

        for idx, test_case in enumerate(test_cases, 1):
            total_tests += 1
            query = test_case["query"]
            description = test_case["description"]
            user_context = test_case["user_context"]

            print(f"\n测试 #{total_tests}: {description}")
            print(f"查询: {query[:50]}{'...' if len(query) > 50 else ''}")
            print(f"领域: {user_context.get('domain', 'general')}")
            print(f"用户级别: {user_context.get('user_expertise_level', 'unknown')}")

            try:
                # 执行路由决策
                mode, characteristics, decision_info = await router.route_query(
                    query, user_context
                )

                # 输出结果
                print(f"✓ 任务类型: {characteristics.task_type.value}")
                print(f"✓ 处理模式: {mode.value}")
                print(f"✓ 步骤数量: {characteristics.steps_required}")

                if 'error' in decision_info:
                    print(f"⚠ 警告: {decision_info['error']}")
                else:
                    passed_tests += 1

            except Exception as e:
                print(f"✗ 测试失败: {str(e)}")

    # 性能测试
    print(f"\n{'=' * 50}")
    print("性能测试")
    print(f"{'=' * 50}")

    # 并发测试
    async def concurrent_test(num_queries=10):
        queries = [
            (f"测试查询{i}", {"domain": "technology", "user_expertise_level": "intermediate"})
            for i in range(num_queries)
        ]

        tasks = [
            router.route_query(query, context)
            for query, context in queries
        ]

        import time
        start_time = time.time()
        results = await asyncio.gather(*tasks)
        end_time = time.time()

        print(f"并发处理 {num_queries} 个查询耗时: {end_time - start_time:.2f}秒")
        print(f"平均每个查询: {(end_time - start_time) / num_queries:.3f}秒")

    await concurrent_test(10)

    # 统计结果
    print(f"\n{'=' * 50}")
    print("测试总结")
    print(f"{'=' * 50}")
    print(f"总测试用例: {total_tests}")
    print(f"成功测试: {passed_tests}")
    print(f"失败测试: {total_tests - passed_tests}")
    print(f"成功率: {passed_tests / total_tests * 100:.1f}%")

    # 边界条件测试
    print(f"\n{'=' * 50}")
    print("边界条件测试")
    print(f"{'=' * 50}")

    boundary_tests = [
        # 极端长的查询
        ("极长查询" * 100, {"domain": "general"}),
        # 只有符号的查询
        ("!@#$%^&*()", {"domain": "general"}),
        # 重复的查询（测试缓存）
        ("什么是人工智能？", {"domain": "technology"}),
        ("什么是人工智能？", {"domain": "technology"}),
    ]

    for query, context in boundary_tests:
        try:
            mode, characteristics, _ = await router.route_query(query, context)
            print(f"✓ 边界测试通过: {query[:30]}... -> {mode.value}")
        except Exception as e:
            print(f"✗ 边界测试失败: {str(e)}")


async def test_specific_scenarios():
    """特定场景深度测试"""

    config = {"max_history_size": 1000}
    router = ClassifierRouter(config)

    print("\n" + "=" * 80)
    print("特定场景深度测试")
    print("=" * 80)

    # 场景1: 用户明确指定需求
    print("\n场景1: 用户明确指定处理模式")
    explicit_requirements = ProcessingRequirements()
    explicit_requirements.preferred_mode = RouteDecision.AGENTIC_RAG

    result = await router.route_query(
        "设计一个智能家居系统",
        {"domain": "technology", "user_expertise_level": "expert"},
        explicit_requirements
    )
    print(f"用户指定模式: {explicit_requirements.preferred_mode.value}")
    print(f"实际选择模式: {result[0].value}")

    # 场景2: 高风险领域
    print("\n场景2: 高风险领域（医疗）")
    medical_queries = [
        "心脏搭桥手术的风险",
        "癌症治疗方案选择",
        "药物相互作用查询",
    ]

    for query in medical_queries:
        mode, characteristics, _ = await router.route_query(
            query,
            {"domain": "medical", "user_expertise_level": "professional"}
        )
        print(f"查询: {query[:30]}... -> 模式: {mode.value}")

    # 场景3: 实时性要求
    print("\n场景3: 实时数据查询")
    real_time_queries = [
        "当前比特币价格",
        "实时交通路况",
        "股市最新动态",
        "天气预报更新",
    ]

    for query in real_time_queries:
        mode, characteristics, _ = await router.route_query(
            query,
            {"domain": "general", "user_expertise_level": "intermediate", "real_time": True}
        )
        print(f"实时查询: {query[:30]}... -> 模式: {mode.value}")


async def test_learning_capability():
    """测试系统的学习能力"""

    config = {
        "max_history_size": 100,
        "enable_learning": True
    }

    router = ClassifierRouter(config)

    print("\n" + "=" * 80)
    print("学习能力测试")
    print("=" * 80)

    # 模拟用户反馈学习
    test_queries = [
        ("如何煮鸡蛋？", "cooking", "beginner"),
        ("如何煮鸡蛋？", "cooking", "beginner"),
        ("如何煮鸡蛋？", "cooking", "beginner"),  # 重复查询，测试是否会学习
        ("机器学习算法对比", "technology", "expert"),
        ("投资策略分析", "finance", "intermediate"),
    ]

    for i, (query, domain, level) in enumerate(test_queries, 1):
        print(f"\n学习测试 #{i}")
        print(f"查询: {query}")

        mode, characteristics, decision_info = await router.route_query(
            query,
            {"domain": domain, "user_expertise_level": level}
        )

        print(f"决策: {mode.value}")
        if 'similar_queries' in decision_info:
            print(f"找到相似历史查询: {len(decision_info['similar_queries'])}条")


# 添加新的测试用例来验证协调机制
async def test_coordination_mechanism():
    """测试LLM和系统推荐的协调"""

    config = {"max_history_size": 1000}
    router = ClassifierRouter(config)

    test_scenarios = [
        {
            "name": "LLM和系统一致",
            "query": "如何诊断糖尿病？",
            "user_context": {"domain": "medical", "user_expertise_level": "intermediate"},
            "expected_coordination": "llm_in_list"  # LLM推荐应在系统推荐列表中
        },
        {
            "name": "高风险领域，LLM推荐过于简单",
            "query": "心脏搭桥手术的详细步骤",
            "user_context": {"domain": "medical", "user_expertise_level": "expert"},
            # 系统应该拒绝LLM的native推荐，选择更安全的模式
        },
        {
            "name": "LLM推荐更复杂模式",
            "query": "Python列表和元组的简单区别",
            "user_context": {"domain": "technology", "user_expertise_level": "beginner"},
            # 如果LLM推荐agentic但系统推荐native，系统应保持native
        }
    ]

    for scenario in test_scenarios:
        print(f"\n测试场景: {scenario['name']}")
        result = await router.route_query(
            scenario["query"],
            scenario["user_context"]
        )

        # 检查协调结果
        decision_info = result[2]
        if 'recommendation_comparison' in decision_info:
            comparison = decision_info['recommendation_comparison']
            print(f"  系统推荐: {comparison['system_recommended']}")
            print(f"  LLM推荐: {comparison['llm_recommended']}")
            print(f"  最终选择: {comparison['selected']}")
            print(f"  选择原因: {comparison['reason']}")

if __name__ == "__main__":
    print("开始执行智能路由器全面测试...")
    print("=" * 100)

    # 运行所有测试
    asyncio.run(test_intelligent_router_extended())
    asyncio.run(test_specific_scenarios())
    asyncio.run(test_learning_capability())
    asyncio.run(test_coordination_mechanism())

    print("\n" + "=" * 100)
    print("所有测试执行完成！")
    print("=" * 100)