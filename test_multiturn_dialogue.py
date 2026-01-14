#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
测试多轮对话功能
使用test_row_data.json中的数据进行测试
"""

import asyncio
import json
from typing import List, Dict, Any
from src.graph import Graph
from utils.async_task import async_run
from langchain_core.messages import HumanMessage


def load_test_data() -> Dict[str, Any]:
    """加载测试数据"""
    with open('/Users/sdy/hybridRag/tests/eval/test_row_data.json', 'r', encoding='utf-8') as f:
        data = json.load(f)
        # 返回第一个测试用例
        return data[0]


async def test_multiturn_dialogue():
    """测试多轮对话功能"""
    print("=" * 60)
    print("开始测试多轮对话功能")
    print("=" * 60)
    
    # 加载测试数据
    test_case = load_test_data()
    text_context = test_case["text"]
    qas = test_case["qas"]
    
    print(f"背景文本长度: {len(text_context)} 字符")
    print(f"问题数量: {len(qas)}")
    print()
    
    # 初始化图
    graph = await Graph().graph
    
    # 构建对话历史
    messages = []
    
    # 添加背景信息作为上下文
    print("系统: 我已加载相关背景信息，您可以开始提问。")
    print("-" * 40)
    
    # 依次提出所有问题，模拟多轮对话
    for i, qa in enumerate(qas, 1):
        question = qa["question"]
        expected_answer = qa["answer"]
        
        print(f"用户问题 {i}: {question}")
        print(f"期望答案: {expected_answer}")
        
        # 添加用户问题到消息历史
        messages.append(HumanMessage(content=question))
        
        # 配置
        user_id = "test_user"
        thread_id = "test_thread"
        config = {
            'configurable': {
                'thread_id': f'{thread_id}_{i}',
                'user_id': user_id
            }
        }
        
        # 输入
        inputs = {
            "messages": messages
        }
        
        print("助手回答: ", end="")
        
        # 流式处理响应
        response_text = ""
        async for output in graph.astream(inputs, config=config):
            for key, value in output.items():
                if "messages" in value:
                    # 获取最新的消息
                    latest_message = value["messages"][-1]
                    if hasattr(latest_message, 'content'):
                        response_text += latest_message.content
                        print(latest_message.content, end="", flush=True)
        
        print("\n" + "-" * 40)
        
        # 更新消息历史
        from langchain_core.messages import AIMessage
        messages.append(AIMessage(content=response_text))
    
    print("多轮对话测试完成！")


async def test_specific_questions():
    """测试特定的问题序列"""
    print("=" * 60)
    print("测试特定问题序列")
    print("=" * 60)
    
    graph = await Graph().graph
    
    # 问题序列
    questions = [
        "这个游戏的基本玩法是什么？",
        "生命数耗完即算为什么？",  # 直接引用测试数据中的问题
        "如果游戏中途离开会怎样？",  # 相关问题
        "游戏有哪些模式可以选择？"  # 相关问题
    ]
    
    messages = []
    
    for i, question in enumerate(questions, 1):
        print(f"问题 {i}: {question}")
        
        messages.append(HumanMessage(content=question))
        
        config = {
            'configurable': {
                'thread_id': f'test_thread_{i}',
                'user_id': 'test_user'
            }
        }
        
        inputs = {
            "messages": messages
        }
        
        print("回答: ", end="")
        
        response_text = ""
        async for output in graph.astream(inputs, config=config):
            for key, value in output.items():
                if "messages" in value:
                    latest_message = value["messages"][-1]
                    if hasattr(latest_message, 'content'):
                        response_text += latest_message.content
                        print(latest_message.content, end="", flush=True)
        
        print("\n" + "-" * 50)
        
        # 更新消息历史
        from langchain_core.messages import AIMessage
        messages.append(AIMessage(content=response_text))


async def main():
    """主函数"""
    print("选择测试模式:")
    print("1. 完整测试数据对话")
    print("2. 特定问题序列")
    
    choice = input("请输入选择 (1 或 2): ").strip()
    
    if choice == "1":
        await test_multiturn_dialogue()
    elif choice == "2":
        await test_specific_questions()
    else:
        print("无效选择，运行完整测试...")
        await test_multiturn_dialogue()


if __name__ == "__main__":
    asyncio.run(main())