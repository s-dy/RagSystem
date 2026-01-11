import json

from langchain_core.messages import SystemMessage, ToolMessage
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import BaseTool
from langgraph.graph import MessagesState,START,END,StateGraph

from src.services.llm.models import get_qwen_model
from src.services.tools.ToolsPool import ToolsPool


class ToolsAgent:
    def __init__(self):
        self.llm = get_qwen_model()
        self.tools:dict[str,BaseTool] = {} # name -> tool

        self.init_tools()
        self.workflow = self.init_graph()

    def init_tools(self):
        self.tools = ToolsPool().get_tools()
        tools_list = list(self.tools.values())
        self.llm = self.llm.bind_tools(tools_list)

    def init_graph(self):
        workflow = StateGraph(MessagesState)
        workflow.add_node('tool_node',self.tool_node)
        workflow.add_node('agent',self.call_model)

        workflow.add_edge(START,'agent')
        workflow.add_conditional_edges('agent',self.should_continue,{'end':END,'continue':'tool_node'})
        workflow.add_edge('tool_node','agent')

        graph = workflow.compile()
        return graph

    def tool_node(self,state:MessagesState):
        results = []
        for tool_call in state['messages'][-1].tool_calls:
            tool_result = self.tools[tool_call['name']].invoke(tool_call['args'])
            if isinstance(tool_result,dict):
                tool_result = json.dumps(tool_result)
            results.append(ToolMessage(
                content=tool_result,
                name=tool_call["name"],
                tool_call_id=tool_call["id"]
            ))
        return {'messages':results}

    def call_model(self,state:MessagesState,config:RunnableConfig):
        system_prompt = SystemMessage(
            "你是一个有用的AI助手，请尽你所能回答用户的问题！"
        )
        messages = state['messages']
        response = self.llm.invoke([system_prompt] + messages,config=config)
        return {'messages':[response]}

    def should_continue(self,state:MessagesState):
        messages = state['messages']
        last_message = messages[-1]
        if not last_message.tool_calls:
            return 'end'
        else:
            return 'continue'

    @property
    def graph(self):
        return self.workflow


def tool_agent():
    from langchain.agents import create_agent
    llm = get_qwen_model()
    tools = ToolsPool().get_tools()
    tools_list = list(tools.values())
    agent = create_agent(llm, tools_list)
    return agent


if __name__ == '__main__':
    from langchain_core.messages import HumanMessage
    def print_stream(stream):
        for s in stream:
            message = s["messages"][-1]
            if isinstance(message, tuple):
                print(message)
            else:
                message.pretty_print()

    graph = ToolsAgent().graph
    inputs = {"messages": [HumanMessage("搜索几篇美食的文章并进行解读")]}

    print_stream(graph.stream(inputs, stream_mode="values"))
    print_stream(tool_agent().stream(inputs, stream_mode="values"))
