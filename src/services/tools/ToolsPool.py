from typing import Any
from langchain_core.tools import BaseTool
from langchain_mcp_adapters.client import MultiServerMCPClient

from config.Config import MCP_SERVER
from src.monitoring.logger import monitor_task_status
from utils.async_task import async_run
from utils.decortor import singleton


@singleton
class ToolsPool:
    def __init__(self):
        self.tools:dict[str,BaseTool] = {}

        self.system_default_tools()
        self.init_mcp_tools()

    def system_default_tools(self):
        """系统默认工具"""
        from langchain_community.tools import DuckDuckGoSearchResults
        ddg_tool = DuckDuckGoSearchResults(name='ddg_search')
        self.add_tool(ddg_tool)

    def init_mcp_tools(self):
        # 连接到 MCP 服务器
        client = MultiServerMCPClient(MCP_SERVER)

        # 获取工具
        for server in MCP_SERVER.keys():
            tools = async_run(client.get_tools(server_name=server))
            print(f"{server} 已获取 {len(tools)} 个工具")
            [self.add_tool(tool) for tool in tools]


    def add_tool(self, tool):
        """添加工具"""
        if isinstance(tool,BaseTool):
            monitor_task_status('add tool',tool.name)
            self.tools[tool.name] = tool

    def get_tool(self,name) -> BaseTool:
        """获取工具"""
        return self.tools.get(name)

    def get_tools(self) -> dict[str,BaseTool]:
        return self.tools

    def call_tool(self,name:str,tool_input:str | dict[str, Any],*args,**kwargs):
        """调用工具"""
        tool = self.get_tool(name)
        if not tool:
            monitor_task_status('not exists tool',name)
        for _ in range(3):
            try:
                if isinstance(tool,BaseTool):
                    return tool.invoke(tool_input,*args,**kwargs)
            except Exception as e:
                monitor_task_status(f'tool call error 【{name}】',e)

    def get_format_tool(self) -> str:
        """获取工具格式化的列表"""
        result = []
        for tool in self.tools.values():
            if isinstance(tool,BaseTool):
                result.append(f"- {tool.name}: {tool.description}")

        return "\n".join(result)

if __name__ == '__main__':
    print(ToolsPool().call_tool('bing_search', {'query':'搜索美食方面的一片文章，并进行解析'}))
    print(ToolsPool().get_tools())
