from typing import Any
import json
from langchain_core.tools import BaseTool
from langchain_mcp_adapters.client import MultiServerMCPClient

from config.Config import MCP_SERVER
from src.monitoring.logger import monitor_task_status
from utils.async_task import async_run
from utils.decorator import singleton


@singleton
class ToolsPool:
    def __init__(self):
        self.tools:dict[str,BaseTool] = {}
        self.init_instance = False

    async def initialize(self, *args, **kwargs):
        self.system_default_tools()
        await self.init_mcp_tools()
        self.init_instance = True

    def system_default_tools(self):
        """系统默认工具"""
        # from langchain_community.tools import DuckDuckGoSearchResults
        # ddg_tool = DuckDuckGoSearchResults(name='ddg_search')
        # self.add_tool(ddg_tool)

    async def init_mcp_tools(self):
        # 连接到 MCP 服务器
        client = MultiServerMCPClient(MCP_SERVER)

        # 获取工具
        for server in MCP_SERVER.keys():
            try:
                tools = await client.get_tools(server_name=server)
                print(f"{server} 已获取 {len(tools)} 个工具")
                [self.add_tool(tool) for tool in tools]
            except Exception as e:
                monitor_task_status(f'MCP工具【{server}】获取失败',e)


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

    async def call_tool(self,name:str,tool_input:str | dict[str, Any],*args,**kwargs):
        """调用工具"""
        tool = self.get_tool(name)
        if not tool:
            monitor_task_status('not exists tool',name)
        for _ in range(3):
            try:
                if isinstance(tool,BaseTool):
                    result = await tool.ainvoke(tool_input,*args,**kwargs)
                    monitor_task_status('call tool result',result)
                    return result
            except Exception as e:
                monitor_task_status(f'tool call error 【{name}】',e)

    def get_format_tool(self) -> str:
        """获取工具格式化的列表"""
        result = []
        for tool in self.tools.values():
            if isinstance(tool,BaseTool):
                result.append(f"- {tool.name}: {tool.description}")

        return "\n".join(result)

    def get_response(self,response) -> list:
        result = []
        if not response:
            return []
        for item in response:
            if isinstance(item,dict) and 'text' in item:
                cur = item['text']
                if not cur:
                    continue
                try:
                    cur = json.loads(cur)
                except Exception as e:
                    pass
                result.append(cur)
        return result


if __name__ == '__main__':
    pool = ToolsPool()
    async_run(pool.initialize())
    result = pool.call_tool('crawl_webpage', {'uuids':['1'],"url_map":{"1":'https://www.sohu.com/a/804240518_122031860'}})
    print(pool.get_response(result))

