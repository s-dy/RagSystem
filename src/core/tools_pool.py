from typing import Any
import json
from langchain_core.tools import BaseTool
from langchain_mcp_adapters.client import MultiServerMCPClient

from config import MCP_SERVER
from src.observability.logger import get_logger
from utils.decorator import singleton

logger = get_logger(__name__)

@singleton
class ToolsPool:
    def __init__(self):
        self.tools: dict[str, BaseTool] = {}
        self.initialized = False
        self._initializing = False

    async def ensure_initialized(self):
        """懒加载：首次调用工具时才连接 MCP 服务器加载工具"""
        if self.initialized or self._initializing:
            return
        self._initializing = True
        try:
            self._load_system_default_tools()
            await self._load_mcp_tools()
            self.initialized = True
            logger.info(f"[ToolsPool] 初始化完成: tools_count={len(self.tools)}")
        except Exception as exc:
            logger.error(f"[ToolsPool] 初始化失败: {exc}")
        finally:
            self._initializing = False

    async def initialize(self, *args, **kwargs):
        """兼容旧调用方式，委托给 ensure_initialized"""
        await self.ensure_initialized()

    def _load_system_default_tools(self):
        """加载系统默认工具"""
        pass

    async def _load_mcp_tools(self):
        """连接 MCP 服务器并加载工具"""
        client = MultiServerMCPClient(MCP_SERVER)
        for server in MCP_SERVER.keys():
            try:
                tools = await client.get_tools(server_name=server)
                logger.info(f"[ToolsPool] MCP工具加载成功: server={server}, tools_count={len(tools)}")
                for tool in tools:
                    self.add_tool(tool)
            except Exception as exc:
                logger.warning(f"[ToolsPool] MCP工具加载失败: server={server}, error={exc}")

    def add_tool(self, tool):
        """添加工具"""
        if isinstance(tool,BaseTool):
            logger.debug(f"[ToolsPool] 添加工具: {tool.name}")
            self.tools[tool.name] = tool

    def get_tool(self,name) -> BaseTool:
        """获取工具"""
        return self.tools.get(name)

    def get_tools(self) -> dict[str,BaseTool]:
        return self.tools

    async def call_tool(self, name: str, tool_input: str | dict[str, Any], *args, **kwargs):
        """调用工具（首次调用时自动初始化）"""
        if not self.initialized:
            await self.ensure_initialized()
        tool = self.get_tool(name)
        if not tool:
            logger.warning(f"[ToolsPool] 工具不存在: {name}")
            return None
        for _ in range(3):
            try:
                if isinstance(tool, BaseTool):
                    result = await tool.ainvoke(tool_input, *args, **kwargs)
                    result_preview = str(result)[:200] if result else None
                    logger.debug(f"[ToolsPool] 工具调用成功: tool={name}, result_preview={result_preview}")
                    return result
            except Exception as exc:
                logger.error(f"[ToolsPool] 工具调用失败: tool={name}, error={exc}")
        return None

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