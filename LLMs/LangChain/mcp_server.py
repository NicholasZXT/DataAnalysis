"""
简单的 MCP Server
"""
from typing import TYPE_CHECKING, Literal
import logging
import click
from mcp import types
from mcp.server import FastMCP
from mcp.server import Server  # 等价于 from mcp.server.lowlevel import Server
# 上面的 Server 底层依赖下面的 ServerSession —— 它底层直接基于 anyio 构建，而不是 starlette
from mcp.server.session import ServerSession
from mcp.server.fastmcp import Context
from mcp.server.fastmcp.prompts import base as prompts_base
from mcp.server.fastmcp.tools import base as tools_base
from mcp.server.stdio import stdio_server

logger = logging.getLogger(__name__)

# -------------- 实例化 FastMCP ----------------
mcp = FastMCP(
    name="Some-MCP-Server",
    # 以下两个适用于 sse/streamable-http 启动方式，均为默认值
    host="127.0.0.1",
    port=8000,
)

@mcp.tool(
    name="get_weather",
    description="获取天气信息",
)
def get_weather(city: str) -> str:
    """获取天气信息"""
    return f"{city}的天气是晴天"

@click.command()
@click.option("--transport", default="streamable-http", help="Transport protocol to use (stdio, sse, streamable-http)")
def run_mcp_server(transport: Literal["stdio", "sse", "streamable-http"]):
    logger.info(f'MCP Server starting with transport: {transport} ...')
    mcp.run(transport=transport)


if __name__ == '__main__':
    run_mcp_server()
