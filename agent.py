import os
import asyncio
import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Any, Tuple

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from pydantic_ai import RunContext, Tool
from pydantic_ai.tools import ToolDefinition

from gui_tools import GuiToolDeps

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)


@dataclass
class CombinedDeps(GuiToolDeps):
    """Dependencies for combined agent with GUI and MCP tools."""
    mcp_session: Optional[ClientSession] = None


async def create_mcp_tool_executor(session: ClientSession, tool_name: str, timeout: float = 120.0):
    """Create a function that will execute the tool on the MCP server."""
    async def mcp_tool_executor(ctx: RunContext, **kwargs):
        start_time = asyncio.get_event_loop().time()
        logging.info(f"Executing tool {tool_name} with args: {kwargs}")
        
        try:
            # Execute the tool with a timeout
            result = await asyncio.wait_for(
                session.call_tool(tool_name, kwargs),
                timeout=timeout
            )
            
            elapsed = asyncio.get_event_loop().time() - start_time
            logging.info(f"Tool {tool_name} completed in {elapsed:.2f}s")
            return result
            
        except asyncio.TimeoutError:
            logging.error(f"Tool {tool_name} timed out after {timeout}s")
            return {"error": f"Operation timed out after {timeout} seconds"}
            
        except Exception as e:
            elapsed = asyncio.get_event_loop().time() - start_time
            logging.error(f"Tool {tool_name} failed after {elapsed:.2f}s: {str(e)}")
            return {"error": f"Tool execution failed: {str(e)}"}
    
    mcp_tool_executor.__name__ = tool_name
    return mcp_tool_executor


async def create_mcp_tool_preparor(tool_name: str, tool_description: str, tool_schema: Dict):
    """Create a function that will prepare MCP server tool as agent Tool for pydantic."""
    async def prepare(ctx: RunContext, tool_def: ToolDefinition):
        tool_def = ToolDefinition(
            name=tool_name,
            description=tool_description,
            parameters_json_schema=tool_schema,
        )
        return tool_def

    return prepare


async def initialize_puppeteer_server() -> Tuple[ClientSession, Any]:
    """Initialize the Puppeteer MCP server."""
    server_params = StdioServerParameters(
        command="npx",
        args=["@modelcontextprotocol/server-puppeteer"],
        env=os.environ,
    )

    stdio_ctx = stdio_client(server_params)
    read, write = await stdio_ctx.__aenter__()
    session = ClientSession(read, write)
    await session.__aenter__()
    await session.initialize()
    
    return session, stdio_ctx


async def load_mcp_tools(session: ClientSession) -> Tuple[List[Tool], Dict]:
    """Load tools from the MCP server."""
    tools_response = await session.list_tools()
    pydantic_tools = []
    tool_dict = {}
    
    for item in tools_response:
        if isinstance(item, tuple) and item[0] == 'tools':
            for tool in item[1]:
                executor = await create_mcp_tool_executor(session, tool.name)
                preparor = await create_mcp_tool_preparor(tool.name, tool.description, tool.inputSchema)
                mcp_tool = Tool(executor, prepare=preparor)
                
                pydantic_tools.append(mcp_tool)
                tool_dict[tool.name] = {'description': tool.description, 'schema': tool.inputSchema}
    
    logging.info(f"Loaded {len(pydantic_tools)} MCP tools")
    return pydantic_tools, tool_dict
