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

# Configure logging globally
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(name)s - %(message)s'
)

def configure_logging(verbose: bool = False):
    """Configure logging level based on verbosity."""
    logging.basicConfig(
        level=logging.INFO if verbose else logging.WARNING,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )


@dataclass
class CombinedDeps(GuiToolDeps):
    """Dependencies for combined agent with GUI and MCP tools."""
    mcp_session: Optional[ClientSession] = None
    git_session: Optional[ClientSession] = None
    filesystem_session: Optional[ClientSession] = None


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

async def initialize_git_server() -> Tuple[ClientSession, Any]:
    """Initialize the Git MCP server."""
    try:
        logging.info("Starting Git MCP server...")
        server_params = StdioServerParameters(
            command="uvx",
            args=["mcp-server-git", "--repository", os.getcwd()]
        )

        stdio_ctx = stdio_client(server_params)
        read, write = await stdio_ctx.__aenter__()
        session = ClientSession(read, write)
        await session.__aenter__()
        
        logging.info("Initializing Git server session...")
        await session.initialize()
        logging.info("Git server initialized successfully")
        
        # Test the connection with a lightweight call
        tools_response = await session.list_tools()
        # Check if the response is valid without using len()
        if not tools_response:
            raise RuntimeError("No response from Git server")
        
        logging.info("Git server connected successfully")
        return session, stdio_ctx
    
    except Exception as e:
        logging.error(f"Failed to initialize Git server: {str(e)}")
        raise

async def initialize_filesystem_server() -> Tuple[ClientSession, Any]:
    """Initialize the Filesystem MCP server."""
    try:
        # The issue might be with the directory paths. Make sure they exist and are accessible
        current_dir = os.getcwd()
        claude_dir = os.path.expanduser("~/claude-dir")
        
        # Ensure the claude-dir exists
        os.makedirs(claude_dir, exist_ok=True)
        
        logging.info(f"Starting filesystem server with dirs: {current_dir} and {claude_dir}")
        
        # Use full paths and add debug logging
        server_params = StdioServerParameters(
            command="npx",
            args=["@modelcontextprotocol/server-filesystem", current_dir, claude_dir],
            env=os.environ,
        )
        
        stdio_ctx = stdio_client(server_params)
        read, write = await stdio_ctx.__aenter__()
        session = ClientSession(read, write)
        await session.__aenter__()
        
        # Add a longer delay to ensure the server has time to initialize
        await asyncio.sleep(2)
        
        logging.info("Initializing filesystem server session...")
        await session.initialize()
        logging.info("Filesystem server initialized successfully")
        
        # Test the connection with a lightweight call
        tools_response = await session.list_tools()
        # Check if the response is valid without using len()
        if not tools_response:
            raise RuntimeError("No response from filesystem server")
        
        logging.info("Filesystem server connected successfully")
        return session, stdio_ctx
        
    except Exception as e:
        logging.error(f"Failed to initialize filesystem server: {str(e)}")
        raise

async def initialize_puppeteer_server() -> Tuple[ClientSession, Any]:
    """Initialize the Puppeteer MCP server."""
    try:
        logging.info("Starting Puppeteer MCP server...")
        server_params = StdioServerParameters(
            command="npx",
            args=["@modelcontextprotocol/server-puppeteer"],
            env=os.environ,
        )

        stdio_ctx = stdio_client(server_params)
        read, write = await stdio_ctx.__aenter__()
        session = ClientSession(read, write)
        await session.__aenter__()
        
        logging.info("Initializing Puppeteer server session...")
        await session.initialize()
        logging.info("Puppeteer server initialized successfully")
        
        # Test the connection with a lightweight call
        tools_response = await session.list_tools()
        # Check if the response is valid without using len()
        if not tools_response:
            raise RuntimeError("No response from Puppeteer server")
        
        logging.info("Puppeteer server connected successfully")
        return session, stdio_ctx
    
    except Exception as e:
        logging.error(f"Failed to initialize Puppeteer server: {str(e)}")
        raise


async def load_mcp_tools(session: ClientSession) -> Tuple[List[Tool], Dict]:
    """Load tools from the MCP server."""
    try:
        tools_response = await session.list_tools()
        pydantic_tools = []
        tool_dict = {}
        
        if not tools_response:
            logging.warning("Server response was empty or invalid")
            return [], {}
        
        # Extract tools from the response
        tools_list = []
        for item in tools_response:
            if isinstance(item, tuple) and item[0] == 'tools':
                tools_list = item[1]
                break
        
        if not tools_list:
            logging.warning("No tools found in server response")
            return [], {}
        
        # Process each tool
        for tool in tools_list:
            try:
                executor = await create_mcp_tool_executor(session, tool.name)
                preparor = await create_mcp_tool_preparor(tool.name, tool.description, tool.inputSchema)
                mcp_tool = Tool(executor, prepare=preparor)
                
                pydantic_tools.append(mcp_tool)
                tool_dict[tool.name] = {'description': tool.description, 'schema': tool.inputSchema}
            except Exception as e:
                logging.error(f"Failed to create tool for {getattr(tool, 'name', 'unknown')}: {str(e)}")
        
        logging.info(f"Loaded {len(pydantic_tools)} MCP tools")
        return pydantic_tools, tool_dict
    
    except Exception as e:
        logging.error(f"Failed to load MCP tools: {str(e)}")
        return [], {}
