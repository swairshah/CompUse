import os
import asyncio
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple, NamedTuple

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from pydantic_ai import RunContext, Tool
from pydantic_ai.tools import ToolDefinition

from gui_tools import GuiToolDeps

def configure_logging(verbose: bool = False):
    """Configure logging level based on verbosity."""
    level = logging.DEBUG if verbose else logging.ERROR

    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(levelname)s - %(name)s - %(message)s'
    )

    logging.getLogger().setLevel(level)
    
    for logger_name in logging.root.manager.loggerDict:
        logging.getLogger(logger_name).setLevel(level)

@dataclass
class ServerConfig:
    """Configuration for an MCP server."""
    name: str
    command: str
    args: List[str]
    env: Optional[Dict[str, str]] = None
    setup_delay: float = 0.0

PUPPETEER_SERVER = ServerConfig(
    name="Puppeteer",
    command="npx",
    args=["@modelcontextprotocol/server-puppeteer"],
    env=os.environ,
)

GIT_SERVER = ServerConfig(
    name="Git",
    command="uvx",
    args=["mcp-server-git", "--repository", os.getcwd()],
)

def get_filesystem_server_config():
    """Get filesystem server configuration with dynamic paths."""
    current_dir = os.getcwd()
    claude_dir = os.path.expanduser("~/claude-dir")
    
    os.makedirs(claude_dir, exist_ok=True)
    
    return ServerConfig(
        name="Filesystem",
        command="npx",
        args=["@modelcontextprotocol/server-filesystem", current_dir, claude_dir],
        env=os.environ,
        setup_delay=1.0  # Add a delay to ensure server initialization
    )

@dataclass
class ServerConnection:
    """Represents a connection to an MCP server."""
    name: str
    session: ClientSession
    ctx: Any
    tools: List[Tool] = field(default_factory=list)
    tool_dict: Dict = field(default_factory=dict)

@dataclass
class CombinedDeps(GuiToolDeps):
    """Dependencies for combined agent with GUI and MCP tools."""
    server_map: Dict[str, ServerConnection] = field(default_factory=dict)
    
    def __getattr__(self, name):
        """Access server sessions by name with fallback."""
        if name.endswith('_session') and name[:-8] in self.server_map:
            return self.server_map[name[:-8]].session
        return super().__getattribute__(name)


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

async def initialize_server(config: ServerConfig) -> Tuple[ServerConnection, Exception]:
    """Initialize an MCP server from its configuration.
    
    Returns:
        A tuple of (server_connection, error). If successful, error will be None.
        If there's an error, server_connection will be None.
    """
    try:
        logging.info(f"Starting {config.name} MCP server...")
        server_params = StdioServerParameters(
            command=config.command,
            args=config.args,
            env=config.env,
        )

        stdio_ctx = stdio_client(server_params)
        read, write = await stdio_ctx.__aenter__()
        session = ClientSession(read, write)
        await session.__aenter__()
        
        # Add optional delay for servers that need time to initialize
        if config.setup_delay > 0:
            await asyncio.sleep(config.setup_delay)
        
        logging.info(f"Initializing {config.name} server session...")
        await session.initialize()
        logging.info(f"{config.name} server initialized successfully")
        
        # Test the connection with a lightweight call
        tools_response = await session.list_tools()
        if not tools_response:
            raise RuntimeError(f"No response from {config.name} server")
        
        logging.info(f"{config.name} server connected successfully")
        
        # Create the server connection object
        server_conn = ServerConnection(
            name=config.name,
            session=session,
            ctx=stdio_ctx,
        )
        
        # Load the tools
        try:
            tools, tool_dict = await load_mcp_tools(session)
            server_conn.tools = tools
            server_conn.tool_dict = tool_dict
        except Exception as e:
            logging.error(f"Failed to load tools from {config.name} server: {str(e)}")
        
        return server_conn, None
        
    except Exception as e:
        logging.error(f"Failed to initialize {config.name} server: {str(e)}")
        return None, e

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

async def graceful_shutdown(server_conn: ServerConnection) -> None:
    """Gracefully shut down a server connection.
    
    This function handles the proper sequence of cleanup operations to avoid issues with the event loop.
    """
    name = server_conn.name
    
    if not server_conn.session:
        logging.warning(f"No active session for {name} server")
        return
        
    try:
        logging.info(f"Shutting down {name} server...")
        
        # First close the session
        if server_conn.session:
            try:
                logging.info(f"Closing {name} session...")
                await asyncio.wait_for(server_conn.session.__aexit__(None, None, None), timeout=3.0)
                logging.info(f"{name} session closed")
            except Exception as e:
                logging.warning(f"Error closing {name} session: {str(e)}")
        
        # Then close the context
        if server_conn.ctx:
            try:
                logging.info(f"Closing {name} context...")
                await asyncio.wait_for(server_conn.ctx.__aexit__(None, None, None), timeout=3.0)
                logging.info(f"{name} context closed")
            except Exception as e:
                logging.warning(f"Error closing {name} context: {str(e)}")
                
        logging.info(f"{name} server shutdown complete")
    except Exception as e:
        logging.error(f"Error during {name} server shutdown: {str(e)}")

async def shutdown_all_servers(servers: Dict[str, ServerConnection]) -> None:
    """Shut down all server connections gracefully.
    
    Args:
        servers: Dictionary mapping server names to ServerConnection objects
    """
    if not servers:
        logging.info("No servers to shut down")
        return
        
    logging.info(f"Shutting down {len(servers)} servers...")
    
    # Create shutdown tasks
    shutdown_tasks = [
        graceful_shutdown(server) 
        for server in servers.values()
    ]
    
    # Execute all shutdown tasks
    if shutdown_tasks:
        results = await asyncio.gather(*shutdown_tasks, return_exceptions=True)
        
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                server_name = list(servers.keys())[i]
                logging.error(f"Error shutting down {server_name}: {str(result)}")
    
    logging.info("All servers shut down")
