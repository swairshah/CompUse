import os
import asyncio
import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Any

from dotenv import load_dotenv

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from pydantic_ai import Agent, RunContext, Tool
from pydantic_ai.models.openai import OpenAIModel
from pydantic_ai.tools import ToolDefinition

import sys
# sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from gui_tools import (
    screenshot, 
    mouse_move, 
    mouse_click, 
    keyboard_type, 
    key_press, 
    get_screen_size,
    get_mouse_position,
    switch_window,
    focus_application,
    GuiToolDeps
)

# Configure logging
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


async def initialize_puppeteer_server():
    """Initialize the Puppeteer MCP server."""
    server_params = StdioServerParameters(
        command="npx",
        args=["-y", "@modelcontextprotocol/server-puppeteer"],
        env=os.environ,
    )

    stdio_ctx = stdio_client(server_params)
    read, write = await stdio_ctx.__aenter__()
    session = ClientSession(read, write)
    await session.__aenter__()
    await session.initialize()
    
    return session, stdio_ctx


async def load_mcp_tools(session: ClientSession) -> List[Tool]:
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


async def main():
    """Main function to set up a server with both GUI and MCP tools."""
    load_dotenv()
    
    try:
        print("Initializing Puppeteer MCP server...")
        session, stdio_ctx = await initialize_puppeteer_server()
        
        print("Loading MCP tools...")
        mcp_tools, tool_dict = await load_mcp_tools(session)
        
        # List MCP tool names
        print("MCP Tools:")
        for tool_name in tool_dict.keys():
            print(f"- {tool_name}")
        
        # Create screenshots directory
        os.makedirs(os.path.join(os.getcwd(), "screenshots"), exist_ok=True)
        
        # Create model
        model = OpenAIModel('gpt-4o')
        
        # Create tools list
        all_tools = [
            # GUI tools
            screenshot,
            mouse_move,
            mouse_click,
            keyboard_type,
            key_press,
            get_screen_size,
            get_mouse_position,
            switch_window,
            focus_application,
            # MCP tools are added from the list
            *mcp_tools
        ]
        
        # Create agent with all tools
        combined_agent = Agent(
            model,
            system_prompt=(
                'You are a powerful computer control assistant that can control both the desktop and web browsers. '
                'You can interact with the computer using GUI tools and web browsers using Puppeteer tools.'
                '\n\n'
                'You have two types of tools available:'
                '\n1. PyAutoGUI tools (screenshot, mouse_move, mouse_click, etc.) for desktop control'
                '\n2. Puppeteer tools (browser*, navigate, etc.) for web browser automation'
                '\n\n'
                'When a user asks for help, decide which tools are appropriate for the task. '
                'For web browser tasks, use the Puppeteer MCP tools. For desktop application tasks, '
                'use the PyAutoGUI tools. If you need to see what\'s on screen, use the screenshot tool first.'
                '\n\n'
                'IMPORTANT SAFETY RULES:'
                '\n- Never try to access sensitive information'
                '\n- Do not try to install software without explicit permission'
                '\n- Always confirm before clicking on buttons that might change system settings'
                '\n- If you\'re uncertain about an action, ask for clarification first'
            ),
            deps_type=CombinedDeps,
            retries=1,
            tools=all_tools
        )
        
        # Fix MCP tool schemas
        for tool_name, tool_info in tool_dict.items():
            combined_agent._function_tools[tool_name]._parameters_json_schema = tool_info['schema']
            combined_agent._function_tools[tool_name].description = tool_info['description']
        
        # Initialize dependencies with MCP session
        deps = CombinedDeps(mcp_session=session)
        
        print("\nCombined Agent initialized with both PyAutoGUI and MCP tools!")
        print("You can now give commands to control your computer and browser.")
        print("Type 'exit' to quit.")
        
        # Interactive loop
        while True:
            try:
                user_input = input("\nWhat would you like to do? > ")
                
                if user_input.lower() in ['exit', 'quit', 'bye']:
                    print("Goodbye!")
                    break
                    
                print("Processing your request...")
                start_time = asyncio.get_event_loop().time()
                
                # Run the agent with the user's input directly
                result = await combined_agent.run(user_input, deps=deps)
                
                elapsed = asyncio.get_event_loop().time() - start_time
                print(f"Response ({elapsed:.2f}s):")
                print(result.data)
                
            except KeyboardInterrupt:
                print("\nGoodbye!")
                break
            except Exception as e:
                print(f"Error: {str(e)}")
    
    finally:
        # Clean up MCP server
        if 'session' in locals():
            await session.__aexit__(None, None, None)
        if 'stdio_ctx' in locals():
            await stdio_ctx.__aexit__(None, None, None)
        print("MCP server cleaned up")


if __name__ == "__main__":
    asyncio.run(main())
