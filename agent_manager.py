from __future__ import annotations

import os
import asyncio
import logging
from dataclasses import dataclass
from typing import List, Optional, Any
from rich.table import Table

from pydantic_ai import Agent

from pydantic_ai.models.openai import OpenAIModel

from pydantic_ai.messages import (
    ModelMessage,
    ModelMessagesTypeAdapter,
    ModelRequest,
    ModelResponse,
    TextPart,
    UserPromptPart,
)

from agent import (
    initialize_puppeteer_server, 
    load_mcp_tools, 
    CombinedDeps, 
    initialize_apple_server
)

from gui_tools import (
    screenshot, 
    mouse_move, 
    mouse_click, 
    keyboard_type, 
    key_press, 
    get_screen_size,
    get_mouse_position,
    switch_window,
    focus_application
)

@dataclass
class ServerContext:
    """Tracks a server's session and stdio context."""
    session: Any
    stdio_ctx: Any
    name: str

@dataclass
class CombinedDeps:
    """Dependencies for the agent, including MCP sessions."""
    mcp_session: Any = None
    apple_mcp_session: Any = None

@dataclass
class AgentManager:
    """Manages the agent, its tools, and conversation history."""
    agent: Agent
    deps: CombinedDeps
    message_history: List[ModelMessage]
    gui_tools: List[str]
    mcp_tools: List[str]
    server_contexts: List[ServerContext] = None  # Store all server contexts

    @classmethod
    async def initialize(cls) -> AgentManager:
        """Initialize the agent manager with all necessary components."""
        server_contexts = []
        all_mcp_tools = []
        tool_dict = {}
        
        # Initialize Puppeteer server
        puppeteer_session, puppeteer_stdio = await initialize_puppeteer_server()
        server_contexts.append(ServerContext(puppeteer_session, puppeteer_stdio, "puppeteer"))
        
        # Load Puppeteer MCP tools
        puppeteer_tools, puppeteer_tool_dict = await load_mcp_tools(puppeteer_session)
        all_mcp_tools.extend(puppeteer_tools)
        tool_dict.update(puppeteer_tool_dict)
        
        # Initialize Apple server
        apple_session, apple_stdio = await initialize_apple_server()
        server_contexts.append(ServerContext(apple_session, apple_stdio, "apple"))
        
        # Load Apple MCP tools
        apple_tools, apple_tool_dict = await load_mcp_tools(apple_session)
        all_mcp_tools.extend(apple_tools)
        tool_dict.update(apple_tool_dict)
        
        # Create model
        model = OpenAIModel('gpt-4o')
        
        # Create tools list
        gui_tool_list = [
            screenshot, mouse_move, mouse_click, keyboard_type, key_press,
            get_screen_size, get_mouse_position, switch_window, focus_application
        ]
        all_tools = [*gui_tool_list, *all_mcp_tools]
        
        # Create agent
        agent = Agent(
            model,
            deps_type=CombinedDeps,
            retries=1,
            tools=all_tools
        )
        
        # Fix MCP tool schemas
        for tool_name, tool_info in tool_dict.items():
            print(tool_name)
            agent._function_tools[tool_name]._parameters_json_schema = tool_info['schema']
            agent._function_tools[tool_name].description = tool_info['description']
        
        # Initialize dependencies with sessions
        # You'll need to update CombinedDeps to handle multiple sessions or adapt this approach
        deps = CombinedDeps(
            mcp_session=puppeteer_session,
            apple_mcp_session=apple_session
        )
        
        # Get tool names for display
        gui_tool_names = ["screenshot", "mouse_move", "mouse_click", "keyboard_type", "key_press", 
                       "get_screen_size", "get_mouse_position", "switch_window", "focus_application"]
        mcp_tool_names = list(tool_dict.keys())
        
        # Create screenshots directory
        os.makedirs(os.path.join(os.getcwd(), "screenshots"), exist_ok=True)
        
        return cls(
            agent=agent,
            deps=deps,
            message_history=[],
            gui_tools=gui_tool_names,
            mcp_tools=mcp_tool_names,
            server_contexts=server_contexts
        )
    
    async def cleanup(self):
        """Clean up all MCP servers and stdio contexts."""
        for ctx in self.server_contexts:
            # Clean up session
            if ctx.session:
                try:
                    await asyncio.wait_for(ctx.session.__aexit__(None, None, None), timeout=5.0)
                except (asyncio.TimeoutError, Exception) as e:
                    logging.warning(f"{ctx.name} session cleanup warning: {str(e)}")
            
            # Clean up stdio context
            if ctx.stdio_ctx:
                try:
                    await asyncio.wait_for(ctx.stdio_ctx.__aexit__(None, None, None), timeout=5.0)
                except (asyncio.TimeoutError, Exception) as e:
                    logging.warning(f"{ctx.name} STDIO cleanup warning: {str(e)}")
    
    async def run_command(self, command: str) -> tuple[str, float]:
        """Run a command through the agent and return the result and elapsed time."""
        start_time = asyncio.get_event_loop().time()
        
        result = await self.agent.run(
            command,
            message_history=self.message_history,
            deps=self.deps
        )
        
        # Update history
        self.message_history.extend(result.new_messages())
        
        elapsed = asyncio.get_event_loop().time() - start_time
        return result.data, elapsed
    
    def reset_history(self):
        """Reset the conversation history."""
        self.message_history = []
    
    def get_history_table(self) -> Table:
        """Get a formatted table of conversation history."""
        history_table = Table(title="Conversation History")
        history_table.add_column("Role", style="command")
        history_table.add_column("Content", style="info")
        
        for msg in self.message_history:
            if isinstance(msg, ModelRequest):
                first_part = msg.parts[0]
                if isinstance(first_part, UserPromptPart):
                    history_table.add_row("User", first_part.content)
            elif isinstance(msg, ModelResponse):
                first_part = msg.parts[0]
                if isinstance(first_part, TextPart):
                    history_table.add_row("Assistant", first_part.content)
        
        return history_table
    
    def get_tools_table(self) -> Table:
        """Get a formatted table of available tools."""
        tool_table = Table(title="Available Tools")
        tool_table.add_column("Type", style="highlight")
        tool_table.add_column("Tools", style="info")
        
        tool_table.add_row("GUI Tools", ", ".join(self.gui_tools))
        tool_table.add_row("MCP Tools", ", ".join(self.mcp_tools))
        
        return tool_table