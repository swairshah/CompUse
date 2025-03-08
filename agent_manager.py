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
    _session: Optional[Any] = None  # Store MCP session
    _stdio_ctx: Optional[Any] = None  # Store stdio context
    _apple_session: Optional[Any] = None  # Store Apple MCP session
    _apple_stdio_ctx: Optional[Any] = None  # Store Apple stdio context

    @classmethod
    async def initialize(cls) -> AgentManager:
        """Initialize the agent manager with all necessary components."""
        # Initialize Puppeteer server
        session, stdio_ctx = await initialize_puppeteer_server()
        
        # Load MCP tools
        puppeteer_tools, puppeteer_tool_dict = await load_mcp_tools(session)
        
        # Initialize Apple server
        apple_session, apple_stdio_ctx = await initialize_apple_server()
        
        # Load Apple MCP tools
        apple_tools, apple_tool_dict = await load_mcp_tools(apple_session)
        mcp_tools = [*puppeteer_tools, *apple_tools]
        tool_dict = {**puppeteer_tool_dict, **apple_tool_dict}
        
        # Create model
        model = OpenAIModel('gpt-4o')
        
        # Create tools list
        gui_tool_list = [
            screenshot, mouse_move, mouse_click, keyboard_type, key_press,
            get_screen_size, get_mouse_position, switch_window, focus_application
        ]
        all_tools = [*gui_tool_list, *mcp_tools]
        
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
        
        # Initialize dependencies with both sessions
        deps = CombinedDeps(
            mcp_session=session,
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
            _session=session,
            _stdio_ctx=stdio_ctx,
            _apple_session=apple_session,
            _apple_stdio_ctx=apple_stdio_ctx
        )
    
    async def cleanup(self):
        """Clean up MCP servers and stdio contexts."""
        # Clean up main MCP session
        if self._session:
            try:
                await asyncio.wait_for(self._session.__aexit__(None, None, None), timeout=5.0)
            except (asyncio.TimeoutError, Exception) as e:
                logging.warning(f"Session cleanup warning: {str(e)}")
        
        # Clean up main stdio context
        if self._stdio_ctx:
            try:
                await asyncio.wait_for(self._stdio_ctx.__aexit__(None, None, None), timeout=5.0)
            except (asyncio.TimeoutError, Exception) as e:
                logging.warning(f"STDIO cleanup warning: {str(e)}")
        
        # Clean up Apple MCP session
        if self._apple_session:
            try:
                await asyncio.wait_for(self._apple_session.__aexit__(None, None, None), timeout=5.0)
            except (asyncio.TimeoutError, Exception) as e:
                logging.warning(f"Apple session cleanup warning: {str(e)}")
        
        # Clean up Apple stdio context
        if self._apple_stdio_ctx:
            try:
                await asyncio.wait_for(self._apple_stdio_ctx.__aexit__(None, None, None), timeout=5.0)
            except (asyncio.TimeoutError, Exception) as e:
                logging.warning(f"Apple STDIO cleanup warning: {str(e)}")
    
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