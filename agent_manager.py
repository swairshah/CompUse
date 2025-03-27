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
    initialize_git_server, 
    initialize_filesystem_server, 
    load_mcp_tools, 
    CombinedDeps
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
class AgentManager:
    """Manages the agent, its tools, and conversation history."""
    agent: Agent
    deps: CombinedDeps
    message_history: List[ModelMessage]
    gui_tools: List[str]
    mcp_tools: List[str]
    # Store MCP sessions and contexts
    _puppeteer_session: Optional[Any] = None
    _puppeteer_ctx: Optional[Any] = None
    _git_session: Optional[Any] = None
    _git_ctx: Optional[Any] = None
    _filesystem_session: Optional[Any] = None
    _filesystem_ctx: Optional[Any] = None

    @classmethod
    async def initialize(cls) -> AgentManager:
        """Initialize the agent manager with all necessary components."""
        # Initialize Puppeteer server
        puppeteer_session, puppeteer_ctx = await initialize_puppeteer_server()
        
        # Initialize Git server
        git_session, git_ctx = await initialize_git_server()
        
        # Initialize Filesystem server with proper error handling
        try:
            filesystem_session, filesystem_ctx = await initialize_filesystem_server()
            # Load filesystem tools
            filesystem_tools, filesystem_tool_dict = await load_mcp_tools(filesystem_session)
        except Exception as e:
            logging.error(f"Failed to initialize filesystem server: {str(e)}")
            filesystem_session, filesystem_ctx = None, None
            filesystem_tools, filesystem_tool_dict = [], {}
        
        # Load tools from Puppeteer server
        puppeteer_tools, puppeteer_tool_dict = await load_mcp_tools(puppeteer_session)
        
        # Load tools from Git server
        try:
            git_tools, git_tool_dict = await load_mcp_tools(git_session)
        except Exception as e:
            logging.error(f"Failed to load Git tools: {str(e)}")
            git_tools, git_tool_dict = [], {}
        
        # Combine tool dictionaries - only include non-empty dictionaries
        all_mcp_tool_dict = {**puppeteer_tool_dict}
        if git_tool_dict:
            all_mcp_tool_dict.update(git_tool_dict)
        if filesystem_tool_dict:
            all_mcp_tool_dict.update(filesystem_tool_dict)
        
        # Combine tools
        mcp_tools = puppeteer_tools
        if git_tools:
            mcp_tools += git_tools
        if filesystem_tools:
            mcp_tools += filesystem_tools
        
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
        for tool_name, tool_info in all_mcp_tool_dict.items():
            agent._function_tools[tool_name]._parameters_json_schema = tool_info['schema']
            agent._function_tools[tool_name].description = tool_info['description']
        
        # Initialize dependencies with all sessions
        deps = CombinedDeps(
            mcp_session=puppeteer_session,
            git_session=git_session,
            filesystem_session=filesystem_session
        )
        
        # Get tool names for display
        gui_tool_names = ["screenshot", "mouse_move", "mouse_click", "keyboard_type", "key_press", 
                       "get_screen_size", "get_mouse_position", "switch_window", "focus_application"]
        mcp_tool_names = list(all_mcp_tool_dict.keys())
        
        # Create screenshots directory
        os.makedirs(os.path.join(os.getcwd(), "screenshots"), exist_ok=True)
        
        return cls(
            agent=agent,
            deps=deps,
            message_history=[],
            gui_tools=gui_tool_names,
            mcp_tools=mcp_tool_names,
            _puppeteer_session=puppeteer_session,
            _puppeteer_ctx=puppeteer_ctx,
            _git_session=git_session,
            _git_ctx=git_ctx,
            _filesystem_session=filesystem_session,
            _filesystem_ctx=filesystem_ctx
        )
    
    async def cleanup(self):
        """Clean up all MCP servers and stdio contexts."""
        # Clean up sessions
        for name, session in [
            ("Puppeteer", self._puppeteer_session),
            ("Git", self._git_session),
            ("Filesystem", self._filesystem_session)
        ]:
            if session:
                try:
                    await asyncio.wait_for(session.__aexit__(None, None, None), timeout=5.0)
                except (asyncio.TimeoutError, Exception) as e:
                    logging.warning(f"{name} session cleanup warning: {str(e)}")
        
        # Clean up stdio contexts
        for name, ctx in [
            ("Puppeteer", self._puppeteer_ctx),
            ("Git", self._git_ctx),
            ("Filesystem", self._filesystem_ctx)
        ]:
            if ctx:
                try:
                    await asyncio.wait_for(ctx.__aexit__(None, None, None), timeout=5.0)
                except (asyncio.TimeoutError, Exception) as e:
                    logging.warning(f"{name} stdio cleanup warning: {str(e)}")
    
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