from __future__ import annotations

import os
import asyncio
import logging
import argparse
from dataclasses import dataclass, field
from typing import List, Optional, Any, Dict
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

import agent

from agent import (
    CombinedDeps,
    ServerConfig,
    ServerConnection,
    PUPPETEER_SERVER,
    GIT_SERVER,
    get_filesystem_server_config,
    initialize_server,
    shutdown_all_servers
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

def configure_logging():
    """Configure logging based on command line arguments."""
    parser = argparse.ArgumentParser(description="Agent Manager")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")
    args, _ = parser.parse_known_args()
    
    if args.debug:
        logging.basicConfig(
            level=logging.DEBUG,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
    else:
        logging.basicConfig(level=logging.ERROR)
        
    for logger_name in logging.root.manager.loggerDict:
        logging.getLogger(logger_name).setLevel(logging.DEBUG if args.debug else logging.ERROR)

# Call this function early in the script execution
configure_logging()

@dataclass
class AgentManager:
    """Manages the agent, its tools, and conversation history."""
    agent: Agent
    deps: CombinedDeps
    message_history: List[ModelMessage]
    gui_tools: List[str]
    mcp_tools: List[str]
    # Store server connections
    servers: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    async def initialize(cls) -> AgentManager:
        """Initialize the agent manager with all necessary components."""
        # Add this line near the beginning of the method to respect parent logging level
        root_logger = logging.getLogger()
        log_level = root_logger.level
        
        # Override any other logging configuration in imported modules
        for logger_name in logging.root.manager.loggerDict:
            logging.getLogger(logger_name).setLevel(log_level)
        
        # Define the servers to initialize
        server_configs = [
            PUPPETEER_SERVER,
            GIT_SERVER,
            get_filesystem_server_config(),
            # Add more servers here as needed
        ]
        
        # Initialize all servers and collect their connections
        servers = {}
        all_tools = []
        all_tool_dict = {}
        
        for config in server_configs:
            server_conn, error = await initialize_server(config)
            if server_conn:
                servers[config.name] = server_conn
                all_tools.extend(server_conn.tools)
                all_tool_dict.update(server_conn.tool_dict)
        
        # Create model
        model = OpenAIModel('gpt-4o')
        
        # Create GUI tools list
        gui_tool_list = [
            screenshot, mouse_move, mouse_click, keyboard_type, key_press,
            get_screen_size, get_mouse_position, switch_window, focus_application
        ]
        
        # Combine all tools
        all_tools = [*gui_tool_list, *all_tools]
        
        # Create agent
        agent_instance = Agent(
            model,
            deps_type=CombinedDeps,
            retries=1,
            tools=all_tools
        )
        
        # Fix MCP tool schemas
        for tool_name, tool_info in all_tool_dict.items():
            agent_instance._function_tools[tool_name]._parameters_json_schema = tool_info['schema']
            agent_instance._function_tools[tool_name].description = tool_info['description']
        
        # Initialize dependencies with all sessions
        deps = CombinedDeps(
            server_map={name: conn for name, conn in servers.items()}
        )
        
        # Get tool names for display
        gui_tool_names = ["screenshot", "mouse_move", "mouse_click", "keyboard_type", "key_press", 
                       "get_screen_size", "get_mouse_position", "switch_window", "focus_application"]
        mcp_tool_names = list(all_tool_dict.keys())
        
        # Create screenshots directory
        os.makedirs(os.path.join(os.getcwd(), "screenshots"), exist_ok=True)
        
        return cls(
            agent=agent_instance,
            deps=deps,
            message_history=[],
            gui_tools=gui_tool_names,
            mcp_tools=mcp_tool_names,
            servers=servers
        )
    
    async def cleanup(self):
        """Clean up all MCP servers and stdio contexts."""
        try:
            if self.servers:
                logging.debug(f"Starting cleanup of {len(self.servers)} servers...")
                await agent.shutdown_all_servers(self.servers)
                self.servers.clear()
            logging.debug("Cleanup complete")
        except Exception as e:
            logging.error(f"Error during cleanup: {str(e)}")
    
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