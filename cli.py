from __future__ import annotations

import os
import asyncio
import logging
from typing import Dict, List
import readline
from pathlib import Path

from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress
from rich.syntax import Syntax
from rich.prompt import Prompt
from rich import print as rprint
from rich.theme import Theme
from rich.table import Table

from pydantic_ai import Agent, Tool
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
from pydantic_ai.models.openai import OpenAIModel
from dotenv import load_dotenv

# Setup Rich console with custom theme
custom_theme = Theme({
    "info": "medium_turquoise",
    "warning": "yellow",
    "error": "red",
    "success": "grey74",
    "command": "bold blue",
    "highlight": "dark_orange3",
})

console = Console(theme=custom_theme)

# Setup command history and autocompletion
# Try to use a local history file instead of one in home directory to avoid permission issues
HISTORY_FILE = os.path.join(os.getcwd(), ".compuse_history")
COMMANDS = ["/help", "/tools", "/history", "/clear", "/reset", "/exit", "/quit", "/bye"]

def setup_history():
    """Setup command history for the CLI."""
    try:
        # Create history file if it doesn't exist
        if not os.path.exists(HISTORY_FILE):
            Path(HISTORY_FILE).touch()
        
        # Load history from file
        readline.read_history_file(HISTORY_FILE)
        # Set history length
        readline.set_history_length(1000)
    except (FileNotFoundError, PermissionError, OSError) as e:
        # If we can't access the history file, use a temporary in-memory history
        console.print(f"[warning]Could not access history file: {e}[/warning]")
        console.print("[info]Using temporary command history for this session[/info]")
        
    # Set up tab completion for commands
    def completer(text, state):
        """Tab completion function for readline."""
        options = [cmd for cmd in COMMANDS if cmd.startswith(text)]
        if state < len(options):
            return options[state]
        else:
            return None

    readline.set_completer(completer)
    readline.parse_and_bind("tab: complete")


def display_help():
    """Display help information."""
    help_table = Table(title="CompUse CLI Commands")
    help_table.add_column("Command", style="command")
    help_table.add_column("Description", style="info")
    
    help_table.add_row("/help", "Display this help message")
    help_table.add_row("/tools", "List all available tools")
    help_table.add_row("/history", "Show command history") 
    help_table.add_row("/clear", "Clear the screen")
    help_table.add_row("/reset", "Reset the conversation history")
    help_table.add_row("/exit, /quit, /bye", "Exit the application")
    
    console.print(help_table)

async def run_cli():
    """Main CLI function to set up a server with both GUI and MCP tools."""
    load_dotenv()
    
    # Setup command history
    setup_history()
    
    # Display welcome banner
    console.print(Panel.fit(
        "[grey70]CompUse CLI[/grey70]\n"
        "[grey70]Desktop & Browser Automation Assistant[/grey70]",
        border_style="grey70"
    ))
    
    try:
        # Use standard Python print instead of rich for progress indications
        print("\033[36mInitializing Puppeteer MCP server...\033[0m", end="", flush=True)
        session, stdio_ctx = await initialize_puppeteer_server()
        print(" \033[32mDone\033[0m")
        
        print("\033[36mLoading MCP tools...\033[0m", end="", flush=True)
        mcp_tools, tool_dict = await load_mcp_tools(session)
        print(" \033[32mDone\033[0m")
        
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
        
        # Initialize conversation history with system message
        system_prompt = (
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
        )
        
        # Initialize message history with system message
        message_history: List[ModelMessage] = []
        
        # Display tool information
        gui_tools = ["screenshot", "mouse_move", "mouse_click", "keyboard_type", "key_press", 
                    "get_screen_size", "get_mouse_position", "switch_window", "focus_application"]
        
        tool_table = Table(title="Available Tools")
        tool_table.add_column("Type", style="highlight")
        tool_table.add_column("Tools", style="info")
        
        tool_table.add_row("GUI Tools", ", ".join(gui_tools))
        tool_table.add_row("MCP Tools", ", ".join(list(tool_dict.keys())))
        
        console.print(tool_table)
        console.print("[success]Combined Agent initialized with both PyAutoGUI and MCP tools![/success]")
        console.print("[info]You can now give commands to control your computer and browser.[/info]")
        console.print("[info]Type [bold]/help[/bold] for available commands or [bold]/exit[/bold] to quit.[/info]")
        
        while True:
            try:
                # Save history before each command
                try:
                    readline.write_history_file(HISTORY_FILE)
                except (PermissionError, OSError):
                    pass  
                
                print()  
                prompt = "\033[1;32m > \033[0m"
                user_input = input(prompt)
                
                if user_input.lower() in ['/exit', '/quit', '/bye', 'exit', 'quit', 'bye']:
                    console.print("[info]Shutting down...[/info]")
                    # Break out of the loop to reach the finally block
                    break
                
                elif user_input.lower() == '/help':
                    display_help()
                    continue
                
                elif user_input.lower() == '/clear':
                    console.clear()
                    continue
                
                elif user_input.lower() == '/reset':
                    # Re-initialize message history
                    message_history = []
                    console.print("[success]Conversation history has been reset[/success]")
                    continue
                
                elif user_input.lower() == '/tools':
                    console.print(tool_table)
                    continue
                
                elif user_input.lower() == '/history':
                    history_table = Table(title="Conversation History")
                    history_table.add_column("Role", style="command")
                    history_table.add_column("Content", style="info")
                    
                    for msg in message_history:
                        if isinstance(msg, ModelRequest):
                            first_part = msg.parts[0]
                            if isinstance(first_part, UserPromptPart):
                                history_table.add_row("User", first_part.content)
                        elif isinstance(msg, ModelResponse):
                            first_part = msg.parts[0]
                            if isinstance(first_part, TextPart):
                                history_table.add_row("Assistant", first_part.content)
                    
                    console.print(history_table)
                    continue
                
                # Handle unknown slash commands
                elif user_input.startswith('/'):
                    console.print(f"[error]Command not found: {user_input}[/error]")
                    console.print("[info]Type [bold]/help[/bold] to see available commands[/info]")
                    continue
                
                # Process regular commands
                with console.status("[dark_orange3]Processing your request...", spinner="dots") as status:
                    start_time = asyncio.get_event_loop().time()
                    
                    # Run the agent with the full conversation history
                    result = await combined_agent.run(
                        user_input,
                        message_history=message_history,
                        deps=deps
                    )
                    
                    # Add new messages to history
                    message_history.extend(result.new_messages())
                    
                    elapsed = asyncio.get_event_loop().time() - start_time
                    status.update("[bold cyan]Processing complete!")
                    await asyncio.sleep(0.5)

                # Display result
                console.print(f"[bold cyan]Response ({elapsed:.2f}s):[/bold cyan]")
                console.print(f"{result.data}")
                
            except KeyboardInterrupt:
                console.print("\n[info]Goodbye![/info]")
                break
            except Exception as e:
                console.print(f"[error]Error: {str(e)}[/error]")
    
    finally:
        # Save history
        try:
            readline.write_history_file(HISTORY_FILE)
        except (PermissionError, OSError, Exception):
            pass  # Continue with cleanup if we can't write history
            
        # Clean up MCP server
        try:
            # First close the session
            if 'session' in locals() and session is not None:
                try:
                    await asyncio.wait_for(session.__aexit__(None, None, None), timeout=5.0)
                except (asyncio.TimeoutError, Exception) as e:
                    console.print(f"[warning]Session cleanup warning: {str(e)}[/warning]")
                    
            # Then close the stdio context
            if 'stdio_ctx' in locals() and stdio_ctx is not None:
                try:
                    await asyncio.wait_for(stdio_ctx.__aexit__(None, None, None), timeout=5.0)
                except (asyncio.TimeoutError, Exception) as e:
                    console.print(f"[warning]STDIO cleanup warning: {str(e)}[/warning]")
                    
            console.print("[success]Resources cleaned up successfully[/success]")
        except Exception as e:
            console.print(f"[error]Error during cleanup: {str(e)}[/error]")
        finally:
            console.print("[info]Goodbye![/info]")


if __name__ == "__main__":
    asyncio.run(run_cli())
