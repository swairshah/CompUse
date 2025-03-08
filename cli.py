from __future__ import annotations

import os
import asyncio
import logging
import argparse
from typing import Dict, List, Optional, Any
import readline
from pathlib import Path
from dataclasses import dataclass

from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress
from rich.syntax import Syntax
from rich.prompt import Prompt
from rich import print as rprint
from rich.theme import Theme
from rich.table import Table

from pydantic_ai import Agent, Tool
from agent import (
    configure_logging
)
from agent_manager import AgentManager
from dotenv import load_dotenv

from pydantic_ai.models.openai import OpenAIModel


# Add this before any other code execution (right after imports)
parser = argparse.ArgumentParser()
parser.add_argument('-v', '--verbose', action='store_true', help='Enable verbose logging')
args = parser.parse_args()

# Configure logging for both cli.py and agent.py
configure_logging(args.verbose)

# Setup Rich console with custom theme
custom_theme = Theme({
    "info": "grey70",
    "warning": "yellow",
    "error": "red",
    "success": "grey74",
    "command": "bold blue",
    "highlight": "dark_orange3",
})

console = Console(theme=custom_theme)

class CLI:
    """Handles CLI interaction and display."""
    
    def __init__(self):
        self.console = Console(theme=custom_theme)
        self.history_file = os.path.join(os.getcwd(), ".compuse_history")
        self.commands = ["/help", "/tools", "/history", "/clear", "/reset", "/exit", "/quit", "/bye"]
        
    def setup_history(self):
        """Setup command history and autocompletion."""
        try:
            if not os.path.exists(self.history_file):
                Path(self.history_file).touch()
            
            readline.read_history_file(self.history_file)
            readline.set_history_length(1000)
        except (FileNotFoundError, PermissionError, OSError) as e:
            self.console.print(f"[warning]Could not access history file: {e}[/warning]")
            self.console.print("[info]Using temporary command history for this session[/info]")
        
        def completer(text, state):
            options = [cmd for cmd in self.commands if cmd.startswith(text)]
            return options[state] if state < len(options) else None

        readline.set_completer(completer)
        readline.parse_and_bind("tab: complete")
    
    def display_help(self):
        """Display help information."""
        help_table = Table(title="CompUse CLI Commands")
        help_table.add_column("Command", style="command")
        help_table.add_column("Description", style="info")
        
        help_table.add_row("/help", "Display this help message")
        help_table.add_row("/tools", "List all available tools")
        help_table.add_row("/history", "Show conversation history")
        help_table.add_row("/clear", "Clear the screen")
        help_table.add_row("/reset", "Reset the conversation history")
        help_table.add_row("/exit, /quit, /bye", "Exit the application")
        
        self.console.print(help_table)

async def run_cli():
    """Main CLI function to set up a server with both GUI and MCP tools."""
    load_dotenv()
    
    # Initialize CLI
    cli = CLI()
    cli.setup_history()
    
    # Display welcome banner
    cli.console.print(Panel.fit(
        "[grey70]CompUse CLI[/grey70]\n"
        "[grey70]Desktop & Browser Automation Assistant[/grey70]",
        border_style="grey70"
    ))
    
    agent_manager = None
    try:
        # Initialize agent manager
        with cli.console.status("[dark_orange3]Initializing...", spinner="dots"):
            agent_manager = await AgentManager.initialize()
        
        # Display initial information
        cli.console.print(agent_manager.get_tools_table())
        cli.console.print("[success]Combined Agent initialized with both PyAutoGUI and MCP tools![/success]")
        cli.console.print("[info]You can now give commands to control your computer and browser.[/info]")
        cli.console.print("[info]Type [bold]/help[/bold] for available commands or [bold]/exit[/bold] to quit.[/info]")
        
        while True:
            try:
                # Save history before each command
                try:
                    readline.write_history_file(cli.history_file)
                except (PermissionError, OSError):
                    pass
                
                print()
                prompt = "\033[1;32m > \033[0m"
                user_input = input(prompt)
                
                if user_input.lower() in ['/exit', '/quit', '/bye', 'exit', 'quit', 'bye']:
                    cli.console.print("[info]Shutting down...[/info]")
                    # Add proper cleanup before breaking
                    if agent_manager:
                        await agent_manager.cleanup()
                        cli.console.print("[success]Resources cleaned up successfully[/success]")
                    break
                
                elif user_input.lower() == '/help':
                    cli.display_help()
                    continue
                
                elif user_input.lower() == '/clear':
                    cli.console.clear()
                    continue
                
                elif user_input.lower() == '/reset':
                    agent_manager.reset_history()
                    cli.console.print("[success]Conversation history has been reset[/success]")
                    continue
                
                elif user_input.lower() == '/tools':
                    cli.console.print(agent_manager.get_tools_table())
                    continue
                
                elif user_input.lower() == '/history':
                    cli.console.print(agent_manager.get_history_table())
                    continue
                
                elif user_input.startswith('/'):
                    cli.console.print(f"[error]Command not found: {user_input}[/error]")
                    cli.console.print("[info]Type [bold]/help[/bold] to see available commands[/info]")
                    continue
                
                # Process regular commands
                with cli.console.status("[dark_orange3] ", spinner="point") as status:
                    result, elapsed = await agent_manager.run_command(user_input)
                    # status.update("[bold cyan] Processed.")
                    await asyncio.sleep(0.5)

                # Display result
                # cli.console.print(f"[bold cyan]Response ({elapsed:.2f}s):[/bold cyan]")
                cli.console.print(f"{result}")
                
            except KeyboardInterrupt:
                break
            except Exception as e:
                cli.console.print(f"[error]Error: {str(e)}[/error]")
    
    finally:
        # Save history only
        try:
            readline.write_history_file(cli.history_file)
        except (PermissionError, OSError, Exception):
            pass
        
        cli.console.print("[info]Goodbye![/info]")

def main():
    asyncio.run(run_cli())

if __name__ == "__main__":
    main()
