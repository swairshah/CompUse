from __future__ import annotations

import os
import asyncio
import logging
import argparse
from typing import Optional, Dict, Any
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

from cli import CLI
from agent_manager import AgentManager
from voice_tools import AsyncVoiceCommandManager, VoiceCommandConfig
from dotenv import load_dotenv

# Add this before any other code execution (right after imports)
parser = argparse.ArgumentParser()
parser.add_argument('-v', '--verbose', action='store_true', help='Enable verbose logging')
parser.add_argument('--no-voice', action='store_true', help='Disable voice commands')
parser.add_argument('--push-to-talk', action='store_true', help='Use push-to-talk instead of continuous listening')
parser.add_argument('--no-wake-word', action='store_true', help='Disable wake word detection')
parser.add_argument('--wake-word', type=str, default='computer', help='Set custom wake word (default: "computer")')
args = parser.parse_args()

# Configure logging
logging.basicConfig(
    level=logging.INFO if args.verbose else logging.WARNING,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# Setup Rich console with custom theme (same as in cli.py)
custom_theme = Theme({
    "info": "grey70",
    "warning": "yellow",
    "error": "red",
    "success": "grey74",
    "command": "bold blue",
    "highlight": "dark_orange3",
})

console = Console(theme=custom_theme)

class VoiceCLI(CLI):
    """Extends the CLI with voice command capabilities."""
    
    def __init__(self):
        super().__init__()
        self.console = Console(theme=custom_theme)
        self.commands.extend(["/voice", "/voice:on", "/voice:off", "/voice:config"])
        self.voice_manager = None
        self.voice_enabled = not args.no_voice
        
        # Configure voice recognition
        self.voice_config = VoiceCommandConfig(
            continuous_listening=not args.push_to_talk,
            use_wake_word=not args.no_wake_word,
            wake_word=args.wake_word
        )
    
    def display_help(self):
        """Display help information with voice commands."""
        help_table = Table(title="CompUse CLI Commands")
        help_table.add_column("Command", style="command")
        help_table.add_column("Description", style="info")
        
        help_table.add_row("/help", "Display this help message")
        help_table.add_row("/tools", "List all available tools")
        help_table.add_row("/history", "Show conversation history")
        help_table.add_row("/clear", "Clear the screen")
        help_table.add_row("/reset", "Reset the conversation history")
        help_table.add_row("/voice", "Show voice command status")
        help_table.add_row("/voice:on", "Enable voice commands")
        help_table.add_row("/voice:off", "Disable voice commands")
        help_table.add_row("/voice:config", "Show voice command configuration")
        help_table.add_row("/exit, /quit, /bye", "Exit the application")
        
        self.console.print(help_table)
    
    def display_voice_status(self):
        """Display voice command status."""
        status_table = Table(title="Voice Command Status")
        status_table.add_column("Setting", style="command")
        status_table.add_column("Value", style="info")
        
        status_table.add_row("Enabled", "✅ Yes" if self.voice_enabled else "❌ No")
        if self.voice_enabled:
            status_table.add_row("Listening Mode", 
                                "Continuous" if self.voice_config.continuous_listening else 
                                f"Push-to-talk ({self.voice_config.push_to_talk_key})")
            status_table.add_row("Wake Word", 
                                f"'{self.voice_config.wake_word}'" if self.voice_config.use_wake_word else "Disabled")
            status_table.add_row("Language", self.voice_config.language)
        
        self.console.print(status_table)
    
    def display_voice_config(self):
        """Display detailed voice command configuration."""
        config_table = Table(title="Voice Command Configuration")
        config_table.add_column("Setting", style="command")
        config_table.add_column("Value", style="info")
        
        # Add all config parameters
        for key, value in self.voice_config.__dict__.items():
            config_table.add_row(key, str(value))
        
        self.console.print(config_table)


async def run_voice_cli():
    """Main CLI function with voice command support."""
    load_dotenv()
    
    # Initialize CLI
    cli = VoiceCLI()
    cli.setup_history()
    
    # Display welcome banner
    cli.console.print(Panel.fit(
        "[grey70]CompUse Voice CLI[/grey70]\n"
        "[grey70]Desktop & Browser Automation Assistant with Voice Control[/grey70]",
        border_style="grey70"
    ))
    
    agent_manager = None
    voice_manager = None
    
    try:
        # Initialize agent manager
        with cli.console.status("[dark_orange3]Initializing agent...", spinner="dots"):
            agent_manager = await AgentManager.initialize()
        
        # Initialize voice command manager if enabled
        if cli.voice_enabled:
            with cli.console.status("[dark_orange3]Initializing voice recognition...", spinner="dots"):
                voice_manager = AsyncVoiceCommandManager(cli.voice_config)
                await voice_manager.start()
                cli.voice_manager = voice_manager
        
        # Display initial information
        cli.console.print(agent_manager.get_tools_table())
        cli.console.print("[success]Combined Agent initialized with both PyAutoGUI and MCP tools![/success]")
        
        if cli.voice_enabled:
            cli.console.print("[success]Voice command interface activated![/success]")
            if cli.voice_config.use_wake_word:
                cli.console.print(f"[info]Say '[bold]{cli.voice_config.wake_word}[/bold]' followed by your command.[/info]")
            else:
                cli.console.print("[info]Voice commands are active. Just speak your command.[/info]")
        
        cli.console.print("[info]You can also type commands. Type [bold]/help[/bold] for available commands or [bold]/exit[/bold] to quit.[/info]")
        
        # Main loop
        while True:
            try:
                # Check for voice commands if enabled
                voice_command = None
                if cli.voice_enabled and voice_manager:
                    # Use non-blocking check for voice commands
                    voice_command = voice_manager.get_next_command_nowait()
                
                if voice_command:
                    # Process voice command
                    cli.console.print(f"[bold cyan]Voice command:[/bold cyan] {voice_command}")
                    
                    # Check if it's a CLI command
                    if voice_command.lower().startswith(("exit", "quit", "bye")):
                        cli.console.print("[info]Exit command received. Shutting down...[/info]")
                        break
                    
                    # Process with agent
                    with cli.console.status("[dark_orange3] ", spinner="point") as status:
                        result, elapsed = await agent_manager.run_command(voice_command)
                        await asyncio.sleep(0.5)
                    
                    # Display result
                    cli.console.print(f"{result}")
                    
                # Check for text input (non-blocking)
                await asyncio.sleep(0.1)  # Small delay to prevent CPU hogging
                
                # Save history before each command
                try:
                    readline.write_history_file(cli.history_file)
                except (PermissionError, OSError):
                    pass
                
                # Use a prompt that doesn't block the event loop
                print()
                prompt = "\033[1;32m > \033[0m"
                
                # This is a blocking call, but we need it for text input
                # In a more sophisticated implementation, we could use asyncio.create_subprocess_exec
                # to run a separate process for input, but that's beyond the scope of this example
                user_input = input(prompt)
                
                if not user_input:
                    continue
                
                if user_input.lower() in ['/exit', '/quit', '/bye', 'exit', 'quit', 'bye']:
                    cli.console.print("[info]Shutting down...[/info]")
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
                
                elif user_input.lower() == '/voice':
                    cli.display_voice_status()
                    continue
                
                elif user_input.lower() == '/voice:config':
                    cli.display_voice_config()
                    continue
                
                elif user_input.lower() == '/voice:on':
                    if not cli.voice_enabled:
                        cli.voice_enabled = True
                        if not voice_manager:
                            voice_manager = AsyncVoiceCommandManager(cli.voice_config)
                        await voice_manager.start()
                        cli.voice_manager = voice_manager
                        cli.console.print("[success]Voice commands enabled[/success]")
                    else:
                        cli.console.print("[info]Voice commands are already enabled[/info]")
                    continue
                
                elif user_input.lower() == '/voice:off':
                    if cli.voice_enabled:
                        cli.voice_enabled = False
                        if voice_manager:
                            await voice_manager.stop()
                        cli.console.print("[success]Voice commands disabled[/success]")
                    else:
                        cli.console.print("[info]Voice commands are already disabled[/info]")
                    continue
                
                elif user_input.startswith('/'):
                    cli.console.print(f"[error]Command not found: {user_input}[/error]")
                    cli.console.print("[info]Type [bold]/help[/bold] to see available commands[/info]")
                    continue
                
                # Process regular commands
                with cli.console.status("[dark_orange3] ", spinner="point") as status:
                    result, elapsed = await agent_manager.run_command(user_input)
                    await asyncio.sleep(0.5)
                
                # Display result
                cli.console.print(f"{result}")
                
            except KeyboardInterrupt:
                break
            except Exception as e:
                cli.console.print(f"[error]Error: {str(e)}[/error]")
    
    finally:
        # Clean up voice manager
        if voice_manager:
            with cli.console.status("[dark_orange3]Stopping voice recognition...", spinner="dots"):
                await voice_manager.stop()
        
        # Clean up agent manager
        if agent_manager:
            with cli.console.status("[dark_orange3]Cleaning up resources...", spinner="dots"):
                await agent_manager.cleanup()
                cli.console.print("[success]Resources cleaned up successfully[/success]")
        
        # Save history
        try:
            readline.write_history_file(cli.history_file)
        except (PermissionError, OSError, Exception):
            pass
        
        cli.console.print("[info]Goodbye![/info]")


def main():
    """Entry point for the voice CLI."""
    asyncio.run(run_voice_cli())


if __name__ == "__main__":
    main()