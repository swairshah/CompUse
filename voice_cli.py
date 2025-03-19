"""
CLI interface for voice commands in CompUse.

This module provides a command-line interface for using voice commands
with CompUse, leveraging the Pipecat framework for speech recognition.
"""

import os
import asyncio
import argparse
import logging
from typing import Optional
from datetime import datetime

from rich.console import Console
from rich.panel import Panel
from rich.theme import Theme
from rich.table import Table
from dotenv import load_dotenv

from agent_manager import AgentManager
from voice_tools import VoiceCommandManager

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Setup Rich console with custom theme
custom_theme = Theme({
    "info": "grey70",
    "warning": "yellow",
    "error": "red",
    "success": "green",
    "command": "bold blue",
    "highlight": "dark_orange3",
    "voice": "bold magenta",
})

console = Console(theme=custom_theme)

class VoiceCLI:
    """Handles CLI interaction with voice commands."""
    
    def __init__(self):
        self.console = Console(theme=custom_theme)
        self.agent_manager = None
        self.voice_manager = None
        self.is_listening = False
        
    async def initialize(self):
        """Initialize the CLI with agent and voice managers."""
        try:
            # Initialize agent manager
            with self.console.status("[dark_orange3]Initializing agent...", spinner="dots"):
                self.agent_manager = await AgentManager.initialize()
                
            self.console.print("[success]Agent initialized successfully![/success]")
            
            # Initialize voice command manager
            with self.console.status("[dark_orange3]Initializing voice recognition...", spinner="dots"):
                self.voice_manager = VoiceCommandManager(
                    wake_word=os.getenv("COMPUSE_WAKE_WORD", "computer"),
                    feedback_enabled=True
                )
                await self.voice_manager.initialize(self.process_voice_command)
                
            self.console.print("[success]Voice recognition initialized successfully![/success]")
            
            return True
        except Exception as e:
            self.console.print(f"[error]Error initializing: {str(e)}[/error]")
            return False
    
    async def process_voice_command(self, command: str):
        """Process a voice command by sending it to the agent."""
        if not command:
            return
            
        self.console.print(f"[voice]Voice command:[/voice] {command}")
        
        try:
            # Process the command through the agent manager
            result, elapsed = await self.agent_manager.run_command(command)
            
            # Display the result
            self.console.print(f"[success]Response ({elapsed:.2f}s):[/success]")
            self.console.print(f"{result}")
        except Exception as e:
            self.console.print(f"[error]Error processing command: {str(e)}[/error]")
    
    async def start_voice_recognition(self):
        """Start voice recognition."""
        if self.is_listening:
            self.console.print("[warning]Voice recognition is already active[/warning]")
            return
            
        try:
            with self.console.status("[dark_orange3]Starting voice recognition...", spinner="dots"):
                await self.voice_manager.start_listening()
                
            self.is_listening = True
            self.console.print(
                Panel.fit(
                    f"[voice]Voice recognition active[/voice]\n"
                    f"[info]Wake word: [bold]{os.getenv('COMPUSE_WAKE_WORD', 'computer')}[/bold][/info]\n"
                    f"[info]Say '[bold]{os.getenv('COMPUSE_WAKE_WORD', 'computer')} stop listening[/bold]' to deactivate[/info]",
                    border_style="voice"
                )
            )
        except Exception as e:
            self.console.print(f"[error]Error starting voice recognition: {str(e)}[/error]")
    
    async def stop_voice_recognition(self):
        """Stop voice recognition."""
        if not self.is_listening:
            self.console.print("[warning]Voice recognition is not active[/warning]")
            return
            
        try:
            with self.console.status("[dark_orange3]Stopping voice recognition...", spinner="dots"):
                await self.voice_manager.stop_listening()
                
            self.is_listening = False
            self.console.print("[info]Voice recognition deactivated[/info]")
        except Exception as e:
            self.console.print(f"[error]Error stopping voice recognition: {str(e)}[/error]")
    
    async def show_command_history(self):
        """Display the voice command history."""
        if not self.voice_manager:
            self.console.print("[error]Voice manager not initialized[/error]")
            return
            
        history = self.voice_manager.get_command_history()
        
        if not history:
            self.console.print("[info]No voice commands have been recorded yet[/info]")
            return
            
        history_table = Table(title="Voice Command History")
        history_table.add_column("Time", style="info")
        history_table.add_column("Command", style="voice")
        
        for entry in history:
            timestamp = entry.get("timestamp", 0)
            time_str = datetime.fromtimestamp(timestamp).strftime("%H:%M:%S")
            command = entry.get("command", "")
            history_table.add_row(time_str, command)
            
        self.console.print(history_table)
    
    async def cleanup(self):
        """Clean up resources."""
        try:
            # Clean up voice manager
            if self.voice_manager:
                with self.console.status("[dark_orange3]Cleaning up voice recognition...", spinner="dots"):
                    await self.voice_manager.cleanup()
                    
            # Clean up agent manager
            if self.agent_manager:
                with self.console.status("[dark_orange3]Cleaning up agent...", spinner="dots"):
                    await self.agent_manager.cleanup()
                    
            self.console.print("[success]Resources cleaned up successfully[/success]")
        except Exception as e:
            self.console.print(f"[error]Error during cleanup: {str(e)}[/error]")

async def run_voice_cli():
    """Main function to run the voice CLI."""
    load_dotenv()
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="CompUse Voice Command Interface")
    parser.add_argument(
        "--wake-word", 
        default=os.getenv("COMPUSE_WAKE_WORD", "computer"),
        help="Wake word to activate voice commands (default: 'computer')"
    )
    parser.add_argument(
        "--auto-start", 
        action="store_true",
        help="Automatically start voice recognition on startup"
    )
    parser.add_argument(
        "--push-to-talk",
        action="store_true",
        help="Use push-to-talk mode instead of wake word (press Ctrl+Space to talk)"
    )
    args = parser.parse_args()
    
    # Set wake word in environment
    os.environ["COMPUSE_WAKE_WORD"] = args.wake_word
    
    # Initialize CLI
    cli = VoiceCLI()
    
    # Display welcome banner
    cli.console.print(Panel.fit(
        "[grey70]CompUse Voice Command Interface[/grey70]\n"
        "[grey70]Control your computer with voice commands[/grey70]",
        border_style="grey70"
    ))
    
    try:
        # Initialize CLI components
        success = await cli.initialize()
        if not success:
            return
        
        # Display available tools
        cli.console.print(cli.agent_manager.get_tools_table())
        
        # Display voice command instructions
        if args.push_to_talk:
            cli.console.print(
                Panel.fit(
                    f"[voice]Voice Command Instructions (Push-to-Talk Mode)[/voice]\n"
                    f"[info]Press [bold]Ctrl+Space[/bold] to start recording, release to process command[/info]\n"
                    f"[info]Example command: [bold]take a screenshot[/bold][/info]\n"
                    f"[info]Example command: [bold]click at 500 300[/bold][/info]",
                    border_style="voice"
                )
            )
        else:
            cli.console.print(
                Panel.fit(
                    f"[voice]Voice Command Instructions[/voice]\n"
                    f"[info]Wake word: [bold]{args.wake_word}[/bold][/info]\n"
                    f"[info]Example: '[bold]{args.wake_word} take a screenshot[/bold]'[/info]\n"
                    f"[info]Example: '[bold]{args.wake_word} click at 500 300[/bold]'[/info]",
                    border_style="voice"
                )
            )
        
        # Auto-start voice recognition if requested
        if args.auto_start:
            await cli.start_voice_recognition()
        else:
            cli.console.print("[info]Type [bold]start[/bold] to activate voice recognition or [bold]exit[/bold] to quit[/info]")
        
        # Main command loop
        while True:
            try:
                command = input("\033[1;35m > \033[0m").strip().lower()
                
                if command in ["exit", "quit", "bye"]:
                    break
                elif command == "start":
                    await cli.start_voice_recognition()
                elif command == "stop":
                    await cli.stop_voice_recognition()
                elif command == "status":
                    status = "active" if cli.is_listening else "inactive"
                    cli.console.print(f"[info]Voice recognition is [bold]{status}[/bold][/info]")
                elif command == "history":
                    await cli.show_command_history()
                elif command == "help":
                    cli.console.print(
                        Panel.fit(
                            "[voice]Available Commands[/voice]\n"
                            "[info]start - Start voice recognition[/info]\n"
                            "[info]stop - Stop voice recognition[/info]\n"
                            "[info]status - Check voice recognition status[/info]\n"
                            "[info]history - Show voice command history[/info]\n"
                            "[info]help - Show this help message[/info]\n"
                            "[info]exit - Exit the application[/info]",
                            border_style="voice"
                        )
                    )
                else:
                    cli.console.print("[warning]Unknown command. Type [bold]help[/bold] for available commands[/warning]")
                    
            except KeyboardInterrupt:
                break
            except Exception as e:
                cli.console.print(f"[error]Error: {str(e)}[/error]")
    
    finally:
        # Clean up resources
        await cli.cleanup()
        cli.console.print("[info]Goodbye![/info]")

def main():
    """Entry point for the voice CLI."""
    asyncio.run(run_voice_cli())

if __name__ == "__main__":
    main()