from __future__ import annotations

import os
import sys
import asyncio
import logging
import argparse
from typing import Dict, List, Optional, Any, Union
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
from transcriber import (
    AudioConfig, VADConfig, TranscriberConfig,
    SanitizerConfig, StreamManager, MicrophoneHandler
)

from pydantic_ai.models.openai import OpenAIModel


parser = argparse.ArgumentParser()
parser.add_argument('-v', '--verbose', action='store_true', help='Enable verbose logging')
parser.add_argument('--transcriber', choices=['openai', 'deepgram'], default='openai', help='Transcription provider')
args = parser.parse_args()

stdin_lock = asyncio.Lock()

async def async_input(prompt: str = "") -> str:
    """Thread-safe async input function with lock protection and enhanced reliability"""
    # Use the lock to ensure only one input operation at a time
    async with stdin_lock:
        try:
            # Print the prompt ourselves to ensure it's visible
            if prompt:
                print(prompt, end="", flush=True)
                
            # Run blocking input in a thread pool to avoid blocking the event loop
            # Use "" as prompt since we've already displayed it
            result = await asyncio.get_event_loop().run_in_executor(None, lambda: input(""))
            return result
        except EOFError:
            # Handle EOF (Ctrl+D) gracefully
            return ""
        except Exception as e:
            logging.error(f"Input error: {e}")
            return ""

configure_logging(args.verbose)

custom_theme = Theme({
    "info": "grey70",
    "warning": "yellow",
    "error": "red",
    "success": "grey74",
    "command": "bold blue",
    "highlight": "dark_orange3",
})

console = Console(theme=custom_theme)

class VoiceInputManager:
    """Manages voice input with async streaming capabilities"""
    
    def __init__(self, console: Console):
        self.console = console
        self.stream_manager = None
        self.mic_handler = None
        self.voice_input_queue = asyncio.Queue()
        self.voice_task = None
        self.is_active = False
        
    async def initialize(self):
        """Initialize voice components"""
        transcriber_config = TranscriberConfig(
            provider=args.transcriber,
            openai_model="whisper-1"
        )
        
        sanitizer_config = SanitizerConfig(
            enabled=True,
            model_type="openai",
            openai_model="gpt-4o-mini"
        )
        
        self.stream_manager = StreamManager(transcriber_config, sanitizer_config)
        await self.stream_manager.start()
        
        self.mic_handler = MicrophoneHandler(self.stream_manager)
        
        return self
        
    async def start_listening(self):
        """Start listening for voice input in the background"""
        if self.is_active:
            return
            
        self.is_active = True
        
        if not self.stream_manager or not self.stream_manager.is_active:
            transcriber_config = TranscriberConfig(
                provider=args.transcriber,
                openai_model="whisper-1"
            )
            
            sanitizer_config = SanitizerConfig(
                enabled=True,
                model_type="openai",
                openai_model="gpt-4o-mini"
            )
            
            self.stream_manager = StreamManager(transcriber_config, sanitizer_config)
            await self.stream_manager.start()
            
            if not self.mic_handler:
                self.mic_handler = MicrophoneHandler(self.stream_manager)
        
        import pyaudio
        from transcriber import AudioConfig
        
        # Override the _transcribe_chunk method to capture output
        original_transcribe = self.stream_manager._transcribe_chunk
        
        async def capture_transcribe(file_path):
            """Wrapper to capture transcription results"""
            try:
                transcript_text = await self.stream_manager.transcriber.transcribe(file_path)
                if transcript_text and transcript_text.strip():
                    # Sanitize the text if possible
                    try:
                        sanitized_text = await self.stream_manager.sanitizer.sanitize(transcript_text)
                        if sanitized_text and sanitized_text.strip():
                            await self.voice_input_queue.put(sanitized_text)
                        else:
                            await self.voice_input_queue.put(transcript_text)
                    except:
                        # Fallback to raw transcript if sanitization fails
                        await self.voice_input_queue.put(transcript_text)
            except Exception as e:
                logging.error(f"Error during transcription capture: {e}")
                
        # Replace the method
        self.stream_manager._transcribe_chunk = capture_transcribe
        
        # Create a silent version of start_streaming to avoid console output
        async def silent_streaming():
            """Modified version of start_streaming that doesn't print messages"""
            if self.mic_handler.is_streaming:
                return
                
            p = pyaudio.PyAudio()
            
            stream = p.open(
                format=AudioConfig.format,
                channels=AudioConfig.channels,
                rate=AudioConfig.sample_rate,
                input=True,
                frames_per_buffer=AudioConfig.chunk_size
            )
            
            self.mic_handler.is_streaming = True
            
            try:
                while self.mic_handler.is_streaming:
                    # Read audio data from microphone
                    audio_data = stream.read(AudioConfig.chunk_size, exception_on_overflow=False)
                    # Send to stream manager
                    await self.stream_manager.add_audio_chunk(audio_data)
                    # Small delay to prevent overwhelming the stream
                    await asyncio.sleep(0.01)
                    
            except Exception as e:
                logging.error(f"Error in audio streaming: {e}")
            finally:
                # Cleanup
                stream.stop_stream()
                stream.close()
                p.terminate()
        
        # Start listening in background with our silent version
        self.voice_task = asyncio.create_task(silent_streaming())
        
    async def stop_listening(self):
        """Stop voice input streaming"""
        if not self.is_active:
            return
            
        self.is_active = False
        
        # Stop microphone handler
        if self.mic_handler:
            await self.mic_handler.stop_streaming()
        
        # Cancel voice task first
        if self.voice_task:
            self.voice_task.cancel()
            try:
                await self.voice_task
            except asyncio.CancelledError:
                pass
                
        # Don't close the stream manager, just stop listening
        # We'll reinitialize it if needed later
            
    async def get_next_voice_input(self, timeout: float = 0.1) -> Optional[str]:
        """Get the next voice input if available, with timeout"""
        try:
            return await asyncio.wait_for(self.voice_input_queue.get(), timeout=timeout)
        except asyncio.TimeoutError:
            return None

class CLI:
    """Handles CLI interaction and display."""
    
    def __init__(self):
        self.console = Console(theme=custom_theme)
        self.history_file = os.path.join(os.getcwd(), ".compuse_history")
        self.commands = ["/help", "/tools", "/history", "/clear", "/reset", "/exit", "/quit", "/bye", "/voice", "/dual"]
        self.voice_mode = False
        self.voice_manager = None
        self.dual_input_mode = False
        
        # Create a unified input queue - 
        # all inputs (text and voice) will flow through this
        self.input_queue = asyncio.Queue()
        # Task for voice input processing
        self.voice_input_task = None
        
    async def initialize(self):
        """Initialize CLI components"""
        self.setup_history()
        
        # Initialize voice manager
        self.voice_manager = VoiceInputManager(self.console)
        await self.voice_manager.initialize()
        
        return self
        
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
        help_table.add_row("/voice", f"Toggle voice input mode (currently {'on' if self.voice_mode else 'off'})")
        help_table.add_row("/dual", f"Toggle dual input mode - listen to microphone and text simultaneously (currently {'on' if self.dual_input_mode else 'off'})")
        help_table.add_row("/exit, /quit, /bye", "Exit the application")
        
        self.console.print(help_table)
        
    async def toggle_dual_mode(self):
        """Toggle dual input mode (voice + text simultaneous)"""
        was_enabled = self.dual_input_mode
        self.dual_input_mode = not self.dual_input_mode
        
        if self.dual_input_mode:
            # Redirect stdout temporarily to suppress messages
            import sys
            original_stdout = sys.stdout
            sys.stdout = open(os.devnull, 'w')
            
            try:
                # Start voice listening in background
                await self.voice_manager.start_listening()
            finally:
                # Restore stdout
                sys.stdout.close()
                sys.stdout = original_stdout
                
            self.console.print("[success]Dual input mode enabled - listening silently via microphone[/success]")
        else:
            # Just set the flag - the actual cleanup will happen in the main loop with a full reset
            # This is the key difference - we don't try to clean up here, but let the main loop handle it
            self.console.print("[success]Dual input mode disabled[/success]")
            
            # IMPORTANT: We do NOT stop the voice manager here!
            # The reset logic in the main loop will handle that completely
            # This prevents the double-cleanup issue that was causes to exit
            # on exiting dual mode
            
    async def toggle_dual_mode(self):
        """Toggle dual input mode (voice + text simultaneous) using the input queue approach"""
        self.dual_input_mode = not self.dual_input_mode
        
        if self.dual_input_mode:
            self.console.print("[success]Dual input mode enabled - listening silently via microphone[/success]")
            
            # Set up the voice processing task
            async def voice_input_processor():
                """Process voice input and add it to the main input queue"""
                try:
                    await self.voice_manager.start_listening()
                    
                    is_dual_mode_active = True
                    
                    while is_dual_mode_active:
                        # Check if dual mode is still active - exit loop if not
                        if not self.dual_input_mode:
                            logging.debug("Voice processor detected dual mode disabled")
                            break
                            
                        try:
                            voice_text = await self.voice_manager.get_next_voice_input(timeout=0.2)
                            
                            if not self.dual_input_mode:
                                logging.debug("Dual mode disabled during voice input processing")
                                break
                            
                            if voice_text and voice_text.strip():
                                # We'll print the voice text in the main loop when processing the queue
                                # Add to history for tab completion
                                if voice_text.strip():
                                    readline.add_history(voice_text)
                                    
                                # Add to the input queue for processing - use put_nowait to avoid blocking
                                try:
                                    # Clean up the voice text before adding to queue
                                    clean_voice_text = voice_text.strip()
                                    # Only add to queue if we're still in dual mode
                                    if self.dual_input_mode:
                                        self.input_queue.put_nowait(clean_voice_text)
                                        logging.debug(f"Added voice input to queue: {clean_voice_text}")
                                except asyncio.QueueFull:
                                    logging.warning("Input queue full, discarding voice input")
                                except Exception as e:
                                    logging.error(f"Error adding voice to queue: {e}")
                        except asyncio.TimeoutError:
                            # Normal timeout, just update our active flag and continue
                            is_dual_mode_active = self.dual_input_mode
                        except Exception as e:
                            if self.dual_input_mode:  # Only log if still in dual mode
                                logging.error(f"Voice input error: {e}")
                            await asyncio.sleep(0.1)
                            
                        # Check dual mode state again
                        is_dual_mode_active = self.dual_input_mode
                        
                except Exception as e:
                    logging.error(f"Voice processor error: {e}")
                finally:
                    # Always stop voice when exiting
                    try:
                        await self.voice_manager.stop_listening()
                        logging.debug("Voice processor stopped listening")
                    except Exception as e:
                        logging.error(f"Error stopping voice manager: {e}")
            
            # Start the voice processor task
            self.voice_input_task = asyncio.create_task(voice_input_processor())
            
            # Small delay to ensure the voice task is fully started
            await asyncio.sleep(0.2)
            
        else:
            # Disable dual mode flag first
            self.dual_input_mode = False
            self.console.print("[success]Dual input mode disabled[/success]")
            
            # Small delay to ensure clean transition
            await asyncio.sleep(0.2)
            
            # Stop the voice input task
            if self.voice_input_task and not self.voice_input_task.done():
                self.voice_input_task.cancel()
                try:
                    # Use timeout to avoid waiting indefinitely
                    await asyncio.wait_for(asyncio.shield(self.voice_input_task), timeout=0.5)
                except (asyncio.CancelledError, asyncio.TimeoutError):
                    pass  # Expected cancellation or timeout
                    
                # Set task to None to ensure clean state
                self.voice_input_task = None
            
            # Stop voice listening and drain the queue
            await self.voice_manager.stop_listening()
            
            # Clear any pending items in the input queue
            while not self.input_queue.empty():
                try:
                    self.input_queue.get_nowait()
                except asyncio.QueueEmpty:
                    break
                    
            print("") # Add a newline for clean transition
    
    async def cleanup(self):
        """Clean up resources"""
        # Cancel dual mode if active
        if self.dual_input_mode:
            await self.toggle_dual_mode()
        
        # Cancel any pending voice input task
        if self.voice_input_task and not self.voice_input_task.done():
            self.voice_input_task.cancel()
            try:
                await asyncio.wait_for(self.voice_input_task, timeout=0.5)
            except (asyncio.TimeoutError, asyncio.CancelledError):
                pass  # Expected when cancelled
                
        # Clean up voice manager resources
        if self.voice_manager:
            try:
                await self.voice_manager.stop_listening()
                
                # Fully clean up the stream manager if it exists
                if self.voice_manager.stream_manager and self.voice_manager.stream_manager.is_active:
                    await self.voice_manager.stream_manager.close()
            except Exception as e:
                logging.error(f"Error during voice manager cleanup: {e}")

async def run_cli():
    """Main CLI function to set up a server with both GUI and MCP tools."""
    load_dotenv()
    
    cli = await CLI().initialize()
    
    cli.console.print(Panel.fit(
        "[grey70]CompUse CLI[/grey70]\n"
        "[grey70]Desktop & Browser Automation Assistant[/grey70]",
        border_style="grey70"
    ))
    
    agent_manager = None
    try:
        with cli.console.status("[dark_orange3]Initializing...", spinner="dots"):
            agent_manager = await AgentManager.initialize()
        
        cli.console.print(agent_manager.get_tools_table())
        cli.console.print("[success]Combined Agent initialized with both PyAutoGUI and MCP tools![/success]")
        cli.console.print("[info]You can now give commands to control your computer and browser.[/info]")
        cli.console.print("[info]Type [bold]/help[/bold] for available commands or [bold]/exit[/bold] to quit.[/info]")
        cli.console.print("[info]Type [bold]/dual[/bold] to enable simultaneous voice & text input.[/info]")
        
        main_loop_running = True
        while main_loop_running:
            try:
                # Save history before each command
                try:
                    readline.write_history_file(cli.history_file)
                except (PermissionError, OSError):
                    pass
                
                
                # Initialize user_input
                user_input = None
                
                # Always print a newline to ensure clean prompt separation
                print("") 
                
                # Set prompt - using ANSI color codes for visibility
                prompt = "\033[1;32m > \033[0m"
                
                # Print prompt explicitly (our async_input function will handle it properly)
                print(prompt, end="", flush=True)
                
                # Two ways to get input:
                # 1. Directly from the keyboard (for all modes)
                # 2. From the input queue (for voice input in dual mode)
                
                # Create a task to get keyboard input (no prompt needed since we printed it already)
                keyboard_task = asyncio.create_task(async_input(""))
                
                # Create a task to get from input queue (only used in dual mode)
                queue_task = None
                if cli.dual_input_mode:
                    # Check if anything is in the queue first (non-blocking)
                    try:
                        if not cli.input_queue.empty():
                            queue_task = asyncio.create_task(cli.input_queue.get_nowait())
                        else:
                            queue_task = asyncio.create_task(cli.input_queue.get())
                    except asyncio.QueueEmpty:
                        # Queue was empty when we checked, create a normal wait task
                        queue_task = asyncio.create_task(cli.input_queue.get())
                        
                # Prepare wait tasks - always include keyboard input
                wait_tasks = [keyboard_task]
                if queue_task:  # Include queue task if in dual mode
                    wait_tasks.append(queue_task)
                
                # Wait for either keyboard input or queue input (dual mode)
                done, pending = await asyncio.wait(
                    wait_tasks,
                    return_when=asyncio.FIRST_COMPLETED
                )
                
                for task in pending:
                    task.cancel()
                
                # Wait for cancellations to complete
                if pending:
                    try:
                        await asyncio.wait(pending, timeout=0.1)
                    except Exception:
                        pass  # Ignore cancellation errors
                
                user_input = None
                input_from_voice = False
                
                for task in done:
                    try:
                        result = task.result()
                        if result and isinstance(result, str):  # Skip empty inputs, ensure it's a string
                            user_input = result
                            
                            # Check if this was voice input
                            if queue_task and task == queue_task:
                                input_from_voice = True
                                # Print the voice input to make it visible
                                if not result.startswith("\nVoice:"):  # Avoid double-printing
                                    print(f"\nVoice: {result}")
                                    
                                # For voice input, we need to reset the prompt display
                                # This is critical to avoid the double prompt issue
                                print("")  # Add a blank line after voice response
                            
                            # Add to history if not empty and not from voice
                            if user_input.strip() and not input_from_voice:
                                readline.add_history(user_input)
                                
                            # Debug info
                            logging.debug(f"Got input: '{result}' (from_voice={input_from_voice})")
                            break
                    except Exception as e:
                        cli.console.print(f"[error]Input task error: {e}[/error]")
                        logging.error(f"Input task error: {e}")
                
                # If we didn't get input, try again
                if not user_input:
                    continue
                
                # Handle classic voice mode
                if cli.voice_mode and not cli.dual_input_mode:
                    # If using classic voice mode and user just pressed Enter
                    if not user_input.strip():
                        # Get voice input
                        with cli.console.status("[dark_orange3]Listening...", spinner="toggle7"):
                            user_input = await cli.voice_manager.get_next_voice_input(timeout=10.0)
                        
                        if user_input:
                            print(f"{prompt}{user_input}")
                            # Add to readline history
                            if user_input.strip():
                                readline.add_history(user_input)
                        else:
                            cli.console.print("[warning]No speech detected or transcription failed[/warning]")
                            continue

                if not user_input or not user_input.strip():
                    continue
                    
                if user_input.lower() in ['/exit', '/quit', '/bye', 'exit', 'quit', 'bye']:
                    cli.console.print("[info]Shutting down...[/info]")
                    # Force exit instead of trying to cleanup gracefully
                    # This is a drastic measure but guarantees clean exit
                    import os
                    os._exit(0)
                
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
                    # Can't use voice mode and dual mode at the same time
                    if cli.dual_input_mode:
                        cli.console.print("[warning]Please disable dual input mode first with /dual[/warning]")
                        continue
                        
                    cli.voice_mode = not cli.voice_mode
                    mode_status = "enabled" if cli.voice_mode else "disabled"
                    cli.console.print(f"[success]Voice input mode {mode_status}[/success]")
                    continue
                    
                elif user_input.lower() == '/dual':
                    # Can't use voice mode and dual mode at the same time
                    if cli.voice_mode:
                        cli.console.print("[warning]Please disable voice mode first with /voice[/warning]")
                        continue
                    
                    was_enabled = cli.dual_input_mode
                        
                    # Toggle dual mode
                    await cli.toggle_dual_mode()
                    
                    # No special handling needed with the new input queue approach
                    continue
                
                elif user_input.startswith('/'):
                    cli.console.print(f"[error]Command not found: {user_input}[/error]")
                    cli.console.print("[info]Type [bold]/help[/bold] to see available commands[/info]")
                    continue
                
                # Process regular commands
                try:
                    with cli.console.status("[dark_orange3] ", spinner="point") as status:
                        result, elapsed = await agent_manager.run_command(user_input)
                        await asyncio.sleep(0.5)
    
                    # Display result
                    cli.console.print(f"{result}")
                    
                    # Critical: If the input came from voice, we need to reset the prompt properly
                    if input_from_voice:
                        # Add a newline to ensure the next prompt is properly positioned
                        print("")
                        
                except Exception as e:
                    cli.console.print(f"[error]Error processing command: {str(e)}[/error]")
                    import traceback
                    traceback.print_exc()
                
                # Reset input_from_voice flag after processing to avoid lingering effects
                input_from_voice = False
                
            except KeyboardInterrupt:
                break
            except Exception as e:
                cli.console.print(f"[error]Error: {str(e)}[/error]")
                import traceback
                traceback.print_exc()
                
            # No reset logic needed with the queue-based approach
    
    finally:
        # Clean up CLI resources
        await cli.cleanup()
        
        # Clean up agent manager with proper error handling
        if agent_manager:
            try:
                cli.console.print("[info]Cleaning up resources...[/info]")
                # Don't use a spinner here as it can cause issues during shutdown
                # Instead, just set a reasonable timeout
                try:
                    await asyncio.wait_for(agent_manager.cleanup(), timeout=3.0)
                    cli.console.print("[success]Resources cleaned up successfully[/success]")
                except asyncio.TimeoutError:
                    cli.console.print("[warning]Cleanup timed out, but continuing shutdown[/warning]")
                except Exception as e:
                    cli.console.print(f"[warning]Cleanup error: {e}, but continuing shutdown[/warning]")
            except Exception as e:
                logging.error(f"Error during cleanup: {e}")
        
        # Save history
        try:
            readline.write_history_file(cli.history_file)
        except (PermissionError, OSError, Exception):
            pass
        
        cli.console.print("[info]Goodbye![/info]")

def main():
    """Main entry point with graceful shutdown handling"""
    try:
        # Run with properly configured asyncio and signal handling
        asyncio.run(run_cli())
    except KeyboardInterrupt:
        # Handle Ctrl+C gracefully
        print("\nShutdown requested by keyboard interrupt")
        print("Goodbye!")
    except Exception as e:
        # Log other exceptions but don't display full traceback to user
        logging.error(f"Unexpected error during execution: {e}")
        print(f"\nError: {e}")
        print("Exiting due to error")

if __name__ == "__main__":
    main()
