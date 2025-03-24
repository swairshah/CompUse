from __future__ import annotations

import os
import asyncio
import logging
import argparse
import readline
import sounddevice as sd
import wavio
import numpy as np
import torch
from rich.console import Console
from rich.panel import Panel
from rich.theme import Theme
from rich.table import Table
from openai import AsyncOpenAI
from silero_vad import load_silero_vad
import aioconsole
from agent import configure_logging
from agent_manager import AgentManager
from dotenv import load_dotenv

console = Console(theme=Theme({
    "info": "grey70",
    "warning": "yellow",
    "error": "red",
    "success": "grey74",
    "command": "bold blue",
    "highlight": "dark_orange3",
}))

print_queue = asyncio.Queue()

async def logger():
    prompt = "\033[1;32m > \033[0m"
    while True:
        message = await print_queue.get()
        if message is None:
            break
        # Special marker to show prompt
        if message == "__SHOW_PROMPT__":
            console.file.write(prompt)
            console.file.flush()
            continue
        # Clear current line and move to the beginning before printing
        console.file.write("\r\033[K")  # Clear current line
        console.print(message)
        console.file.flush()

async def safe_print(message):
    await print_queue.put(message)
    # Signal that a prompt should be displayed after this message
    await print_queue.put("__SHOW_PROMPT__")

client = AsyncOpenAI(api_key=os.environ['OPENAI_API_KEY'])

SAMPLE_RATE = 16000
CHANNELS = 1
RECORD_SECONDS = 5

audio_queue = asyncio.Queue()
vad_model = load_silero_vad()


def is_too_quiet(audio_np, rms_threshold=0.005):
    return np.sqrt(np.mean(audio_np ** 2)) < rms_threshold


def is_voice_present(audio_np, sample_rate=16000, frame_size=512, threshold=0.9):
    audio_tensor = torch.from_numpy(audio_np).float()
    num_frames = len(audio_tensor) // frame_size
    chunks = audio_tensor[:num_frames * frame_size].reshape(num_frames, frame_size)
    with torch.no_grad():
        probs = vad_model(chunks, sample_rate)
    return (probs > threshold).any().item()

async def audio_stream():
    loop = asyncio.get_event_loop()
    while True:
        audio = await loop.run_in_executor(
            None, sd.rec, int(RECORD_SECONDS * SAMPLE_RATE), SAMPLE_RATE, CHANNELS, 'int16'
        )

        await loop.run_in_executor(None, sd.wait)
        audio_np = audio.flatten().astype(np.float32) / 32768.0
        if is_too_quiet(audio_np):
            continue
        if is_voice_present(audio_np):
            temp_file = "temp_audio.wav"
            await loop.run_in_executor(None, lambda: wavio.write(temp_file, audio, SAMPLE_RATE, sampwidth=2))
            await audio_queue.put(temp_file)

async def process_voice_input(agent_manager):
    while True:
        audio_file = await audio_queue.get()
        with open(audio_file, "rb") as audio:
            response = await client.audio.transcriptions.create(
                model="whisper-1", language="en", file=audio
            )
        voice_text = response.text
        await safe_print(f"[highlight] > [/] {voice_text}")
        result, _ = await agent_manager.run_command(voice_text)
        await safe_print(result)


async def process_text_input(agent_manager, cli):
    loop = asyncio.get_event_loop()

    while True:
        # Prompt will be handled by the logger
        user_input = await loop.run_in_executor(None, lambda: input(""))
        if user_input.lower() in ['/exit', '/quit', '/bye']:
            await safe_print("[info]Shutting down...[/info]")
            # Raise CancelledError to trigger clean shutdown instead of using os._exit
            raise asyncio.CancelledError()
        elif user_input.lower() == '/help':
            help_text = cli.get_help_text()
            await safe_print(help_text)
        elif user_input.lower() == '/clear':
            console.clear()
        elif user_input.lower() == '/reset':
            agent_manager.reset_history()
            await safe_print("[success]History reset[/success]")
        elif user_input.lower() == '/tools':
            tools_text = agent_manager.get_tools_table()
            await safe_print(tools_text)
        elif user_input.lower() == '/history':
            history_text = agent_manager.get_history_table()
            await safe_print(history_text)
        elif user_input.startswith('/'):
            await safe_print(f"[error]Unknown command: {user_input}[/error]")
        else:
            result, _ = await agent_manager.run_command(user_input)
            await safe_print(result)
class CLI:
    commands = ["/help", "/tools", "/history", "/clear", "/reset", "/exit", "/quit", "/bye"]

    def display_help(self):
        table = Table(title="Commands")
        table.add_column("Command", style="command")
        table.add_column("Description", style="info")
        table.add_row("/help", "Show help")
        table.add_row("/tools", "Show available tools")
        table.add_row("/history", "Show conversation history")
        table.add_row("/clear", "Clear screen")
        table.add_row("/reset", "Reset conversation")
        table.add_row("/exit", "Exit")
        console.print(table)
        
    def get_help_text(self):
        table = Table(title="Commands")
        table.add_column("Command", style="command")
        table.add_column("Description", style="info")
        table.add_row("/help", "Show help")
        table.add_row("/tools", "Show available tools")
        table.add_row("/history", "Show conversation history")
        table.add_row("/clear", "Clear screen")
        table.add_row("/reset", "Reset conversation")
        table.add_row("/exit", "Exit")
        return table

async def run_cli():
    load_dotenv()
    cli = CLI()
    configure_logging(False)
    agent_manager = await AgentManager.initialize()

    await safe_print(Panel.fit("[highlight]CLI with Audio Initialized![/highlight]"))
    # Initial prompt will be added by safe_print's __SHOW_PROMPT__ signal

    log_task = asyncio.create_task(logger())
    tasks = []

    try:
        # Create tasks so we can cancel them properly
        audio_task = asyncio.create_task(audio_stream())
        voice_task = asyncio.create_task(process_voice_input(agent_manager))
        text_task = asyncio.create_task(process_text_input(agent_manager, cli))
        
        tasks = [audio_task, voice_task, text_task]
        
        # Wait for any task to complete (which shouldn't happen unless there's an error)
        await asyncio.gather(*tasks)
    except asyncio.CancelledError:
        pass
    except KeyboardInterrupt:
        pass
    finally:
        # Cancel all running tasks with a timeout
        for task in tasks:
            if not task.done():
                task.cancel()
        
        # Clean up with timeout
        try:
            await print_queue.put(None)  # Close logger
            await asyncio.wait_for(log_task, timeout=1.0)
            await asyncio.wait_for(agent_manager.cleanup(), timeout=2.0)
            console.print("[success]Resources cleaned up successfully[/success]")
        except asyncio.TimeoutError:
            console.print("[warning]Cleanup timed out, forcing exit[/warning]")
        except Exception as e:
            console.print(f"[error]Error during cleanup: {e}[/error]")


if __name__ == "__main__":
    try:
        asyncio.run(run_cli())
    except KeyboardInterrupt:
        print("\nExiting due to keyboard interrupt")
    except Exception as e:
        print(f"\nError: {e}")

