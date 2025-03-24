from __future__ import annotations

import os
import asyncio
import sounddevice as sd
import wavio
import numpy as np
import torch
from rich.text import Text
from textual.app import App, ComposeResult
from textual.widgets import Input, RichLog, Header, Footer, Static
from textual.containers import VerticalScroll, Vertical
from textual.binding import Binding
from textual.reactive import reactive
from openai import AsyncOpenAI
from silero_vad import load_silero_vad
from agent import configure_logging
from agent_manager import AgentManager
from dotenv import load_dotenv
import pyperclip

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

class TitleBar(Static):
    def render(self) -> str:
        return "🎙️"

class AudioCLI(App):
    CSS = """
    Screen {
        background: #282c34;
        color: #abb2bf;
    }

    RichLog {
        border: round #61afef;
        padding: 1;
        height: 1fr;
        background: #21252b;
    }

    TitleBar {
        text-align: center;
        padding: 1;
        background: #3e4451;
        color: #ffffff;
    }

    Header, Footer {
        background: #21252b;
        color: #61afef;
    }
    """

    BINDINGS = [
        Binding("ctrl+y", action="copy_log", description="Copy last message"),
        Binding("ctrl+c", action="quit", description="Quit")
    ]

    log_history: reactive[list[str]] = reactive([])

    async def on_mount(self):
        load_dotenv()
        configure_logging(False)
        self.agent_manager = await AgentManager.initialize()
        self.audio_task = asyncio.create_task(audio_stream())
        self.voice_task = asyncio.create_task(self.handle_audio())

        input_widget = self.query_one(Input)
        input_widget.focus()  # initially set focus to input

    async def handle_audio(self):
        while True:
            audio_file = await audio_queue.get()
            with open(audio_file, "rb") as audio:
                response = await client.audio.transcriptions.create(
                    model="whisper-1", language="en", file=audio
                )
            voice_text = response.text.strip()
            await self.query(voice_text)

    #async def query(self, text):
    #    log_widget = self.query_one(RichLog)
    #    log_widget.write(f"[bold cyan]You:[/bold cyan] {text}", markup=True)
    #    self.log_history.append(text)
    #    result, _ = await self.agent_manager.run_command(text)
    #    log_widget.write(f"[bold green]AI:[/bold green] {result}", markup=True)
    #    self.log_history.append(result)

    async def query(self, text):
        log_widget = self.query_one(RichLog)
        user_text = Text.assemble(("You: ", "bold cyan"), (text, "white"))
        log_widget.write(user_text)
        self.log_history.append(text)

        result, _ = await self.agent_manager.run_command(text)
        ai_text = Text.assemble(("AI: ", "bold green"), (result, "white"))
        log_widget.write(ai_text)
        self.log_history.append(result)

    def compose(self) -> ComposeResult:
        #with VerticalScroll():
        with Vertical():
            yield RichLog(auto_scroll=True, wrap=True)
            yield Input(placeholder="Type your command here...")
        #yield Footer()


    async def on_input_submitted(self, event: Input.Submitted):
        user_input = event.value
        if user_input.lower() in ['/exit', '/quit', '/bye']:
            await self.action_quit()
            return
        await self.query(user_input)
        event.input.value = ""
        event.input.focus()  # explicitly focus back to input

    async def action_copy_log(self):
        try:
            pyperclip.copy(self.log_history[-1])
            self.notify("Message copied to clipboard!", title="Copied")
        except pyperclip.PyperclipException as exc:
            self.notify(str(exc), title="Clipboard error", severity="error")

    async def action_quit(self):
        if self.audio_task:
            self.audio_task.cancel()
        if self.voice_task:
            self.voice_task.cancel()
        await asyncio.gather(
            *(task for task in [self.audio_task, self.voice_task] if task),
            return_exceptions=True
        )
        if self.agent_manager:
            await self.agent_manager.cleanup()
        self.exit()


if __name__ == "__main__":
    AudioCLI().run()

