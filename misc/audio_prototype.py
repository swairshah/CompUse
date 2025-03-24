import os
import sys
import rich 
from rich.console import Console

import torch
import asyncio
import signal
import numpy as np
import sounddevice as sd
import wavio
from openai import AsyncOpenAI
from silero_vad import load_silero_vad, read_audio, get_speech_timestamps
import aioconsole

client = AsyncOpenAI(api_key=os.environ['OPENAI_API_KEY'])
console = Console()

SAMPLE_RATE = 16000  # Use 16kHz for Silero VAD
CHANNELS = 1
RECORD_SECONDS = 5

audio_queue = asyncio.Queue()  

vad_model =  load_silero_vad()

def is_too_quiet(audio_np, rms_threshold=0.005):
    rms = np.sqrt(np.mean(audio_np ** 2))
    return rms < rms_threshold

def is_voice_present(audio_np, sample_rate=16000, frame_size=512, threshold=0.9):
    # Convert full audio to a tensor
    audio_tensor = torch.from_numpy(audio_np).float()

    # Break into 512-sample chunks
    num_frames = len(audio_tensor) // frame_size
    chunks = audio_tensor[:num_frames * frame_size].reshape(num_frames, frame_size)

    # Run VAD on each chunk and get probs
    with torch.no_grad():
        probs = vad_model(chunks, sample_rate)

    # Determine if any frame has speech probability > threshold
    return (probs > threshold).any().item()

async def audio_stream():
    loop = asyncio.get_event_loop()
    while True:
        audio = await loop.run_in_executor(
            None,
            sd.rec,            
            int(RECORD_SECONDS * SAMPLE_RATE),
            SAMPLE_RATE,
            CHANNELS,
            'int16'
        )
        await loop.run_in_executor(None, sd.wait)

        # Check if audio has voice activity
        audio_np = audio.flatten().astype(np.float32) / 32768.0  # Normalize to [-1, 1]
        if is_too_quiet(audio_np):
            continue  # Skip silent recording
        if is_voice_present(audio_np):
            # Save and queue file
            temp_file = "temp_audio.wav"
            await loop.run_in_executor(None, lambda: wavio.write(temp_file, audio, SAMPLE_RATE, sampwidth=2))
            await audio_queue.put(temp_file)

async def capture_audio():
    return await audio_queue.get()  

async def process_voice_input():
    try:
        while True:
            audio_file = await capture_audio()
            if audio_file is None:
                break  # Graceful shutdown
            with open(audio_file, "rb") as audio:
                response = await client.audio.transcriptions.create(
                    model="whisper-1",
                    language="en",
                    file=audio
                )
            voice_text = response.text
            await handle_input(f"{voice_text}", caller="voice")
    except asyncio.CancelledError:
        console.print("[yellow]Voice input task cancelled.[/]")

async def process_text_input():
    """Process text input from console"""
    while True:
        text = await aioconsole.ainput("> ")
        await handle_input(f"{text}", caller="text")

async def llm_call(input_text):
    response_text = ""

    try:
        stream = await client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a helpful assistant"},
                {"role": "user", "content": input_text}
            ],
            stream=True
        )

        async for chunk in stream:
            delta = chunk.choices[0].delta.content
            if delta:
                response_text += delta
                await asyncio.to_thread(console.print, delta, end="")

    except Exception as e:
        await asyncio.to_thread(console.print, f"\n[red]Streaming failed:[/] {e}")

    await asyncio.to_thread(console.print, "")  
    return response_text

async def handle_input(input_text, caller):
    if caller == "voice":
        await asyncio.to_thread(console.print, f"{input_text}")
    llm_response = await llm_call(input_text)
    if caller == "voice":
        await asyncio.to_thread(console.print, "> ", end="")

async def shutdown(loop, signal=None):
    print("\nShutting down...")
    tasks = [t for t in asyncio.all_tasks() if t is not asyncio.current_task()]
    [task.cancel() for task in tasks]
    await asyncio.gather(*tasks, return_exceptions=True)
    loop.stop()

async def main():
    try:
        await asyncio.gather(
            audio_stream(),
            process_text_input(),
            process_voice_input()
        )
    except asyncio.CancelledError:
        console.print("[red]Main loop cancelled by Ctrl+C[/]")

if __name__ == "__main__":
    asyncio.run(main())

