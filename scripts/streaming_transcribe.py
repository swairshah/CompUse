import asyncio
import os
import sounddevice as sd
import numpy as np
from openai import AsyncOpenAI
from dotenv import load_dotenv
from rich.console import Console
from rich.spinner import Spinner
from rich.live import Live
import io
import wave

load_dotenv()

SAMPLE_RATE = 16000
CHANNELS = 1
DURATION = 5  # seconds per recording

client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
console = Console()

async def transcribe_audio_buffer(audio_np):
    # Create in-memory buffer
    audio_buffer = io.BytesIO()
    
    # Write WAV data to memory buffer
    with wave.open(audio_buffer, 'wb') as wf:
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(2)  # 16-bit audio
        wf.setframerate(SAMPLE_RATE)
        wf.writeframes((audio_np * 32767).astype(np.int16).tobytes())
    
    # Reset buffer position
    audio_buffer.seek(0)
    
    response = await client.audio.transcriptions.create(
        model="whisper-1",
        file=("audio.wav", audio_buffer),  # Passing as tuple with filename
        language="en"
    )
    return response.text

async def transcribe_audio(audio_np):
    # Save the audio to a temporary file
    import tempfile
    import wave
    
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
        temp_path = temp_file.name
        
    # Write as WAV file
    with wave.open(temp_path, 'wb') as wf:
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(2)  # 16-bit audio
        wf.setframerate(SAMPLE_RATE)
        wf.writeframes((audio_np * 32767).astype(np.int16).tobytes())
    
    try:
        with open(temp_path, "rb") as audio_file:
            response = await client.audio.transcriptions.create(
                model="whisper-1",
                file=audio_file,
                language="en"
            )
        return response.text
    finally:
        os.remove(temp_path)

async def record_and_transcribe():
    loop = asyncio.get_event_loop()
    console.print("\n🎙️ Starting real-time transcription... Press Ctrl+C to stop.")

    try:
        while True:
            # Listening
            with Live(Spinner("point", style="grey70"), console=console, refresh_per_second=10, transient=True) as live:
                audio = await loop.run_in_executor(
                    None, sd.rec, int(DURATION * SAMPLE_RATE), SAMPLE_RATE, CHANNELS, "float32"
                )
                try:
                    await loop.run_in_executor(None, sd.wait)
                except KeyboardInterrupt:
                    sd.stop()
                    raise
                audio_np = audio.flatten()
            
            # Transcribing
            with Live(Spinner("point", style="bold green"), console=console, refresh_per_second=10, transient=True) as live:
                transcription = await transcribe_audio_buffer(audio_np)
            
            console.print(f"{transcription}")

    except KeyboardInterrupt:
        console.print("\n👋 Stopping transcription.")
        sd.stop()
    finally:
        sd.stop()


if __name__ == "__main__":
    asyncio.run(record_and_transcribe())

