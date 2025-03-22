import os
import wave
import asyncio
import warnings
import numpy as np
import pyaudio
import webrtcvad
from dataclasses import dataclass
from openai import OpenAI
from anthropic import Anthropic

warnings.filterwarnings("ignore")
   
DEBUG = False

@dataclass
class AudioConfig:
    """Configuration for audio recording"""
    sample_rate: int = 16000
    channels: int = 1
    format: int = pyaudio.paInt16
    chunk_size: int = 1024  # Number of frames per buffer 
 
@dataclass
class VADConfig:
    """Configuration for Voice Activity Detection"""
    mode: int = 1  # 0-3, higher means more aggressive filtering
    silence_threshold: int = 3  # Number of silent chunks before processing
    energy_threshold: int = 30  # RMS energy threshold for silence detection
    min_audio_length: int = int(AudioConfig.sample_rate * 0.1 / AudioConfig.chunk_size)  # Minimum 0.1 seconds of audio

def debug_print(message):
    """Print only if debug mode is enabled"""
    if DEBUG:
        print(message)

def is_silent(audio_data, threshold=VADConfig.energy_threshold):
    """Check if audio chunk is below energy threshold."""
    try:
        # Convert bytes to int16 array
        audio_array = np.frombuffer(audio_data, dtype=np.int16)
        # Calculate RMS energy
        energy = np.sqrt(np.mean(np.square(audio_array.astype(np.float32))))
        return energy < threshold
    except Exception as e:
        print(f"Error in silence detection: {e}")
        return True

class StreamManager:
    """Manages streaming of audio to OpenAI's API"""
    
    def __init__(self):
        self.client = OpenAI()
        self.audio_queue = asyncio.Queue()
        self.is_active = False
        self.audio_buffer = []
        self.process_task = None
        
        # Voice activity detection
        self.vad = webrtcvad.Vad(VADConfig.mode)
        self.silence_frames = 0
        
    async def start(self):
        """Start the stream processing."""
        self.is_active = True
        self.process_task = asyncio.create_task(self.process_audio_queue())
        return self
    
    async def process_audio_queue(self):
        """Process audio chunks from the queue."""
        while self.is_active:
            try:
                audio_bytes = await self.audio_queue.get()
                await self._handle_audio_input(audio_bytes)
                self.audio_queue.task_done()
            except asyncio.CancelledError:
                break
            except Exception as e:
                debug_print(f"Error processing audio queue: {e}")
    
    async def _handle_audio_input(self, audio_bytes):
        """Process audio input and send for transcription."""
        if not audio_bytes:
            debug_print("No audio bytes received")
            return
        
        try:
            # Check for voice activity
            is_speech = False
            try:
                is_speech = self.vad.is_speech(audio_bytes, AudioConfig.sample_rate)
                debug_print(f"VAD detected speech: {is_speech}")
            except Exception as e:
                debug_print(f"VAD error: {e}")
                is_speech = True  # Default to true on error
            
            # Also check energy level
            is_not_silent = not is_silent(audio_bytes)
            debug_print(f"Energy check - not silent: {is_not_silent}")
            
            # More lenient condition: either speech OR not silent
            if is_speech or is_not_silent:
                self.audio_buffer.append(audio_bytes)
                self.silence_frames = 0
                debug_print(f"Added chunk to buffer. Buffer size: {len(self.audio_buffer)}")
            else:
                self.silence_frames += 1
                debug_print(f"Silent frame detected. Silent frames: {self.silence_frames}")
            
            # Process buffer if:
            # 1. We have enough speech data AND
            # 2. Either we hit silence or buffer is getting too large
            buffer_size = len(self.audio_buffer)
            if buffer_size >= VADConfig.min_audio_length and (
                self.silence_frames >= VADConfig.silence_threshold or 
                buffer_size >= int(AudioConfig.sample_rate * 2 / AudioConfig.chunk_size)  # Max 2 seconds
            ):
                # Combine chunks and send for transcription
                audio_data = b''.join(self.audio_buffer)
                self.audio_buffer = []  # Clear buffer
                self.silence_frames = 0
                
                # Only process if we have enough audio data
                if len(audio_data) >= CHUNK_SIZE * VADConfig.min_audio_length:
                    # Create a temporary file for the audio chunk
                    temp_file = "temp_chunk.wav"
                    self._save_audio_chunk(audio_data, temp_file)
                    
                    # Send for transcription
                    await self._transcribe_chunk(temp_file)
                    
                    # Clean up
                    if os.path.exists(temp_file):
                        os.remove(temp_file)
                
        except Exception as e:
            debug_print(f"Error processing audio: {e}")
    
    def _save_audio_chunk(self, audio_data, file_path):
        """Save audio chunk to a WAV file."""
        with wave.open(file_path, 'wb') as wf:
            wf.setnchannels(AudioConfig.channels)
            wf.setsampwidth(pyaudio.get_sample_size(AudioConfig.format))
            wf.setframerate(AudioConfig.sample_rate)
            wf.writeframes(audio_data)
    
    async def _transcribe_chunk(self, file_path):
        """Transcribe an audio chunk using OpenAI's API."""
        try:
            with open(file_path, "rb") as audio_file:
                response = await asyncio.get_event_loop().run_in_executor(
                    None,
                    lambda: self.client.audio.transcriptions.create(
                        file=audio_file,
                        model="whisper-1",
                        language="en"
                    )
                )
                
                if response.text.strip():
                    print(f"Transcription: {response.text}")
                
        except Exception as e:
            debug_print(f"Error during transcription: {e}")
    
    async def add_audio_chunk(self, audio_bytes):
        """Add an audio chunk to the stream."""
        await self.audio_queue.put(audio_bytes)
    
    async def close(self):
        """Close the stream properly."""
        if not self.is_active:
            return
            
        self.is_active = False
        if self.process_task:
            self.process_task.cancel()
            try:
                await self.process_task
            except asyncio.CancelledError:
                pass
        
        self.audio_buffer = []

class MicrophoneHandler:
    """Handles microphone input and streaming."""
    
    def __init__(self, stream_manager):
        self.stream_manager = stream_manager
        self.is_streaming = False
        self.audio_task = None
        
    async def start_streaming(self):
        """Start streaming audio from the microphone."""
        if self.is_streaming:
            return
            
        p = pyaudio.PyAudio()
        
        # Open stream
        stream = p.open(
            format=AudioConfig.format,
            channels=AudioConfig.channel,
            rate=AudioConfig.rate,
            input=True,
            frames_per_buffer=AudioConfig.chunk_size
        )
        
        print("Starting audio streaming. Speak into your microphone...")
        print("Press Enter to stop streaming...")
        
        self.is_streaming = True
        
        try:
            while self.is_streaming:
                # Read audio data from microphone
                audio_data = stream.read(CHUNK_SIZE, exception_on_overflow=False)
                # Send to stream manager
                await self.stream_manager.add_audio_chunk(audio_data)
                # Small delay to prevent overwhelming the stream
                await asyncio.sleep(0.01)
                
        except Exception as e:
            debug_print(f"Error in audio streaming: {e}")
        finally:
            # Cleanup
            stream.stop_stream()
            stream.close()
            p.terminate()
            debug_print("Audio streaming stopped")
    
    async def stop_streaming(self):
        """Stop streaming audio."""
        if not self.is_streaming:
            return
            
        self.is_streaming = False
        
        if self.audio_task and not self.audio_task.done():
            self.audio_task.cancel()
            try:
                await self.audio_task
            except asyncio.CancelledError:
                pass
async def main(debug=False):
    """Main function to run the application."""
    global DEBUG
    DEBUG = debug
    
    # Create and start stream manager
    stream_manager = StreamManager()
    await stream_manager.start()
    
    # Create audio streamer
    audio_streamer = MicrophoneHandler(stream_manager)
    
    try:
        # Start streaming audio
        await audio_streamer.start_streaming()
        
    except KeyboardInterrupt:
        print("\nInterrupted by user")
    finally:
        # Stop streaming and clean up
        await audio_streamer.stop_streaming()
        await stream_manager.close()


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Real-time Audio Transcription')
    parser.add_argument('--debug', action='store_true', help='Enable debug mode')
    args = parser.parse_args()

    try:
        asyncio.run(main(debug=args.debug))
    except Exception as e:
        print(f"Application error: {e}")
        if args.debug:
            import traceback
            traceback.print_exc()
