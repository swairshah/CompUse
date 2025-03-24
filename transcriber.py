import os
import wave
import asyncio
import warnings
import numpy as np
import pyaudio
import webrtcvad
from dataclasses import dataclass
from typing import Optional, Dict, Any, Union
from abc import ABC, abstractmethod
from openai import OpenAI
from anthropic import Anthropic

try:
    from dotenv import load_dotenv
    load_dotenv()
    print("Environment variables loaded from .env file")
except ImportError:
    pass

try:
    from deepgram import DeepgramClient, PrerecordedOptions, FileSource
    DEEPGRAM_AVAILABLE = True
except ImportError:
    DEEPGRAM_AVAILABLE = False

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
    silence_threshold: int = 15  # Number of silent chunks before processing
    energy_threshold: int = 800  # RMS energy threshold for silence detection
    min_audio_length: int = int(AudioConfig.sample_rate * 0.2 / AudioConfig.chunk_size)  # Min 0.2 seconds of audio

@dataclass
class TranscriberConfig:
    """Configuration for transcription services"""
    provider: str = "openai"  # "openai", "deepgram", etc.
    
    openai_model: str = "whisper-1"
    
    # Deepgram specific settings
    deepgram_model: str = "nova-3"
    deepgram_smart_format: bool = True
    deepgram_language: Optional[str] = None
    deepgram_options: Optional[Dict[str, Any]] = None

@dataclass
class SanitizerConfig:
    """Configuration for transcript sanitization"""
    enabled: bool = True
    model_type: str = "openai"  # "openai" or "anthropic"
    openai_model: str = "gpt-4o-mini"
    anthropic_model: str = "claude-3-haiku-20240307"
    streaming: bool = True  # Whether to use streaming for sanitization
    system_prompt: str = ("You act as a transcriber that fixes transcription errors."
                         "Keep the text concise and faithful to the original meaning," 
                         "but fix any obvious word errors. "
                         "<example> <input> you what is you going you on </input> "
                         "  <o> what is going on </o> </example>"
                         "<example> <input> .... </input> "
                         "  <o>  </o> </example>"
                         "<example> <input> uhm yes a i think uh so </input> "
                         "  <o> yes i think so </o> </example>")

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
        
        # Check if below threshold
        is_silent = energy < threshold
        
        if DEBUG and not is_silent:
            print(f"Energy level: {energy:.2f}, Threshold: {threshold}")
            
        return is_silent
    except Exception as e:
        print(f"Error in silence detection: {e}")
        return True
        
class Transcriber(ABC):
    """Abstract base class for transcription services"""
    
    @abstractmethod
    async def transcribe(self, file_path: str) -> str:
        """Transcribe audio from a file"""
        pass

class OpenAITranscriber(Transcriber):
    """OpenAI-based transcription service"""
    
    def __init__(self, config: TranscriberConfig):
        self.client = OpenAI()
        self.model = config.openai_model
        
    async def transcribe(self, file_path: str) -> str:
        """Transcribe audio using OpenAI's API"""
        try:
            with open(file_path, "rb") as audio_file:
                response = await asyncio.get_event_loop().run_in_executor(
                    None,
                    lambda: self.client.audio.transcriptions.create(
                        model=self.model,
                        file=audio_file,
                        response_format="text",
                        language="en"
                    )
                )
                
                if isinstance(response, str):
                    return response.strip()
                elif hasattr(response, 'text'):
                    return response.text.strip()
                else:
                    debug_print(f"Unexpected response type: {type(response)}")
                    return str(response).strip()
                
        except Exception as e:
            debug_print(f"Error in OpenAI transcription: {e}")
            return ""

class DeepgramTranscriber(Transcriber):
    """Deepgram-based transcription service"""
    
    def __init__(self, config: TranscriberConfig):
        if not DEEPGRAM_AVAILABLE:
            raise ImportError("Deepgram SDK is not installed. Install with 'pip install deepgram-sdk'")
        
        api_key = os.environ.get("DEEPGRAM_API_KEY")
        if not api_key:
            raise ValueError("DEEPGRAM_API_KEY environment variable must be set for Deepgram transcription")
            
        try:
            self.client = DeepgramClient(api_key)
            debug_print("Deepgram client initialized successfully")
        except Exception as e:
            debug_print(f"Error initializing Deepgram client: {e}")
            raise
            
        self.model = config.deepgram_model
        self.smart_format = config.deepgram_smart_format
        self.language = config.deepgram_language
        self.extra_options = config.deepgram_options or {}
        
        debug_print(f"Initialized Deepgram client with model {self.model}")
        
    async def transcribe(self, file_path: str) -> str:
        """Transcribe audio using Deepgram's API"""
        try:
            debug_print(f"Starting Deepgram transcription for file: {file_path}")
            
            # Set up the Deepgram configuration
            options = {
                "model": self.model,
                "smart_format": self.smart_format
            }
            
            # Add language if specified
            if self.language:
                options["language"] = self.language
                
            # Add any extra options
            for key, value in self.extra_options.items():
                options[key] = value
                
            debug_print(f"Using Deepgram options: {options}")
            
            # Convert execution to a non-blocking call
            def sync_transcribe():
                try:
                    with open(file_path, 'rb') as audio:
                        source = FileSource(audio)
                        # Create options object
                        config = PrerecordedOptions(
                            model=self.model,
                            smart_format=self.smart_format
                        )
                        
                        # Add language if specified
                        if self.language:
                            config.language = self.language
                        
                        # Add any additional options
                        for key, value in self.extra_options.items():
                            setattr(config, key, value)
                            
                        return self.client.transcribe(source, config)
                except Exception as e:
                    debug_print(f"Sync transcription error: {e}")
                    return None
            
            # Run the synchronous method in a thread pool
            response = await asyncio.get_event_loop().run_in_executor(None, sync_transcribe)
            
            if not response:
                debug_print("No response received from Deepgram")
                return ""
                
            debug_print("Successfully received response from Deepgram API")
            
            # Extract transcript text
            try:
                # Get the transcript from the result
                transcript = response.results.channels[0].alternatives[0].transcript
                debug_print(f"Extracted transcript: {transcript}")
                return transcript
            except Exception as extract_error:
                debug_print(f"Error extracting transcript: {extract_error}")
                # Try to extract using the new API structure
                try:
                    transcript = response.results.transcript
                    debug_print(f"Extracted transcript using alternative method: {transcript}")
                    return transcript
                except:
                    if hasattr(response, 'to_dict'):
                        debug_print(f"Response dict: {response.to_dict()}")
                    return ""
                
        except Exception as e:
            debug_print(f"Error in Deepgram transcription: {e}")
            import traceback
            debug_print(traceback.format_exc())
            return ""
            
def get_transcriber(config: TranscriberConfig) -> Transcriber:
    """Factory function to create the appropriate transcriber based on config"""
    if config.provider.lower() == "openai":
        return OpenAITranscriber(config)
    elif config.provider.lower() == "deepgram":
        if not DEEPGRAM_AVAILABLE:
            raise ImportError("Deepgram SDK is not installed. Install with 'pip install deepgram-sdk'")
        return DeepgramTranscriber(config)
    else:
        raise ValueError(f"Unsupported transcription provider: {config.provider}")
        
class TranscriptSanitizer:
    """Sanitizes transcriptions using a language model to fix errors"""
    
    def __init__(self, config=SanitizerConfig()):
        self.config = config
        
        if not self.config.enabled:
            return
            
        if self.config.model_type == "openai":
            self.client = OpenAI()
        elif self.config.model_type == "anthropic":
            self.client = Anthropic()
        else:
            raise ValueError(f"Unsupported model type: {self.config.model_type}")
        
        debug_print(f"Sanitizer initialized with model type: {self.config.model_type}")
        
    async def sanitize(self, text):
        """Sanitize the transcription text."""
        if not text or not text.strip() or not self.config.enabled:
            return text
            
        debug_print(f"Sanitizing text: {text}")
        
        try:
            if self.config.streaming:
                return await self._sanitize_streaming(text)
            else:
                return await self._sanitize_non_streaming(text)
        except Exception as e:
            debug_print(f"Error sanitizing text: {e}")
            return text  # Return original on error
            
    async def _sanitize_non_streaming(self, text):
        """Sanitize text using non-streaming API call."""
        try:
            if self.config.model_type == "openai":
                response = await asyncio.get_event_loop().run_in_executor(
                    None,
                    lambda: self.client.chat.completions.create(
                        model=self.config.openai_model,
                        messages=[
                            {"role": "system", "content": self.config.system_prompt},
                            {"role": "user", "content": text}
                        ],
                        max_tokens=1024
                    )
                )
                return response.choices[0].message.content
            else:  # anthropic
                response = await asyncio.get_event_loop().run_in_executor(
                    None,
                    lambda: self.client.messages.create(
                        model=self.config.anthropic_model,
                        system=self.config.system_prompt,
                        messages=[
                            {"role": "user", "content": text}
                        ],
                        max_tokens=1024
                    )
                )
                return response.content[0].text
        except Exception as e:
            debug_print(f"Error in non-streaming sanitization: {e}")
            return text
            
    async def _sanitize_streaming(self, text):
        """Sanitize text using streaming API call."""
        sanitized_text = ""
        
        try:
            if self.config.model_type == "openai":
                stream = await asyncio.get_event_loop().run_in_executor(
                    None,
                    lambda: self.client.chat.completions.create(
                        model=self.config.openai_model,
                        messages=[
                            {"role": "system", "content": self.config.system_prompt},
                            {"role": "user", "content": text}
                        ],
                        max_tokens=1024,
                        stream=True
                    )
                )
                
                for chunk in stream:
                    if chunk.choices[0].delta.content:
                        sanitized_text += chunk.choices[0].delta.content
                        
            else:  # anthropic
                stream = await asyncio.get_event_loop().run_in_executor(
                    None,
                    lambda: self.client.messages.create(
                        model=self.config.anthropic_model,
                        system=self.config.system_prompt,
                        messages=[
                            {"role": "user", "content": text}
                        ],
                        max_tokens=1024,
                        stream=True
                    )
                )
                
                async for chunk in stream:
                    if chunk.delta.text:
                        sanitized_text += chunk.delta.text
                        
            return sanitized_text if sanitized_text.strip() else text
            
        except Exception as e:
            debug_print(f"Error in streaming sanitization: {e}")
            return text

class StreamManager:
    """Manages streaming of audio to transcription services"""
    
    def __init__(self, transcriber_config=TranscriberConfig(), sanitizer_config=SanitizerConfig()):
        self.audio_queue = asyncio.Queue()
        self.is_active = False
        self.audio_buffer = []
        self.process_task = None
        
        # Voice activity detection
        self.vad = webrtcvad.Vad(VADConfig.mode)
        self.silence_frames = 0
        
        # Initialize transcriber
        self.transcriber = get_transcriber(transcriber_config)
        
        # Transcript sanitizer
        self.sanitizer = TranscriptSanitizer(sanitizer_config)
        
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
            # Initialize active_speech_detected if not present
            if not hasattr(self, 'active_speech_detected'):
                self.active_speech_detected = False
            
            # Check for voice activity
            is_speech = False
            try:
                is_speech = self.vad.is_speech(audio_bytes, AudioConfig.sample_rate)
            except Exception as e:
                is_speech = False  # Default to false on error
            
            # Check energy level
            is_not_silent = not is_silent(audio_bytes)
            debug_print(f"Energy check - not silent: {is_not_silent}")
            
            # Start recording when we detect speech (either VAD or energy)
            if is_speech or is_not_silent:
                self.active_speech_detected = True
                self.audio_buffer.append(audio_bytes)
                self.silence_frames = 0
                debug_print("Speech detected, added chunk to buffer")
            # Continue recording if we've already detected speech but now just hear silence
            elif self.active_speech_detected:
                self.audio_buffer.append(audio_bytes)
                self.silence_frames += 1
                debug_print(f"Continuing to record through silence. Silent frames: {self.silence_frames}")
                
                # Reset active speech detection if we've had enough silence
                if self.silence_frames >= VADConfig.silence_threshold:
                    self.active_speech_detected = False
                    debug_print("Speech ended, preparing to process buffer")
            else:
                # Not recording, just counting silence
                self.silence_frames += 1
                debug_print(f"No speech detected. Silent frames: {self.silence_frames}")
            
            # Process buffer if:
            # 1. We have enough speech data AND
            # 2. Either we hit silence or buffer is getting too large
            buffer_size = len(self.audio_buffer)
            if buffer_size >= VADConfig.min_audio_length and (
                self.silence_frames >= VADConfig.silence_threshold or 
                buffer_size >= int(AudioConfig.sample_rate * 10 / AudioConfig.chunk_size)  # Max 10 seconds
            ):
                # Combine chunks and send for transcription
                audio_data = b''.join(self.audio_buffer)
                self.audio_buffer = []  # Clear buffer
                self.silence_frames = 0
                
                # Only process if we have enough audio data
                if len(audio_data) >= AudioConfig.chunk_size * VADConfig.min_audio_length:
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
        """Transcribe an audio chunk using the configured transcriber."""
        try:
            transcript_text = await self.transcriber.transcribe(file_path)
                
            if transcript_text:
                debug_print(f"Raw transcription: {transcript_text}")
                sanitized_text = await self.sanitizer.sanitize(transcript_text)
                
                # Output the sanitized text
                print(f"Transcription: {sanitized_text}")
                
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
            channels=AudioConfig.channels,
            rate=AudioConfig.sample_rate,
            input=True,
            frames_per_buffer=AudioConfig.chunk_size
        )
        
        print("Starting audio streaming. Speak into your microphone...")
        print("Press Enter to stop streaming...")
        
        self.is_streaming = True
        
        try:
            while self.is_streaming:
                # Read audio data from microphone
                audio_data = stream.read(AudioConfig.chunk_size, exception_on_overflow=False)
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

async def main(debug=False, transcriber_config=None, sanitizer_config=None):
    """Main function to run the application."""
    global DEBUG
    DEBUG = debug
    
    if transcriber_config is None:
        transcriber_config = TranscriberConfig()
        
    if sanitizer_config is None:
        sanitizer_config = SanitizerConfig()
    
    # Create and start stream manager
    stream_manager = StreamManager(transcriber_config, sanitizer_config)
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
    
    parser = argparse.ArgumentParser(
        description='Real-time Audio Transcription',
        epilog="""
Environment Variables:
  OPENAI_API_KEY      Required for OpenAI transcription
  DEEPGRAM_API_KEY    Required for Deepgram transcription
  ANTHROPIC_API_KEY   Required for Anthropic-based sanitization
        """
    )
    parser.add_argument('--debug', action='store_true', help='Enable debug mode')
    
    # Transcriber arguments
    transcriber_group = parser.add_argument_group('Transcription Options')
    transcriber_group.add_argument('--transcriber', choices=['openai', 'deepgram'], default='openai', 
                               help='Transcription service to use')
    
    # OpenAI transcriber options
    openai_group = parser.add_argument_group('OpenAI Transcription Options')
    openai_group.add_argument('--openai-transcription-model', default='gpt-4o-transcribe', 
                          help='OpenAI model to use for transcription')
    
    # Deepgram transcriber options
    deepgram_group = parser.add_argument_group('Deepgram Transcription Options')
    deepgram_group.add_argument('--deepgram-model', default='nova-3', 
                             help='Deepgram model to use for transcription')
    deepgram_group.add_argument('--deepgram-smart-format', action='store_true', default=True, 
                             help='Enable smart formatting in Deepgram')
    deepgram_group.add_argument('--deepgram-language', help='Specify language for Deepgram transcription')
    
    # Sanitizer arguments
    sanitizer_group = parser.add_argument_group('Sanitization Options')
    sanitizer_group.add_argument('--sanitize', action='store_true', help='Enable transcript sanitization')
    sanitizer_group.add_argument('--sanitizer-model', choices=['openai', 'anthropic'], default='openai', 
                              help='Model to use for sanitization')
    sanitizer_group.add_argument('--openai-sanitizer-model', default='gpt-4o-mini', 
                              help='OpenAI model to use for sanitization')
    sanitizer_group.add_argument('--anthropic-model', default='claude-3-haiku-20240307', 
                              help='Anthropic model to use for sanitization')
    sanitizer_group.add_argument('--streaming', action='store_true', help='Use streaming for sanitization')
    sanitizer_group.add_argument('--system-prompt', help='Custom system prompt for the sanitizer')
    
    args = parser.parse_args()
    
    transcriber_config = TranscriberConfig(
        provider=args.transcriber,
        openai_model=args.openai_transcription_model,
        deepgram_model=args.deepgram_model,
        deepgram_smart_format=args.deepgram_smart_format,
        deepgram_language=args.deepgram_language
    )
    
    # Configure sanitizer
    sanitizer_config = SanitizerConfig(
        enabled=args.sanitize,
        model_type=args.sanitizer_model,
        openai_model=args.openai_sanitizer_model,
        anthropic_model=args.anthropic_model,
        streaming=args.streaming
    )
    
    # Update system prompt if provided
    if args.system_prompt:
        sanitizer_config.system_prompt = args.system_prompt

    try:
        asyncio.run(main(
            debug=args.debug, 
            transcriber_config=transcriber_config, 
            sanitizer_config=sanitizer_config
        ))
    except Exception as e:
        print(f"Application error: {e}")
        if args.debug:
            import traceback
            traceback.print_exc()