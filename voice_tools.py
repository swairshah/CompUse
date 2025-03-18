"""
Voice command interface for CompUse using Pipecat.

This module provides voice recognition capabilities for CompUse using the Pipecat framework.
It allows users to control their computer using voice commands, which are processed and
executed through the CompUse agent.
"""

import os
import asyncio
import logging
from typing import Optional, Dict, Any, List, Callable

import aiohttp
from pydantic_ai import RunContext, Tool
from pydantic_ai.tools import ToolDefinition

# Pipecat imports
from pipecat.frames.frames import AudioFrame, EndFrame, TextFrame
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.task import PipelineTask
from pipecat.pipeline.runner import PipelineRunner
from pipecat.services.whisper import WhisperSTTService
from pipecat.services.elevenlabs import ElevenLabsTTSService
from pipecat.transports.local import LocalTransport, LocalParams

# Set up logging
logger = logging.getLogger(__name__)

class VoiceToolDeps:
    """Dependencies for voice tools."""
    aiohttp_session: Optional[aiohttp.ClientSession] = None
    pipeline_runner: Optional[PipelineRunner] = None
    stt_service: Optional[WhisperSTTService] = None
    tts_service: Optional[ElevenLabsTTSService] = None
    transport: Optional[LocalTransport] = None
    voice_task: Optional[PipelineTask] = None
    command_callback: Optional[Callable[[str], None]] = None

class VoiceCommandManager:
    """Manages voice command recognition and processing."""
    
    def __init__(self, 
                 whisper_api_key: Optional[str] = None,
                 elevenlabs_api_key: Optional[str] = None,
                 elevenlabs_voice_id: Optional[str] = None,
                 wake_word: Optional[str] = None,
                 feedback_enabled: bool = True):
        """
        Initialize the voice command manager.
        
        Args:
            whisper_api_key: OpenAI API key for Whisper STT (if None, uses env var)
            elevenlabs_api_key: ElevenLabs API key for TTS (if None, uses env var)
            elevenlabs_voice_id: ElevenLabs voice ID for TTS
            wake_word: Optional wake word to activate voice listening
            feedback_enabled: Whether to provide audio feedback
        """
        self.whisper_api_key = whisper_api_key or os.getenv("OPENAI_API_KEY")
        self.elevenlabs_api_key = elevenlabs_api_key or os.getenv("ELEVENLABS_API_KEY")
        self.elevenlabs_voice_id = elevenlabs_voice_id or os.getenv("ELEVENLABS_VOICE_ID")
        self.wake_word = wake_word
        self.feedback_enabled = feedback_enabled
        self.deps = VoiceToolDeps()
        self.is_listening = False
        self.is_initialized = False
        
    async def initialize(self, command_callback: Callable[[str], None]):
        """
        Initialize the voice command system.
        
        Args:
            command_callback: Function to call when a command is recognized
        """
        if self.is_initialized:
            return
            
        # Create aiohttp session
        self.deps.aiohttp_session = aiohttp.ClientSession()
        
        # Set up the command callback
        self.deps.command_callback = command_callback
        
        # Initialize Whisper STT service
        self.deps.stt_service = WhisperSTTService(
            aiohttp_session=self.deps.aiohttp_session,
            api_key=self.whisper_api_key
        )
        
        # Initialize ElevenLabs TTS service if feedback is enabled
        if self.feedback_enabled and self.elevenlabs_api_key:
            self.deps.tts_service = ElevenLabsTTSService(
                aiohttp_session=self.deps.aiohttp_session,
                api_key=self.elevenlabs_api_key,
                voice_id=self.elevenlabs_voice_id
            )
        
        # Initialize local transport for audio
        self.deps.transport = LocalTransport(
            name="CompUse Voice",
            params=LocalParams(
                audio_in_enabled=True,
                audio_out_enabled=self.feedback_enabled
            )
        )
        
        # Create pipeline runner
        self.deps.pipeline_runner = PipelineRunner()
        
        # Create pipeline for speech-to-text
        pipeline_components = [self.deps.transport.input(), self.deps.stt_service]
        
        # Add TTS if feedback is enabled
        if self.feedback_enabled and self.deps.tts_service:
            pipeline_components.extend([self.deps.tts_service, self.deps.transport.output()])
        
        pipeline = Pipeline(pipeline_components)
        
        # Create pipeline task
        self.deps.voice_task = PipelineTask(pipeline)
        
        # Register frame handler for text frames (speech recognition results)
        @self.deps.voice_task.frame_handler(TextFrame)
        async def on_text_frame(task: PipelineTask, frame: TextFrame):
            text = frame.text.strip()
            logger.info(f"Recognized speech: {text}")
            
            # Check for wake word if configured
            if self.wake_word:
                if text.lower().startswith(self.wake_word.lower()):
                    # Remove wake word from command
                    command = text[len(self.wake_word):].strip()
                    await self._process_command(command)
            else:
                # No wake word, process all recognized speech
                await self._process_command(text)
        
        self.is_initialized = True
        logger.info("Voice command system initialized")
    
    async def _process_command(self, command: str):
        """Process a recognized command."""
        if not command:
            return
            
        logger.info(f"Processing command: {command}")
        
        # Call the command callback with the recognized command
        if self.deps.command_callback:
            self.deps.command_callback(command)
            
        # Provide audio feedback if enabled
        if self.feedback_enabled and self.deps.tts_service:
            await self.deps.voice_task.queue_frames([
                TextFrame(f"Processing command: {command}"),
                EndFrame()
            ])
    
    async def start_listening(self):
        """Start listening for voice commands."""
        if not self.is_initialized:
            logger.error("Voice command system not initialized")
            return
            
        if self.is_listening:
            logger.warning("Already listening for voice commands")
            return
            
        # Start the pipeline runner
        await self.deps.pipeline_runner.run(self.deps.voice_task)
        self.is_listening = True
        
        # Provide audio feedback if enabled
        if self.feedback_enabled and self.deps.tts_service:
            await self.deps.voice_task.queue_frames([
                TextFrame("Voice command system activated. I'm listening."),
                EndFrame()
            ])
            
        logger.info("Started listening for voice commands")
    
    async def stop_listening(self):
        """Stop listening for voice commands."""
        if not self.is_listening:
            return
            
        # Provide audio feedback if enabled
        if self.feedback_enabled and self.deps.tts_service:
            await self.deps.voice_task.queue_frames([
                TextFrame("Voice command system deactivated."),
                EndFrame()
            ])
            
        # Stop the pipeline runner
        await self.deps.pipeline_runner.stop()
        self.is_listening = False
        logger.info("Stopped listening for voice commands")
    
    async def cleanup(self):
        """Clean up resources."""
        if self.is_listening:
            await self.stop_listening()
            
        if self.deps.aiohttp_session:
            await self.deps.aiohttp_session.close()
            
        self.is_initialized = False
        logger.info("Voice command system cleaned up")

# Tool for starting voice recognition
async def voice_recognition_start(ctx: RunContext[VoiceToolDeps]) -> Dict[str, Any]:
    """Start voice recognition to listen for commands.
    
    Args:
        ctx: The context with voice tool dependencies.
    """
    try:
        voice_manager = ctx.deps.voice_manager
        if not voice_manager:
            return {"error": "Voice command manager not initialized"}
            
        await voice_manager.start_listening()
        return {
            "success": True,
            "message": "Voice recognition started"
        }
    except Exception as e:
        logger.error(f"Error starting voice recognition: {str(e)}")
        return {"error": str(e)}

# Tool for stopping voice recognition
async def voice_recognition_stop(ctx: RunContext[VoiceToolDeps]) -> Dict[str, Any]:
    """Stop voice recognition.
    
    Args:
        ctx: The context with voice tool dependencies.
    """
    try:
        voice_manager = ctx.deps.voice_manager
        if not voice_manager:
            return {"error": "Voice command manager not initialized"}
            
        await voice_manager.stop_listening()
        return {
            "success": True,
            "message": "Voice recognition stopped"
        }
    except Exception as e:
        logger.error(f"Error stopping voice recognition: {str(e)}")
        return {"error": str(e)}

# Create tool definitions
voice_start_tool = Tool(voice_recognition_start)
voice_stop_tool = Tool(voice_recognition_stop)