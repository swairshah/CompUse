from __future__ import annotations

import os
import asyncio
import threading
import time
import logging
from dataclasses import dataclass
from typing import Optional, Callable, Dict, Any, List
import queue

# Import speech recognition library
try:
    import speech_recognition as sr
except ImportError:
    raise ImportError(
        "speech_recognition package is required for voice commands. "
        "Install it with: pip install SpeechRecognition"
    )

# Import for audio feedback (optional)
try:
    import pyttsx3
except ImportError:
    pyttsx3 = None
    logging.warning(
        "pyttsx3 package not found. Audio feedback will be disabled. "
        "Install it with: pip install pyttsx3"
    )

# Import for hotkey detection (optional)
try:
    import keyboard
except ImportError:
    keyboard = None
    logging.warning(
        "keyboard package not found. Push-to-talk hotkey will be disabled. "
        "Install it with: pip install keyboard"
    )

# Set up logging
logger = logging.getLogger(__name__)

@dataclass
class VoiceCommandConfig:
    """Configuration for voice command recognition."""
    # General settings
    enabled: bool = True
    language: str = "en-US"
    
    # Listening mode settings
    continuous_listening: bool = True
    push_to_talk_key: str = "ctrl+space"  # Only used if continuous_listening is False
    
    # Wake word settings
    use_wake_word: bool = True
    wake_word: str = "computer"
    wake_word_timeout: int = 5  # seconds to listen after wake word
    
    # Recognition settings
    energy_threshold: int = 4000  # Microphone sensitivity
    pause_threshold: float = 0.8  # Seconds of silence to consider end of phrase
    dynamic_energy_threshold: bool = True
    
    # Feedback settings
    audio_feedback: bool = True
    visual_feedback: bool = True
    
    # Advanced settings
    timeout: int = 5  # Recognition timeout in seconds
    phrase_time_limit: int = 10  # Max seconds for a single phrase


class VoiceCommandListener:
    """Handles voice command recognition and processing."""
    
    def __init__(self, config: VoiceCommandConfig, command_callback: Callable[[str], None]):
        """Initialize the voice command listener.
        
        Args:
            config: Configuration for voice recognition
            command_callback: Function to call when a command is recognized
        """
        self.config = config
        self.command_callback = command_callback
        self.recognizer = sr.Recognizer()
        self.microphone = sr.Microphone()
        
        # Set up recognizer properties
        self.recognizer.energy_threshold = config.energy_threshold
        self.recognizer.pause_threshold = config.pause_threshold
        self.recognizer.dynamic_energy_threshold = config.dynamic_energy_threshold
        
        # Set up text-to-speech engine for feedback
        self.engine = None
        if config.audio_feedback and pyttsx3:
            self.engine = pyttsx3.init()
            self.engine.setProperty('rate', 150)
            
        # Thread control
        self.running = False
        self.listening_thread = None
        self.command_queue = queue.Queue()
        
        # Calibrate microphone
        with self.microphone as source:
            logger.info("Calibrating microphone...")
            self.recognizer.adjust_for_ambient_noise(source, duration=1)
            logger.info(f"Microphone calibrated. Energy threshold: {self.recognizer.energy_threshold}")
    
    def start(self):
        """Start the voice command listener in a background thread."""
        if self.running:
            logger.warning("Voice command listener is already running")
            return
            
        self.running = True
        
        if self.config.continuous_listening:
            self.listening_thread = threading.Thread(target=self._continuous_listening_loop)
        else:
            if not keyboard:
                logger.error("Push-to-talk requires the keyboard package")
                self.running = False
                return
            self.listening_thread = threading.Thread(target=self._push_to_talk_loop)
            
        self.listening_thread.daemon = True
        self.listening_thread.start()
        
        logger.info("Voice command listener started")
        if self.config.audio_feedback and self.engine:
            self.engine.say("Voice commands activated")
            self.engine.runAndWait()
    
    def stop(self):
        """Stop the voice command listener."""
        self.running = False
        if self.listening_thread:
            self.listening_thread.join(timeout=1.0)
            self.listening_thread = None
            
        logger.info("Voice command listener stopped")
        if self.config.audio_feedback and self.engine:
            self.engine.say("Voice commands deactivated")
            self.engine.runAndWait()
    
    def _continuous_listening_loop(self):
        """Background thread for continuous listening."""
        while self.running:
            try:
                with self.microphone as source:
                    if self.config.use_wake_word:
                        # Listen for wake word
                        logger.debug("Listening for wake word...")
                        audio = self.recognizer.listen(
                            source, 
                            timeout=None,
                            phrase_time_limit=2
                        )
                        
                        try:
                            text = self.recognizer.recognize_google(
                                audio, 
                                language=self.config.language
                            ).lower()
                            
                            if self.config.wake_word.lower() in text:
                                logger.info(f"Wake word detected: {text}")
                                if self.config.audio_feedback and self.engine:
                                    self.engine.say("Listening")
                                    self.engine.runAndWait()
                                
                                # Now listen for the actual command
                                logger.debug("Listening for command...")
                                command_audio = self.recognizer.listen(
                                    source, 
                                    timeout=self.config.wake_word_timeout,
                                    phrase_time_limit=self.config.phrase_time_limit
                                )
                                
                                self._process_audio(command_audio)
                        except sr.UnknownValueError:
                            # Wake word not recognized, continue listening
                            pass
                        except sr.RequestError as e:
                            logger.error(f"Could not request results: {e}")
                    else:
                        # Direct listening without wake word
                        logger.debug("Listening for command...")
                        audio = self.recognizer.listen(
                            source, 
                            timeout=None,
                            phrase_time_limit=self.config.phrase_time_limit
                        )
                        self._process_audio(audio)
            except Exception as e:
                logger.error(f"Error in continuous listening loop: {e}")
                time.sleep(1)  # Prevent tight loop on error
    
    def _push_to_talk_loop(self):
        """Background thread for push-to-talk listening."""
        if not keyboard:
            logger.error("Push-to-talk requires the keyboard package")
            return
            
        while self.running:
            try:
                # Wait for hotkey press
                keyboard.wait(self.config.push_to_talk_key)
                
                if not self.running:
                    break
                    
                logger.info("Push-to-talk key pressed")
                if self.config.audio_feedback and self.engine:
                    self.engine.say("Listening")
                    self.engine.runAndWait()
                
                # Listen for command
                with self.microphone as source:
                    logger.debug("Listening for command...")
                    audio = self.recognizer.listen(
                        source, 
                        timeout=self.config.timeout,
                        phrase_time_limit=self.config.phrase_time_limit
                    )
                    self._process_audio(audio)
            except Exception as e:
                logger.error(f"Error in push-to-talk loop: {e}")
                time.sleep(1)  # Prevent tight loop on error
    
    def _process_audio(self, audio):
        """Process audio data and extract command."""
        try:
            text = self.recognizer.recognize_google(
                audio, 
                language=self.config.language
            )
            
            logger.info(f"Recognized: {text}")
            
            if self.config.audio_feedback and self.engine:
                self.engine.say("Got it")
                self.engine.runAndWait()
                
            # Add command to queue for processing
            self.command_queue.put(text)
            
            # Call the callback function
            self.command_callback(text)
            
        except sr.UnknownValueError:
            logger.info("Could not understand audio")
            if self.config.audio_feedback and self.engine:
                self.engine.say("Sorry, I didn't catch that")
                self.engine.runAndWait()
        except sr.RequestError as e:
            logger.error(f"Could not request results: {e}")
            if self.config.audio_feedback and self.engine:
                self.engine.say("Sorry, I couldn't process that")
                self.engine.runAndWait()
    
    def get_next_command(self) -> Optional[str]:
        """Get the next command from the queue, if available."""
        try:
            return self.command_queue.get_nowait()
        except queue.Empty:
            return None


# Async wrapper for voice commands
class AsyncVoiceCommandManager:
    """Async wrapper for voice command listener to integrate with CompUse."""
    
    def __init__(self, config: Optional[VoiceCommandConfig] = None):
        """Initialize the async voice command manager.
        
        Args:
            config: Optional configuration for voice recognition
        """
        self.config = config or VoiceCommandConfig()
        self.command_queue = asyncio.Queue()
        self.listener = None
    
    async def start(self):
        """Start the voice command listener."""
        if self.listener:
            logger.warning("Voice command listener is already running")
            return
            
        # Create the listener with a callback that puts commands in the async queue
        def command_callback(text):
            asyncio.run_coroutine_threadsafe(
                self.command_queue.put(text),
                asyncio.get_event_loop()
            )
            
        self.listener = VoiceCommandListener(self.config, command_callback)
        self.listener.start()
    
    async def stop(self):
        """Stop the voice command listener."""
        if self.listener:
            self.listener.stop()
            self.listener = None
    
    async def get_next_command(self) -> str:
        """Wait for and return the next voice command."""
        return await self.command_queue.get()
    
    def get_next_command_nowait(self) -> Optional[str]:
        """Get the next command without waiting, returns None if no command available."""
        try:
            return self.command_queue.get_nowait()
        except asyncio.QueueEmpty:
            return None


# Example usage
async def main():
    """Example usage of voice command manager."""
    # Create voice command manager with default config
    manager = AsyncVoiceCommandManager()
    
    # Start listening
    await manager.start()
    
    print("Voice command listener started. Say something!")
    print("Press Ctrl+C to exit")
    
    try:
        while True:
            # Wait for next command
            command = await manager.get_next_command()
            print(f"Command received: {command}")
            
            # Process command (example)
            if "exit" in command.lower() or "quit" in command.lower():
                print("Exit command received, stopping...")
                break
    except KeyboardInterrupt:
        print("Interrupted by user")
    finally:
        # Stop listening
        await manager.stop()
        print("Voice command listener stopped")


if __name__ == "__main__":
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    # Run the example
    asyncio.run(main())