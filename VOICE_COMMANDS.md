# Voice Command Interface for CompUse

This document explains how to use the voice command interface for CompUse, which is implemented using the Pipecat framework.

## Overview

The voice command interface allows you to control your computer using voice commands. It uses:

- **Pipecat**: An open-source framework for building voice and multimodal conversational agents
- **Whisper**: OpenAI's speech recognition model for accurate transcription
- **ElevenLabs**: For high-quality text-to-speech feedback (optional)
- **Voice Activity Detection (VAD)**: For detecting when you've finished speaking

## Installation

1. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Set up your API keys in a `.env` file:
   ```
   OPENAI_API_KEY=your_openai_api_key
   ELEVENLABS_API_KEY=your_elevenlabs_api_key  # Optional, for voice feedback
   ELEVENLABS_VOICE_ID=your_elevenlabs_voice_id  # Optional
   COMPUSE_WAKE_WORD=computer  # Default wake word
   ```

## Usage

### Starting the Voice Interface

Run the voice command interface:

```bash
python voice_cli.py
```

Optional arguments:
- `--wake-word TEXT`: Set a custom wake word (default: "computer")
- `--auto-start`: Automatically start voice recognition on startup
- `--push-to-talk`: Use push-to-talk mode instead of wake word (press Ctrl+Space to talk)

### Available Commands

Once the CLI is running, you can use these text commands:

- `start`: Start voice recognition
- `stop`: Stop voice recognition
- `status`: Check if voice recognition is active
- `history`: Show voice command history
- `help`: Show available commands
- `exit`: Exit the application

### Using Voice Commands

When voice recognition is active:

1. Say the wake word followed by your command:
   - "Computer, take a screenshot"
   - "Computer, click at 500 300"
   - "Computer, open Chrome"

2. To stop listening:
   - "Computer, stop listening"

### Push-to-Talk Mode

If you prefer not to use a wake word, you can use push-to-talk mode:

```bash
python voice_cli.py --push-to-talk
```

In this mode:
1. Press and hold Ctrl+Space to start recording
2. Speak your command
3. Release Ctrl+Space to process the command

## Integration with CompUse

The voice command interface integrates with CompUse's existing tools:

- **GUI Tools**: Control mouse, keyboard, take screenshots, etc.
- **Browser Tools**: Control web browsers via Puppeteer
- **System Tools**: Interact with applications and system functions

All tools available in the CompUse CLI are accessible through voice commands.

## Command History

The voice interface keeps track of all commands you've issued. To view your command history:

1. Type `history` in the CLI
2. The system will display a table with timestamps and commands

This is useful for:
- Reviewing what commands you've already tried
- Debugging recognition issues
- Keeping track of your workflow

## Customization

### Changing the Wake Word

You can change the wake word in three ways:

1. Set the `COMPUSE_WAKE_WORD` environment variable
2. Use the `--wake-word` command-line argument
3. Edit the `.env` file

### Disabling Voice Feedback

Voice feedback can be disabled by modifying the `feedback_enabled` parameter in the `VoiceCommandManager` initialization.

## Troubleshooting

### Microphone Issues

If your microphone isn't being detected:

1. Check your system's microphone settings
2. Ensure your microphone is set as the default input device
3. Try running with administrator/sudo privileges

### Recognition Accuracy

If voice recognition accuracy is poor:

1. Speak clearly and at a moderate pace
2. Reduce background noise
3. Use a better quality microphone
4. Consider using a different wake word that's more distinct
5. Try push-to-talk mode instead of wake word detection

### API Key Issues

If you encounter API key errors:

1. Verify your API keys in the `.env` file
2. Check that you have sufficient credits/quota for the services
3. Ensure your network can reach the API endpoints

## Advanced Configuration

For advanced users, the `VoiceCommandManager` class accepts several configuration options:

- `whisper_api_key`: OpenAI API key for Whisper STT
- `elevenlabs_api_key`: ElevenLabs API key for TTS
- `elevenlabs_voice_id`: ElevenLabs voice ID for TTS
- `wake_word`: Wake word to activate voice listening
- `feedback_enabled`: Whether to provide audio feedback

These can be customized when initializing the manager in your code.