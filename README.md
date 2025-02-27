# CompUse

Computer User "agent" PydanticAI + PyAutoGUI + Puppeteer (MCP server)

## Overview

We have combined Pydantic AI agent with MCP so that we register MCP tools as agent tools to a pyantic agent. at the same time we implement
nice GUI tools for the agent to use. and pull puppetter MCP server to use its tools for browser use. Idea is to experiment with the right abstraction layer of tools to implement for a decent computer use agent. 

## Features

- Desktop GUI automation with PyAutoGUI
- Web browser automation with Puppeteer MCP
- Voice (TODO) /text-based computer control
- Screenshot-based interaction (may be need to figure out things like bounding box etc to localize buttons windows)
- Cross-platform support (macOS, Windows, Linux) -- haven't tested on windows..

## How It Works

CompUse creates a bridge between Pydantic AI and MCP by:
1. Starting a Puppeteer MCP server as a subprocess
2. Querying available tools from the MCP server
3. Dynamically generating Pydantic AI-compatible tool wrappers for each MCP tool
4. Registering both GUI tools and MCP tools with a single agent
5. Providing a unified interface for users to control their computer with natural language

## Installation

```bash
pip install -r requirements.txt
npm install -g @modelcontextprotocol/server-puppeteer
```

## Quick Start

1. Start the GUI agent (desktop control only):
   ```bash
   python gui_agent_example.py
   ```

2. Start the combined agent (desktop + browser):
   ```bash
   python agent.py
   ```

## Requirements

- Python 3.7+
- Node.js 16+
- OpenAI API key (set in .env file)
