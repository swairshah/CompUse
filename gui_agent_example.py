import os
import asyncio
import logging
from dataclasses import dataclass
from typing import Any

from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIModel

import sys
from gui_tools import (
    screenshot, 
    mouse_move, 
    mouse_click, 
    keyboard_type, 
    key_press, 
    get_screen_size,
    get_mouse_position,
    switch_window,
    focus_application,
    GuiToolDeps
)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)


@dataclass
class ComputerControlDeps(GuiToolDeps):
    """Add any additional dependencies here if needed."""
    pass


async def main():
    model = OpenAIModel('gpt-4o')
    
    computer_agent = Agent(
        model,
        system_prompt=(
            'You are a computer control assistant. You can use the provided tools to '
            'control the computer with mouse and keyboard actions. You can take screenshots '
            'to see what is on screen, then interact with what you see.'
            '\n\n'
            'IMPORTANT SAFETY RULES:'
            '\n- Never try to access sensitive information'
            '\n- Do not try to install software without explicit permission'
            '\n- Always confirm before clicking on buttons that might change system settings'
            '\n- If you\'re uncertain about an action, ask for clarification first'
            '\n\n'
            'When responding to user queries, first understand what they want, then plan your '
            'actions step by step. If you need to see the screen first, take a screenshot.'
        ),
        deps_type=ComputerControlDeps,
        retries=1,
        tools=[
            screenshot,
            mouse_move,
            mouse_click,
            keyboard_type,
            key_press,
            get_screen_size,
            get_mouse_position,
            switch_window,
            focus_application
        ]
    )
    
    deps = ComputerControlDeps()
    
    # create screenshots directory if it doesn't exist
    os.makedirs(os.path.join(os.getcwd(), "screenshots"), exist_ok=True)
    
    print("Computer Control Assistant initialized!")
    print("You can now give commands to control your computer.")
    print("Type 'exit' to quit.")
    
    # interactive loop for user input
    while True:
        try:
            user_input = input("\nWhat would you like to do? > ")
            
            if user_input.lower() in ['exit', 'quit', 'bye']:
                print("Goodbye!")
                break
                
            print("Processing your request...")
            start_time = asyncio.get_event_loop().time()
            
            # run the agent with the user's input
            result = await computer_agent.run(user_input, deps=deps)
            
            elapsed = asyncio.get_event_loop().time() - start_time
            print(f"Response ({elapsed:.2f}s):")
            print(result.data)
            
        except KeyboardInterrupt:
            print("\nGoodbye!")
            break
        except Exception as e:
            print(f"Error: {str(e)}")


if __name__ == "__main__":
    asyncio.run(main())
