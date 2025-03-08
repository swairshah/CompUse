from setuptools import setup

setup(
    name="compuse",
    version="0.1.0",
    py_modules=["cli"],
    install_requires=[
        "python-dotenv>=1.0.0",
        "requests>=2.31.0",
        "mcp>=1.0.0",
        "uvicorn>=0.32.1",
        "anthropic>=0.18.0",
        "pydantic_ai==0.0.27",
        "pyautogui>=0.9.54",
        "pillow>=10.0.0",
        "pytest",
        "pytest-asyncio",
        "rich>=13.3.5",
        "prompt_toolkit"
    ],
    entry_points={
        'console_scripts': [
            'cue=cli:main',
        ],
    },
) 