from __future__ import annotations

import asyncio
import os
import time
from dataclasses import dataclass
from typing import List, Optional, Dict, Any

import pyautogui
from PIL import Image
from pydantic_ai import Agent, RunContext, ImageUrl

import platform
import logging

# Make PyAutoGUI safe 
pyautogui.PAUSE = 0.1
# Fail-safe: move mouse to upper left to abort
pyautogui.FAILSAFE = True

# Set up logging
logger = logging.getLogger(__name__)

if platform.system() == 'Darwin':
    try:
        from AppKit import NSWorkspace, NSApplicationActivateIgnoringOtherApps
    except ImportError:
        logger.warning("AppKit not available. macOS app functions will not work.")


@dataclass
class GuiToolDeps:
    """Dependencies for GUI tools."""
    pass


gui_agent = Agent(
    'openai:gpt-4o',
    system_prompt=(
        'Use the provided GUI tools to control the computer. '
        'Capture screenshots to see what is on screen, then use mouse and keyboard tools to interact.'
    ),
    deps_type=GuiToolDeps,
    retries=1,
)


@gui_agent.tool
async def screenshot(ctx: RunContext[GuiToolDeps],
                     region: Optional[List[int]] = None,
                     file_dir: Optional[str] = None,
                     file_name: Optional[str] = None
                     ) -> Dict[str, Any]:
    """Take a screenshot and save it to a local file for viewing.

    Args:
        ctx: The context.
        region: Optional region as [x, y, width, height]
        file_path: Optional file path to save the screenshot
    """
    try:
        if region and len(region) == 4:
            x, y, width, height = region
            screenshot = pyautogui.screenshot(region=(x, y, width, height))
        else:
            screenshot = pyautogui.screenshot()

        temp_dir = file_dir
        if not temp_dir:
            temp_dir = os.path.join(os.getcwd(), "screenshots")
            os.makedirs(temp_dir, exist_ok=True)

        if not file_name:
            timestamp = int(time.time())
            file_path = os.path.join(temp_dir, f"screenshot_{timestamp}.png")
        else:
            file_path = os.path.join(temp_dir, file_name)

        screenshot.save(file_path)

        return {
            "file_path": file_path,
            "width": screenshot.width,
            "height": screenshot.height,
            "message": f"Screenshot saved to {file_path}."
        }
    except Exception as e:
        return {"error": str(e)}


@gui_agent.tool
async def mouse_move(ctx: RunContext[GuiToolDeps], x: int, y: int, duration: float = 0.5) -> Dict[str, Any]:
    """Move mouse to x, y coordinates with smooth motion.

    Args:
        ctx: The context.
        x: X coordinate
        y: Y coordinate
        duration: Duration of move in seconds
    """
    try:
        pyautogui.moveTo(x, y, duration=duration)
        return {
            "success": True,
            "position": {"x": x, "y": y}
        }
    except Exception as e:
        return {"error": str(e)}


@gui_agent.tool
async def mouse_click(ctx: RunContext[GuiToolDeps],
                      button: str = "left",
                      x: Optional[int] = None,
                      y: Optional[int] = None,
                      clicks: int = 1) -> Dict[str, Any]:
    """Click at current position or specified x, y coordinates.

    Args:
        ctx: The context.
        button: Mouse button to click ("left", "right", or "middle")
        x: Optional X coordinate
        y: Optional Y coordinate
        clicks: Number of clicks
    """
    try:
        if x is not None and y is not None:
            pyautogui.click(x=x, y=y, button=button, clicks=clicks)
            position = (x, y)
        else:
            # Click at current position
            pyautogui.click(button=button, clicks=clicks)
            position = pyautogui.position()

        return {
            "success": True,
            "action": f"{button} click x{clicks}",
            "position": {"x": position[0], "y": position[1]}
        }
    except Exception as e:
        return {"error": str(e)}


@gui_agent.tool
async def keyboard_type(ctx: RunContext[GuiToolDeps], text: str, interval: float = 0.05) -> Dict[str, Any]:
    """Type text at current cursor position.

    Args:
        ctx: The context.
        text: Text to type
        interval: Interval between keystrokes
    """
    try:
        pyautogui.write(text, interval=interval)
        return {
            "success": True,
            "text": text,
            "chars_typed": len(text)
        }
    except Exception as e:
        return {"error": str(e)}


@gui_agent.tool
async def key_press(ctx: RunContext[GuiToolDeps], keys: List[str]) -> Dict[str, Any]:
    """Press specified keys (can be combinations like ['ctrl', 'c']).

    Args:
        ctx: The context.
        keys: Keys to press, can be single key or combination like ['ctrl', 'c']
    """
    try:
        if len(keys) == 1:
            pyautogui.press(keys[0])
        else:
            # For combinations like ctrl+c
            pyautogui.hotkey(*keys)

        return {
            "success": True,
            "keys_pressed": keys
        }
    except Exception as e:
        return {"error": str(e)}


@gui_agent.tool
async def image_locate(ctx: RunContext[GuiToolDeps],
                       image_path: str,
                       confidence: float = 0.9) -> Dict[str, Any]:
    """Find an image on screen and return its position.

    Args:
        ctx: The context.
        image_path: Path to image file to locate on screen
        confidence: Confidence threshold (0.0-1.0)
    """
    try:
        result = pyautogui.locateOnScreen(image_path, confidence=confidence)
        if result:
            return {
                "found": True,
                "position": {
                    "x": result.left + result.width // 2,
                    "y": result.top + result.height // 2,
                    "left": result.left,
                    "top": result.top,
                    "width": result.width,
                    "height": result.height
                }
            }
        else:
            return {"found": False}
    except Exception as e:
        return {"error": str(e)}


@gui_agent.tool
async def drag_and_drop(ctx: RunContext[GuiToolDeps],
                        start_x: int,
                        start_y: int,
                        end_x: int,
                        end_y: int,
                        duration: float = 0.5) -> Dict[str, Any]:
    """Drag from one position to another.

    Args:
        ctx: The context.
        start_x: Starting X coordinate
        start_y: Starting Y coordinate
        end_x: Ending X coordinate
        end_y: Ending Y coordinate
        duration: Duration of drag in seconds
    """
    try:
        pyautogui.moveTo(start_x, start_y, duration=duration/2)
        pyautogui.dragTo(end_x, end_y, duration=duration/2)
        return {
            "success": True,
            "start": {"x": start_x, "y": start_y},
            "end": {"x": end_x, "y": end_y}
        }
    except Exception as e:
        return {"error": str(e)}


@gui_agent.tool
async def scroll(ctx: RunContext[GuiToolDeps], amount: int, direction: str = "down") -> Dict[str, Any]:
    """Scroll up or down.

    Args:
        ctx: The context.
        amount: Number of "clicks" to scroll
        direction: "up" or "down"
    """
    try:
        clicks = amount * (-1 if direction == "up" else 1)
        pyautogui.scroll(clicks)
        return {
            "success": True,
            "direction": direction,
            "amount": amount
        }
    except Exception as e:
        return {"error": str(e)}


@gui_agent.tool
async def get_screen_size(ctx: RunContext[GuiToolDeps]) -> Dict[str, Any]:
    """Get the size of the screen.

    Args:
        ctx: The context.
    """
    try:
        width, height = pyautogui.size()
        return {
            "width": width,
            "height": height
        }
    except Exception as e:
        return {"error": str(e)}


@gui_agent.tool
async def get_mouse_position(ctx: RunContext[GuiToolDeps]) -> Dict[str, Any]:
    """Get the current mouse position.

    Args:
        ctx: The context.
    """
    try:
        x, y = pyautogui.position()
        return {
            "x": x,
            "y": y
        }
    except Exception as e:
        return {"error": str(e)}


@gui_agent.tool
async def switch_window(ctx: RunContext[GuiToolDeps]) -> Dict[str, Any]:
    """Switch to the next window using Alt+Tab (Windows) or Command+Tab (Mac).

    Args:
        ctx: The context.
    """
    try:
        import platform
        system = platform.system()

        if system == 'Darwin':  # macOS
            pyautogui.hotkey('command', 'tab')
            return {"success": True, "action": "Switched window using Command+Tab (Mac)"}
        elif system == 'Windows':
            pyautogui.hotkey('alt', 'tab')
            return {"success": True, "action": "Switched window using Alt+Tab (Windows)"}
        elif system == 'Linux':
            pyautogui.hotkey('alt', 'tab')
            return {"success": True, "action": "Switched window using Alt+Tab (Linux)"}
        else:
            return {"error": f"Unsupported platform: {system}"}
    except Exception as e:
        return {"error": str(e)}


@gui_agent.tool
async def focus_application(ctx: RunContext[GuiToolDeps], app_name: str) -> Dict[str, Any]:
    """Focus a specific application by name (works best on macOS).

    Args:
        ctx: The context.
        app_name: Name of the application to focus (e.g., "Chrome", "Terminal", "Finder")
    """
    try:
        import platform
        system = platform.system()

        if system == 'Darwin':  
            import subprocess

            # AppleScript to focus application
            script = f'''
            tell application "{app_name}"
                activate
            end tell
            '''

            result = subprocess.run(['osascript', '-e', script],
                                    capture_output=True,
                                    text=True)

            if result.returncode == 0:
                return {
                    "success": True,
                    "action": f"Focused application: {app_name}"
                }
            else:
                return {
                    "success": False,
                    "error": f"Failed to focus {app_name}: {result.stderr}"
                }
        elif system == 'Windows':
            # On Windows, we can try with the window title
            # This is less reliable but can work for some apps
            try:
                import win32gui
                import win32con

                def window_enum_callback(hwnd, results):
                    if win32gui.IsWindowVisible(hwnd) and app_name.lower() in win32gui.GetWindowText(hwnd).lower():
                        results.append(hwnd)

                results = []
                win32gui.EnumWindows(window_enum_callback, results)

                if results:
                    win32gui.SetForegroundWindow(results[0])
                    return {
                        "success": True,
                        "action": f"Focused window containing: {app_name}"
                    }
                else:
                    return {
                        "success": False,
                        "error": f"No visible window found containing: {app_name}"
                    }
            except ImportError:
                return {
                    "success": False,
                    "error": "win32gui not available. Install with: pip install pywin32"
                }
        else:
            return {
                "success": False,
                "error": f"Application focusing not implemented for {system}, try using switch_window instead"
            }
    except Exception as e:
        return {"error": str(e)}


@gui_agent.tool
async def get_frontmost_app(ctx: RunContext[GuiToolDeps]) -> Dict[str, Any]:
    """Get the currently frontmost application (macOS only).
    
    Args:
        ctx: The context.
        
    Returns:
        Information about the frontmost app including name and bundle ID.
    """
    try:
        if platform.system() != 'Darwin':
            return {"error": "This function is only available on macOS"}
            
        ws = NSWorkspace.sharedWorkspace()
        frontmost_app = ws.frontmostApplication()
        if frontmost_app:
            return {
                "success": True,
                "name": frontmost_app.localizedName(),
                "bundle_id": frontmost_app.bundleIdentifier()
            }
        return {"success": False, "error": "No frontmost application found"}
    except Exception as e:
        return {"error": str(e)}


@gui_agent.tool
async def activate_app_by_name(ctx: RunContext[GuiToolDeps], app_name: str) -> Dict[str, Any]:
    """Activate (bring to foreground) an application by name (macOS only).

    Args:
        ctx: The context.
        app_name: Name of the application to activate.
    """
    try:
        if platform.system() != 'Darwin':
            return {"error": "This function is only available on macOS"}
            
        apps = NSWorkspace.sharedWorkspace().runningApplications()

        # Try exact match first
        for app in apps:
            if app.localizedName() == app_name:
                logger.debug(f"Focusing {app_name} application")
                app.activateWithOptions_(NSApplicationActivateIgnoringOtherApps)
                return {"success": True, "name": app_name, "match_type": "exact"}

        # If exact match fails, try case-insensitive match or contains
        for app in apps:
            if app_name.lower() in app.localizedName().lower():
                app_name_found = app.localizedName()
                logger.debug(
                    f"Focusing app with name containing '{app_name}': {app_name_found}"
                )
                app.activateWithOptions_(NSApplicationActivateIgnoringOtherApps)
                return {"success": True, "name": app_name_found, "match_type": "partial"}

        return {"success": False, "error": f"No application named '{app_name}' found"}
    except Exception as e:
        return {"error": str(e)}


@gui_agent.tool
async def activate_app_by_bundle_id(ctx: RunContext[GuiToolDeps], bundle_id: str) -> Dict[str, Any]:
    """Activate (bring to foreground) an application by bundle ID (macOS only).

    Args:
        ctx: The context.
        bundle_id: Bundle ID of the application to activate.
    """
    try:
        if platform.system() != 'Darwin':
            return {"error": "This function is only available on macOS"}
            
        apps = NSWorkspace.sharedWorkspace().runningApplications()

        for app in apps:
            if app.bundleIdentifier() == bundle_id:
                logger.debug(f"Focusing application with bundle ID {bundle_id}")
                app.activateWithOptions_(NSApplicationActivateIgnoringOtherApps)
                return {"success": True, "bundle_id": bundle_id}

        return {"success": False, "error": f"No application with bundle ID '{bundle_id}' found"}
    except Exception as e:
        return {"error": str(e)}


async def main():
    """Example usage of GUI tools."""
    deps = GuiToolDeps()
    
    # Example: Take a screenshot, then click somewhere
    result = await gui_agent.run(
        "Take a screenshot, then click in the middle of the screen",
        deps=deps
    )
    print("Response:", result.data)
    result = await gui_agent.run(
        "Focus on the Firefox browser and then go to Cursor app",
        deps=deps
    )
    print("Response:", result.data)


if __name__ == "__main__":
    asyncio.run(main())
