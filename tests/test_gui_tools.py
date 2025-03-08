import os
import sys
import platform
import subprocess
import pytest
from unittest.mock import patch, MagicMock, AsyncMock
from PIL import Image
from io import BytesIO

from CompUse.gui_tools import (
    screenshot, mouse_move, mouse_click, keyboard_type, key_press,
    image_locate, drag_and_drop, scroll, get_screen_size, 
    get_mouse_position, switch_window, focus_application,
    RunContext, GuiToolDeps
)


# Fixtures for pytest
@pytest.fixture
def ctx():
    """Create a mock RunContext for testing."""
    context = MagicMock(spec=RunContext)
    context.deps = MagicMock(spec=GuiToolDeps)
    return context


@pytest.fixture(autouse=True)
def setup_screenshots_dir():
    """Create a temp directory for screenshots if it doesn't exist."""
    temp_dir = os.path.join(os.getcwd(), "screenshots")
    os.makedirs(temp_dir, exist_ok=True)
    yield
    # Cleanup could go here if needed

@pytest.mark.asyncio
async def test_screenshot(ctx):
    """Test the screenshot function with real screenshots."""
    import os
    from PIL import Image
    
    # Create a test directory
    # test_dir = os.path.join(os.getcwd(), "test_screenshots")
    test_dir = os.path.join("/tmp", "test_screenshots")
    os.makedirs(test_dir, exist_ok=True)
    
    # Test full screen screenshot
    result = await screenshot(ctx, file_dir=test_dir, file_name="test_full.png")
    
    # Verify the screenshot was taken and saved
    assert os.path.exists(result["file_path"])
    with Image.open(result["file_path"]) as img:
        assert img.width > 0
        assert img.height > 0
        assert result["width"] == img.width
        assert result["height"] == img.height
    
    # Test region screenshot (100x100 region from top-left)
    region = [0, 0, 100, 100]
    result = await screenshot(ctx, region=region, file_dir=test_dir, file_name="test_region.png")
    
    # Verify the region screenshot
    assert os.path.exists(result["file_path"])
    with Image.open(result["file_path"]) as img:
        assert img.width == 100
        assert img.height == 100
        assert result["width"] == 100
        assert result["height"] == 100
    
    # Cleanup test files
    # os.remove(os.path.join(test_dir, "test_full.png"))
    # os.remove(os.path.join(test_dir, "test_region.png"))
    # os.rmdir(test_dir)

@pytest.mark.asyncio
@patch('CompUse.gui_tools.pyautogui.moveTo')
async def test_mouse_move_mock(mock_move_to, ctx):
    """Test the mouse_move function."""
    result = await mouse_move(ctx, 100, 200, 0.1)
    
    mock_move_to.assert_called_once_with(100, 200, duration=0.1)
    assert result["success"]
    assert result["position"]["x"] == 100
    assert result["position"]["y"] == 200

@pytest.mark.asyncio
async def test_mouse_move(ctx):
    """Test the mouse_move function with real mouse movement."""
    # Move to top-left corner
    result = await mouse_move(ctx, 10, 10, 0.2)
    assert result["position"]["x"] == 10
    assert result["position"]["y"] == 10
     
    result = await mouse_move(ctx, 500, 500, 0.4)
    assert result["position"]["x"] == 500
    assert result["position"]["y"] == 500


@pytest.mark.asyncio
@patch('CompUse.gui_tools.pyautogui.click')
@patch('CompUse.gui_tools.pyautogui.position')
async def test_mouse_click(mock_position, mock_click, ctx):
    """Test the mouse_click function."""
    # Test click at specific position
    result = await mouse_click(ctx, "left", 100, 200, 2)
    
    mock_click.assert_called_once_with(x=100, y=200, button="left", clicks=2)
    assert result["success"]
    
    # Reset mocks
    mock_click.reset_mock()
    
    # Test click at current position
    mock_position.return_value = (300, 400)
    result = await mouse_click(ctx)
    
    mock_click.assert_called_once_with(button="left", clicks=1)
    assert result["success"]
    assert result["position"]["x"] == 300
    assert result["position"]["y"] == 400

@pytest.mark.asyncio
@patch('CompUse.gui_tools.pyautogui.write')
async def test_keyboard_type(mock_write, ctx):
    """Test the keyboard_type function."""
    result = await keyboard_type(ctx, "Hello, world!", 0.01)
    
    mock_write.assert_called_once_with("Hello, world!", interval=0.01)
    assert result["success"]
    assert result["text"] == "Hello, world!"
    assert result["chars_typed"] == 13

@pytest.mark.asyncio
@patch('CompUse.gui_tools.pyautogui.press')
@patch('CompUse.gui_tools.pyautogui.hotkey')
async def test_key_press(mock_hotkey, mock_press, ctx):
    """Test the key_press function."""
    # Test single key
    result = await key_press(ctx, ["enter"])
    
    mock_press.assert_called_once_with("enter")
    mock_hotkey.assert_not_called()
    assert result["success"]
    
    # Reset mocks
    mock_press.reset_mock()
    
    # Test key combination
    result = await key_press(ctx, ["ctrl", "c"])
    
    mock_press.assert_not_called()
    mock_hotkey.assert_called_once_with("ctrl", "c")
    assert result["success"]

@pytest.mark.asyncio
@patch('CompUse.gui_tools.pyautogui.locateOnScreen')
async def test_image_locate(mock_locate, ctx):
    """Test the image_locate function."""
    # Test image found
    mock_box = MagicMock()
    mock_box.left = 100
    mock_box.top = 200
    mock_box.width = 50
    mock_box.height = 60
    mock_locate.return_value = mock_box
    
    result = await image_locate(ctx, "test.png", 0.8)
    
    mock_locate.assert_called_once_with("test.png", confidence=0.8)
    assert result["found"]
    assert result["position"]["x"] == 125  # 100 + 50/2
    assert result["position"]["y"] == 230  # 200 + 60/2
    
    # Reset mocks
    mock_locate.reset_mock()
    
    # Test image not found
    mock_locate.return_value = None
    
    result = await image_locate(ctx, "test.png")
    
    assert not result["found"]

@pytest.mark.asyncio
@patch('CompUse.gui_tools.pyautogui.moveTo')
@patch('CompUse.gui_tools.pyautogui.dragTo')
async def test_drag_and_drop(mock_drag_to, mock_move_to, ctx):
    """Test the drag_and_drop function."""
    result = await drag_and_drop(ctx, 100, 200, 300, 400, 1.0)
    
    mock_move_to.assert_called_once_with(100, 200, duration=0.5)
    mock_drag_to.assert_called_once_with(300, 400, duration=0.5)
    assert result["success"]
    assert result["start"]["x"] == 100
    assert result["start"]["y"] == 200
    assert result["end"]["x"] == 300
    assert result["end"]["y"] == 400

@pytest.mark.asyncio
@patch('CompUse.gui_tools.pyautogui.scroll')
async def test_scroll(mock_scroll, ctx):
    """Test the scroll function."""
    # Test scroll down
    result = await scroll(ctx, 5, "down")
    
    mock_scroll.assert_called_once_with(5)
    assert result["success"]
    assert result["direction"] == "down"
    
    # Reset mocks
    mock_scroll.reset_mock()
    
    # Test scroll up
    result = await scroll(ctx, 3, "up")
    
    mock_scroll.assert_called_once_with(-3)
    assert result["success"]
    assert result["direction"] == "up"

@pytest.mark.asyncio
@patch('CompUse.gui_tools.pyautogui.size')
async def test_get_screen_size(mock_size, ctx):
    """Test the get_screen_size function."""
    mock_size.return_value = (1920, 1080)
    
    result = await get_screen_size(ctx)
    
    assert mock_size.call_count == 1
    assert result["width"] == 1920
    assert result["height"] == 1080

@pytest.mark.asyncio
@patch('CompUse.gui_tools.pyautogui.position')
async def test_get_mouse_position(mock_position, ctx):
    """Test the get_mouse_position function."""
    mock_position.return_value = (100, 200)
    
    result = await get_mouse_position(ctx)
    
    assert mock_position.call_count == 1
    assert result["x"] == 100
    assert result["y"] == 200

@pytest.mark.asyncio
@patch('platform.system')
@patch('CompUse.gui_tools.pyautogui.hotkey')
async def test_switch_window(mock_hotkey, mock_system, ctx):
    """Test the switch_window function."""
    # Test on macOS
    mock_system.return_value = 'Darwin'
    
    result = await switch_window(ctx)
    
    mock_hotkey.assert_called_once_with('command', 'tab')
    assert result["success"]
    
    # Reset mocks
    mock_hotkey.reset_mock()
    
    # Test on Windows
    mock_system.return_value = 'Windows'
    
    result = await switch_window(ctx)
    
    mock_hotkey.assert_called_once_with('alt', 'tab')
    assert result["success"]

@pytest.mark.asyncio
@patch('platform.system')
@patch('subprocess.run')
async def test_focus_application_mac(mock_run, mock_system, ctx):
    """Test the focus_application function on macOS."""
    # Test on macOS
    mock_system.return_value = 'Darwin'
    mock_result = MagicMock()
    mock_result.returncode = 0
    mock_run.return_value = mock_result
    
    result = await focus_application(ctx, "Chrome")
    
    assert mock_run.call_count == 1
    assert result["success"]
    assert "Focused application: Chrome" in result["action"]
    
    # Test failure case
    mock_result.returncode = 1
    mock_result.stderr = "Application not found"
    
    result = await focus_application(ctx, "NonExistentApp")
    
    assert not result["success"]
    assert "error" in result

@pytest.mark.asyncio
@patch('platform.system')
async def test_focus_application_other(mock_system, ctx):
    """Test the focus_application function on other platforms."""
    # Test on Linux
    mock_system.return_value = 'Linux'
    
    result = await focus_application(ctx, "Chrome")
    
    assert not result["success"]
    assert "error" in result


if __name__ == "__main__":
    # Run the tests with pytest
    pytest.main(['-xvs', __file__])
