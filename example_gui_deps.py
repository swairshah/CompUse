from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Any, Optional

from PIL import Image
import pyautogui
from pydantic_ai import RunContext


@dataclass
class ExampleGuiToolDeps:
    """Example dependencies for GUI tools with advanced features.
    
    example of how the GuiToolDeps class could be extended in the future
    to to support GUI automation with safety features, state tracking,
    and advanced capabilities.
    """
    
    # Configuration settings
    config: Dict[str, Any] = field(default_factory=lambda: {
        "mouse_speed": 0.5,           # Movement duration in seconds
        "screenshot_quality": 80,     # JPEG quality (1-100)
        "safe_zones": [(0, 0, 1920, 1080)],  # Allowed screen regions
        "max_actions_per_minute": 30, # Rate limiting
        "ocr_confidence": 0.7,        # Minimum OCR confidence
        "vision_enabled": False       # Whether to use vision AI
    })
    
    # State management
    last_screenshot: Optional[Dict[str, Any]] = None
    click_history: List[Dict[str, Any]] = field(default_factory=list)
    window_positions: Dict[str, Tuple[int, int, int, int]] = field(default_factory=dict)
    action_counter: int = 0
    session_start_time: float = field(default_factory=time.time)
    
    # helpers services
    ocr_engine: Optional[Any] = None  # Would be initialized with Tesseract
    vision_analyzer: Optional[Any] = None  # Could be connected to Claude Vision API
    
    def __post_init__(self):
        """Initialize optional components if enabled in config."""
        if self.config.get("ocr_enabled", False) and self.ocr_engine is None:
            try:
                # Lazy import 
                import pytesseract
                self.ocr_engine = pytesseract
                print("OCR engine initialized")
            except ImportError:
                print("Warning: pytesseract not installed. OCR features will be disabled.")
                
        if self.config.get("vision_enabled", False) and self.vision_analyzer is None:
            try:
                # Placeholder for vision service setup
                class VisionAnalyzer:
                    def detect_windows(self, image_path):
                        # Sample implementation
                        return [{
                            "title": "Example Window",
                            "bounds": (100, 100, 500, 400),
                            "confidence": 0.95
                        }]
                    
                    def analyze_screen(self, image_path):
                        # Sample implementation
                        return {
                            "elements": [
                                {"type": "button", "text": "OK", "bounds": (400, 300, 450, 330)},
                                {"type": "text_field", "bounds": (200, 200, 400, 230)}
                            ],
                            "description": "A dialog window with an OK button and a text field"
                        }
                
                self.vision_analyzer = VisionAnalyzer()
                print("Vision analyzer initialized")
            except Exception as e:
                print(f"Warning: Failed to initialize vision analyzer: {e}")
    
    # Safety/Guardrails/Monitoring
    def is_action_allowed(self, action_type: str, x: int = None, y: int = None) -> bool:
        """Check if an action is allowed based on rate limits and safe zones.
        
        Args:
            action_type: The type of action being performed (e.g., "click", "drag", "type")
            x: X coordinate for the action (if applicable)
            y: Y coordinate for the action (if applicable)
            
        Returns:
            bool: Whether the action is allowed
        """
        # Rate limiting
        current_time = time.time()
        minute_elapsed = (current_time - self.session_start_time) / 60
        if self.action_counter / max(1, minute_elapsed) > self.config["max_actions_per_minute"]:
            print(f"Rate limit exceeded: {self.action_counter} actions in {minute_elapsed:.2f} minutes")
            return False
            
        # Check if coordinates are in safe zones (if applicable)
        if x is not None and y is not None:
            in_safe_zone = False
            for zone in self.config["safe_zones"]:
                x1, y1, x2, y2 = zone
                if x1 <= x <= x2 and y1 <= y <= y2:
                    in_safe_zone = True
                    break
            if not in_safe_zone:
                print(f"Safety violation: Position ({x}, {y}) is outside of safe zones")
                return False
                
        self.action_counter += 1
        return True
    
    # Screenshot management  
    def store_screenshot(self, screenshot_data: Dict[str, Any]) -> None:
        """Store the latest screenshot data and extract window information.
        
        Args:
            screenshot_data: Dictionary with screenshot metadata and file path
        """
        self.last_screenshot = screenshot_data
        
        # We could analyze the screenshot here to detect windows automatically
        if self.vision_analyzer and self.config.get("vision_enabled", False):
            try:
                windows = self.vision_analyzer.detect_windows(screenshot_data["file_path"])
                for window in windows:
                    self.window_positions[window["title"]] = window["bounds"]
            except Exception as e:
                print(f"Error analyzing screenshot: {e}")
    
    # Text recognition helper
    async def get_text_from_region(self, x: int, y: int, width: int, height: int) -> str:
        """Extract text from a region of the last screenshot using OCR.
        
        Args:
            x: X coordinate of the top-left corner of the region
            y: Y coordinate of the top-left corner of the region
            width: Width of the region
            height: Height of the region
            
        Returns:
            str: Extracted text from the specified region
        """
        if not self.last_screenshot or not self.ocr_engine:
            print("Cannot extract text: No screenshot available or OCR engine not initialized")
            return ""
            
        try:
            img = Image.open(self.last_screenshot["file_path"])
            region = img.crop((x, y, x + width, y + height))
            
            # Save region to temporary file for debugging (optional)
            region_path = f"ocr_region_{int(time.time())}.png"
            region.save(region_path)
            
            # Use OCR engine to extract text
            text = self.ocr_engine.image_to_string(region)
            return text
        except Exception as e:
            print(f"Error extracting text: {e}")
            return ""
    
    # Screen analysis
    async def analyze_screen_content(self) -> Dict[str, Any]:
        """Analyze the screen content using vision API to understand what's visible.
        
        Returns:
            Dict: Analysis results including UI elements, text, and high-level description
        """
        if not self.last_screenshot:
            print("Cannot analyze screen: No screenshot available")
            return {"error": "No screenshot available"}
            
        if not self.vision_analyzer:
            print("Cannot analyze screen: Vision analyzer not initialized")
            return {"error": "Vision analyzer not initialized"}
            
        try:
            analysis = self.vision_analyzer.analyze_screen(self.last_screenshot["file_path"])
            return analysis
        except Exception as e:
            print(f"Error analyzing screen: {e}")
            return {"error": str(e)}


async def smart_click(ctx: RunContext[EnhancedGuiToolDeps], text: str) -> Dict[str, Any]:
    """Click on UI element containing specific text.
    
    Args:
        ctx: The context with enhanced dependencies
        text: Text to find and click on
        
    Returns:
        Dict: Result of the operation
    """
    # Use the OCR engine to find all text regions
    if not ctx.deps.last_screenshot or not ctx.deps.ocr_engine:
        print("No screenshot available or OCR engine not initialized")
        return {
            "success": False,
            "error": "Screenshot or OCR capabilities not available"
        }
    
    # Find text on screen using vision analyzer (simplified example)
    analysis = await ctx.deps.analyze_screen_content()
    
    if "error" in analysis:
        return {
            "success": False,
            "error": f"Failed to analyze screen: {analysis['error']}"
        }
    
    # Find UI element with matching text
    matching_element = None
    for element in analysis.get("elements", []):
        if "text" in element and text.lower() in element["text"].lower():
            matching_element = element
            break
    
    if matching_element:
        # Extract coordinates for center of the element
        x1, y1, x2, y2 = matching_element["bounds"]
        x = (x1 + x2) // 2
        y = (y1 + y2) // 2
        
        # Check if action is allowed (safety)
        if ctx.deps.is_action_allowed("click", x, y):
            # Perform the click with configured speed
            pyautogui.click(x, y, duration=ctx.deps.config["mouse_speed"])
            
            # Record in history
            ctx.deps.click_history.append({
                "type": "text_click",
                "text": text,
                "x": x, 
                "y": y,
                "time": time.time(),
                "element_type": matching_element.get("type", "unknown")
            })
            
            return {
                "success": True,
                "action": f"Clicked on text: '{text}' at ({x}, {y})",
                "element_type": matching_element.get("type", "unknown")
            }
        else:
            return {
                "success": False,
                "error": "Action not allowed due to safety restrictions"
            }
    else:
        return {
            "success": False,
            "error": f"Text '{text}' not found on screen"
        }


# Example usage code (not for execution)
def example_usage():
    """Example of how EnhancedGuiToolDeps would be used in practice."""
    # Create the dependencies with custom configuration
    deps = EnhancedGuiToolDeps(
        config={
            "mouse_speed": 0.3,
            "safe_zones": [(0, 0, 3000, 2000)],  # For multi-monitor setup
            "max_actions_per_minute": 60,  # Higher rate limit
            "ocr_enabled": True,
            "vision_enabled": True
        }
    )
    
    # Initialize a RunContext with these dependencies
    # ctx = RunContext(model=None, usage=None, prompt=None, deps=deps)
    
    # ideally tools would use these dependencies via RunContext
    # e.g.
    # await smart_click(ctx, "OK")
