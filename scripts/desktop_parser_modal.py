import modal
import os
import shutil
import time
from pathlib import Path
from typing import Dict, List
import base64
from PIL import Image

# Modal desktop parser implementation for CompUse
# - Uses pre-downloaded OCR models in the image (PaddleOCR and EasyOCR)
# - Stores OmniParser model weights in persistent volume
# - Uses YOLOv8 for object detection with cached models
# - Main function: parse_desktop_screenshot
# - Performance tracking to identify bottlenecks

app = modal.App("desktop-parser")

# Create a volume to store model weights and OCR models
model_volume = modal.Volume.from_name("models-volume", create_if_missing=True)

# Setup paths - use different paths for the image build and the volume mount
OMNIPARSER_PATH = "/root/OmniParser"
IMAGE_WEIGHTS_PATH = "/root/image_weights"  # For downloading during image build
VOLUME_PATH = "/root/volume_cache"  # This path should be empty for mounting
WEIGHTS_PATH = f"{VOLUME_PATH}/weights"

inference_image = (
    modal.Image.debian_slim(python_version="3.10")
    .apt_install(
        "git",
        # OpenCV system dependencies
        "libgl1-mesa-glx",
        "libglib2.0-0",
        "libsm6",
        "libxrender1", 
        "libxext6",
        # Font support
        "libfreetype6-dev",
        # Additional dependencies that might be needed
        "libfontconfig1",
        "libxcb1",
        "libx11-dev",
        "wget",
	    "unzip",
    )
    .pip_install(
        "torch",
        "torchvision",
        "easyocr",
        "supervision==0.18.0",
        "transformers>=4.35.0",
        "ultralytics==8.3.70",
        "numpy==1.26.4",
        "opencv-python",  # Use full OpenCV instead of headless version
        "openai==1.3.5",
        "pillow",
        "paddlepaddle",
        "paddleocr",
        "timm",
        "einops==0.8.0",
        "huggingface_hub",
        "uiautomation",
        "accelerate",
        "safetensors",
        "sentencepiece",
        "pydantic<2.0.0",  # Some HF models need older pydantic
    )
    .run_commands(
        # Clone the repository
        f"git clone https://github.com/microsoft/OmniParser.git {OMNIPARSER_PATH}/",
        
        # Create directories in the image for initial weights
        f"mkdir -p {IMAGE_WEIGHTS_PATH}/icon_detect",
        f"mkdir -p {IMAGE_WEIGHTS_PATH}/icon_caption_florence",
        f"mkdir -p /root/.paddleocr",
		f"mkdir -p /root/.EasyOCR/model",

        # Download models to the image (will be copied to volume at runtime if needed)
        f"huggingface-cli download microsoft/OmniParser-v2.0 icon_detect/train_args.yaml --local-dir {IMAGE_WEIGHTS_PATH}",
        f"huggingface-cli download microsoft/OmniParser-v2.0 icon_detect/model.pt --local-dir {IMAGE_WEIGHTS_PATH}",
        f"huggingface-cli download microsoft/OmniParser-v2.0 icon_detect/model.yaml --local-dir {IMAGE_WEIGHTS_PATH}",
        f"huggingface-cli download microsoft/OmniParser-v2.0 icon_caption/config.json --local-dir {IMAGE_WEIGHTS_PATH}",
        f"huggingface-cli download microsoft/OmniParser-v2.0 icon_caption/generation_config.json --local-dir {IMAGE_WEIGHTS_PATH}",
        f"huggingface-cli download microsoft/OmniParser-v2.0 icon_caption/model.safetensors --local-dir {IMAGE_WEIGHTS_PATH}",

        f"mkdir -p /root/.paddleocr/whl/det/en/en_PP-OCRv3_det_infer/",
        f"wget https://paddleocr.bj.bcebos.com/PP-OCRv3/english/en_PP-OCRv3_det_infer.tar -O /root/.paddleocr/whl/det/en/en_PP-OCRv3_det_infer/en_PP-OCRv3_det_infer.tar",
        f"mkdir -p /root/.paddleocr/whl/cls/ch_ppocr_mobile_v2.0_cls_infer/",
        f"wget https://paddleocr.bj.bcebos.com/dygraph_v2.0/ch/ch_ppocr_mobile_v2.0_cls_infer.tar -O /root/.paddleocr/whl/cls/ch_ppocr_mobile_v2.0_cls_infer/ch_ppocr_mobile_v2.0_cls_infer.tar",
        f"mkdir -p /root/.paddleocr/whl/rec/en/en_PP-OCRv4_rec_infer/",
        f"wget https://paddleocr.bj.bcebos.com/PP-OCRv4/english/en_PP-OCRv4_rec_infer.tar -O /root/.paddleocr/whl/rec/en/en_PP-OCRv4_rec_infer/en_PP-OCRv4_rec_infer.tar",

		f"wget https://github.com/JaidedAI/EasyOCR/releases/download/v1.3/english_g2.zip -O /root/.EasyOCR/model/english_g2.zip",
		f"wget https://github.com/JaidedAI/EasyOCR/releases/download/pre-v1.1.6/craft_mlt_25k.zip -O /root/.EasyOCR/model/craft_mlt_25k.zip",
		f"unzip /root/.EasyOCR/model/craft_mlt_25k.zip -d /root/.EasyOCR/model/",
		f"unzip /root/.EasyOCR/model/english_g2.zip -d /root/.EasyOCR/model/",

        f"mv {IMAGE_WEIGHTS_PATH}/icon_caption {IMAGE_WEIGHTS_PATH}/icon_caption_florence",
    )
)

def setup():
    """
    Set up the models required for desktop screenshot parsing
    """
    import sys
    import os
    
    # Start timing the setup process
    setup_start = time.time()
    
    # Add OmniParser repository to system path
    sys.path.append(OMNIPARSER_PATH)
    
    # Import utilities from OmniParser repository
    print("Importing OmniParser utilities...")
    import_start = time.time()
    from util.utils import get_yolo_model, get_caption_model_processor
    print(f"OmniParser imports completed in {time.time() - import_start:.2f}s")
    
    # Create necessary directories in the volume for OmniParser models
    os.makedirs(f"{WEIGHTS_PATH}/icon_detect", exist_ok=True)
    os.makedirs(f"{WEIGHTS_PATH}/icon_caption_florence", exist_ok=True)
    
    # Copy the weights from the image to the volume if they don't exist
    if not os.path.exists(f"{WEIGHTS_PATH}/icon_detect/model.pt"):
        print("First run - initializing OmniParser model cache in volume...")
        copy_start = time.time()
        import shutil
        
        for file_path in ["icon_detect/train_args.yaml", "icon_detect/model.pt", "icon_detect/model.yaml"]:
            src = f"{IMAGE_WEIGHTS_PATH}/{file_path}"
            dst = f"{WEIGHTS_PATH}/{file_path}"
            if os.path.exists(src):
                shutil.copy2(src, dst)
        
        # Copy the florence models
        for root, dirs, files in os.walk(f"{IMAGE_WEIGHTS_PATH}/icon_caption_florence"):
            for file in files:
                src = os.path.join(root, file)
                rel_path = os.path.relpath(src, IMAGE_WEIGHTS_PATH)
                dst = os.path.join(WEIGHTS_PATH, rel_path)
                os.makedirs(os.path.dirname(dst), exist_ok=True)
                shutil.copy2(src, dst)
        
        print(f"Copied model files in {time.time() - copy_start:.2f}s")
        model_volume.commit()
    
    # Load YOLO model
    print("Loading YOLO model...")
    yolo_start = time.time()
    yolo_model = get_yolo_model(model_path=f'{WEIGHTS_PATH}/icon_detect/model.pt')
    print(f"YOLO model loaded in {time.time() - yolo_start:.2f}s")
    
    # Load the caption model directly from Hugging Face
    print("Loading caption model...")
    caption_start = time.time()
    from transformers import AutoProcessor, AutoModelForCausalLM
    
    processor = AutoProcessor.from_pretrained("microsoft/florence-2-base", trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained("microsoft/florence-2-base", trust_remote_code=True)
    # Convert to dict format expected by get_som_labeled_img
    caption_model_processor = {'model': model, 'processor': processor}
    print(f"Caption model loaded in {time.time() - caption_start:.2f}s")
    
    print(f"Total setup time: {time.time() - setup_start:.2f}s")
    return yolo_model, caption_model_processor

@app.function(
    gpu="l40s",
    retries=3,
    image=inference_image,
    volumes={VOLUME_PATH: model_volume},
)
def parse_desktop_screenshot(
    image_bytes: bytes,
    box_threshold: float = 0.05,
    iou_threshold: float = 0.1,
    use_paddleocr: bool = True,
    imgsz: int = 640
) -> Dict:
    """
    Process a desktop screenshot and return the parsed GUI elements
    
    Args:
        image_bytes: The image as bytes
        box_threshold: Threshold for object detection boxes (0.01-1.0)
        iou_threshold: IOU threshold for removing overlapping boxes (0.01-1.0)
        use_paddleocr: Whether to use PaddleOCR for text detection
        imgsz: Image size for icon detection
        
    Returns:
        Dictionary with:
            - 'image_base64': Base64 encoded image string with annotations
            - 'parsed_content': Parsed text representation of GUI elements
            - 'elements': List of detected GUI elements with details
            - 'timing': Performance metrics for each processing stage
    """
    # Track overall processing time
    process_start = time.time()
    timing = {}
    
    # Import dependencies
    import torch
    import os
    import sys
    from tempfile import NamedTemporaryFile
    
    # Add OmniParser to path
    sys.path.append(OMNIPARSER_PATH)
    
    # Import utility functions
    import_start = time.time()
    try:
        from util.utils import check_ocr_box, get_som_labeled_img
        
        # Initialize OCR models if needed
        utils_module = sys.modules.get('util.utils')
        if utils_module:
            # Pre-initialize PaddleOCR
            if not hasattr(utils_module, 'paddle_ocr') or utils_module.paddle_ocr is None:
                print("Initializing PaddleOCR...")
                ocr_start = time.time()
                from paddleocr import PaddleOCR
                utils_module.paddle_ocr = PaddleOCR(
                    lang='en',
                    use_angle_cls=False,
                    show_log=True,
                    use_dilation=True,
                    det_db_score_mode='slow'
                )
                timing['paddleocr_init'] = time.time() - ocr_start
            
            # Pre-initialize EasyOCR
            if not hasattr(utils_module, 'reader') or utils_module.reader is None:
                print("Initializing EasyOCR...")
                easyocr_start = time.time()
                import easyocr
                utils_module.reader = easyocr.Reader(['en'])
                timing['easyocr_init'] = time.time() - easyocr_start
    except ImportError:
        print("Importing from repository root...")
        os.chdir(OMNIPARSER_PATH)
        from util.utils import check_ocr_box, get_som_labeled_img
    
    timing['imports'] = time.time() - import_start
    
    # Load models
    model_start = time.time()
    yolo_model, caption_model_processor = setup()
    timing['model_loading'] = time.time() - model_start
    
    # Process the image
    image_save_path = None
    try:
        # Save image to temporary file
        file_start = time.time()
        with NamedTemporaryFile(delete=False, suffix=".png") as temp_file:
            temp_file.write(image_bytes)
            image_save_path = temp_file.name
        timing['file_write'] = time.time() - file_start
        
        # Configure image parameters
        config_start = time.time()
        image = Image.open(image_save_path)
        box_overlay_ratio = image.size[0] / 3200
        draw_bbox_config = {
            'text_scale': 0.8 * box_overlay_ratio,
            'text_thickness': max(int(2 * box_overlay_ratio), 1),
            'text_padding': max(int(3 * box_overlay_ratio), 1),
            'thickness': max(int(3 * box_overlay_ratio), 1),
        }
        timing['config'] = time.time() - config_start
        
        # Run OCR detection
        print("Running OCR detection...")
        ocr_start = time.time()
        ocr_bbox_rslt, _ = check_ocr_box(
            image_save_path, 
            display_img=False, 
            output_bb_format='xyxy', 
            goal_filtering=None, 
            easyocr_args={'paragraph': False, 'text_threshold': 0.9}, 
            use_paddleocr=use_paddleocr
        )
        text, ocr_bbox = ocr_bbox_rslt
        timing['ocr_detection'] = time.time() - ocr_start
        
        # Run object detection and labeling
        print("Running object detection and captioning...")
        detection_start = time.time()
        dino_labled_img, label_coordinates, parsed_content_list = get_som_labeled_img(
            image_save_path, 
            yolo_model, 
            BOX_TRESHOLD=box_threshold, 
            output_coord_in_ratio=True, 
            ocr_bbox=ocr_bbox,
            draw_bbox_config=draw_bbox_config, 
            caption_model_processor=caption_model_processor, 
            ocr_text=text,
            iou_threshold=iou_threshold, 
            imgsz=imgsz,
        )
        timing['object_detection'] = time.time() - detection_start
        
        # Format the parsed content
        format_start = time.time()
        parsed_content_text = '\n'.join([f'icon {i}: ' + str(v) for i, v in enumerate(parsed_content_list)])
        timing['formatting'] = time.time() - format_start
        
        # Total processing time
        timing['total'] = time.time() - process_start
        print(f"Total processing time: {timing['total']:.2f}s")
        
        return {
            'image_base64': dino_labled_img,
            'parsed_content': parsed_content_text,
            'elements': parsed_content_list,
            'timing': timing
        }
    finally:
        # Clean up
        if image_save_path and os.path.exists(image_save_path):
            os.unlink(image_save_path)

@app.local_entrypoint()
def main(image_path: str = None):
    """
    Local entrypoint for testing the desktop parser
    
    Args:
        image_path: Path to the image file to process
    """
    import sys
    
    if not image_path:
        print("Please provide an image path to process")
        print("Usage: modal run desktop_parser_modal.py --image-path /path/to/image.png")
        sys.exit(1)
    
    image_path = Path(image_path)
    if not image_path.exists():
        print(f"Error: Image file {image_path} does not exist")
        sys.exit(1)
    
    print(f"Processing image: {image_path}")
    
    try:
        start_time = time.time()
        image_bytes = image_path.read_bytes()
        result = parse_desktop_screenshot.remote(image_bytes)
        total_time = time.time() - start_time
        
        # Save the processed image
        output_path = image_path.with_suffix('.processed.png')
        with open(output_path, 'wb') as f:
            f.write(base64.b64decode(result['image_base64']))
        
        print(f"Processed image saved to {output_path}")
        print("\nParsed Content:")
        print(result['parsed_content'])
        
        # Display timing information
        print("\nPerformance Metrics:")
        print(f"Total time (including Modal overhead): {total_time:.2f}s")
        
        if 'timing' in result:
            timing = result['timing']
            print("\nBreakdown of processing time:")
            
            # Sort by duration to identify bottlenecks
            sorted_times = sorted(
                [(k, v) for k, v in timing.items() if k != 'total'], 
                key=lambda x: x[1], 
                reverse=True
            )
            
            for stage, duration in sorted_times:
                percentage = (duration / timing['total']) * 100
                print(f"  - {stage}: {duration:.2f}s ({percentage:.1f}%)")
    
    except Exception as e:
        print(f"Error processing image: {str(e)}")
        sys.exit(1)

