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
    )
)

def setup():
    """
    Set up the YOLO model and BLIP2 caption model on GPU for desktop screenshot parsing
    """
    import sys
    import os
    import torch
    from transformers import Blip2Processor, Blip2ForConditionalGeneration
    from util.utils import get_yolo_model

    device = torch.device("cuda")
    print(f"Using GPU: {torch.cuda.get_device_name(0)}")

    setup_start = time.time()

    sys.path.append(OMNIPARSER_PATH)

    # Load YOLO model
    print("Loading YOLO model...")
    yolo_start = time.time()
    yolo_model = get_yolo_model(model_path=f'{WEIGHTS_PATH}/icon_detect/model.pt').to(device).float()
    print(f"YOLO model loaded in {time.time() - yolo_start:.2f}s")

    # Load BLIP2 caption model
    print("Loading BLIP2 caption model...")
    caption_start = time.time()
    
    # Initialize BLIP2 model and processor
    processor = Blip2Processor.from_pretrained("Salesforce/blip2-opt-2.7b")
    model = Blip2ForConditionalGeneration.from_pretrained(
        "Salesforce/blip2-opt-2.7b",
        device_map=None,
        torch_dtype=torch.float16  # BLIP2 works well with float16 on GPU
    ).to(device)
    
    caption_model_processor = {'model': model, 'processor': processor}
    print(f"BLIP2 caption model loaded in {time.time() - caption_start:.2f}s")

    print(f"Total setup time: {time.time() - setup_start:.2f}s")
    return yolo_model, caption_model_processor

def get_som_labeled_img_no_caption(
    image_path, 
    yolo_model, 
    BOX_TRESHOLD=0.05, 
    output_coord_in_ratio=True, 
    ocr_bbox=None,
    draw_bbox_config=None, 
    caption_model_processor=None,  # Ignored
    ocr_text=None,
    iou_threshold=0.1, 
    imgsz=640
):
    """
    Modified version of get_som_labeled_img that works without a caption model
    """
    import cv2
    import numpy as np
    import supervision as sv
    import base64
    
    # Load image
    cv_image = cv2.imread(image_path)
    
    # Run YOLO detection
    results = yolo_model(cv_image, imgsz=imgsz)
    
    # Convert YOLO output to supervision format
    detections = sv.Detections.from_ultralytics(results[0])
    
    # Filter based on confidence
    mask = detections.confidence >= BOX_TRESHOLD
    filtered_detections = detections[mask]
    
    # Draw bounding boxes
    box_annotator = sv.BoxAnnotator(
        thickness=draw_bbox_config.get('thickness', 2),
        text_thickness=draw_bbox_config.get('text_thickness', 2),
        text_scale=draw_bbox_config.get('text_scale', 0.5),
        text_padding=draw_bbox_config.get('text_padding', 5)
    )
    
    # Label with class names
    labels = [f"Object {i} ({confidence:.2f})"
            for i, confidence 
            in enumerate(filtered_detections.confidence)]
    
    # Annotate the image
    annotated_image = box_annotator.annotate(
        scene=cv_image,
        detections=filtered_detections,
        labels=labels
    )
    
    # Convert to base64
    _, buffer = cv2.imencode('.png', annotated_image)
    dino_labled_img = base64.b64encode(buffer).decode('utf-8')
    
    # Create parsed content list directly from detections
    parsed_content_list = []
    for i, (bbox, class_id, confidence) in enumerate(zip(
        filtered_detections.xyxy, 
        filtered_detections.class_id, 
        filtered_detections.confidence
    )):
        x1, y1, x2, y2 = bbox
        
        # Convert to ratio if needed
        if output_coord_in_ratio:
            h, w = cv_image.shape[:2]
            x1, x2 = x1/w, x2/w
            y1, y2 = y1/h, y2/h
        
        parsed_content_list.append({
            'type': 'icon',
            'bbox': [float(x1), float(y1), float(x2), float(y2)],
            'interactivity': True,
            'content': f'Object {i}',
            'source': 'box_yolo_content_yolo',
            'class_id': int(class_id),
            'confidence': float(confidence)
        })
    
    # Create empty label coordinates for compatibility
    label_coordinates = {}
    
    return dino_labled_img, label_coordinates, parsed_content_list

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
    Uses GPU acceleration for faster processing
    
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
            - 'gpu_info': GPU usage statistics (if available)
    """
    # Track overall processing time
    process_start = time.time()
    timing = {}
    gpu_info = {}
    
    # Import dependencies
    import torch
    import os
    import sys
    from tempfile import NamedTemporaryFile
    import numpy as np
    import cv2
    
    # Add OmniParser to path
    sys.path.append(OMNIPARSER_PATH)
    
    # Import utility functions - include get_som_labeled_img for captioning
    import_start = time.time()
    try:
        from util.utils import check_ocr_box, get_som_labeled_img
    except ImportError:
        print("Importing from repository root...")
        os.chdir(OMNIPARSER_PATH)
        from util.utils import check_ocr_box, get_som_labeled_img
    
    timing['imports'] = time.time() - import_start
    
    # Load YOLO model and caption model
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
        
        # Initialize OCR models if needed
        utils_module = sys.modules.get('util.utils')
        if utils_module:
            # Pre-initialize PaddleOCR with GPU
            if not hasattr(utils_module, 'paddle_ocr') or utils_module.paddle_ocr is None:
                print("Initializing PaddleOCR on GPU...")
                from paddleocr import PaddleOCR
                import paddle
                
                # Set Paddle to use GPU
                paddle.device.set_device('gpu')
                
                utils_module.paddle_ocr = PaddleOCR(
                    lang='en',
                    use_angle_cls=False,
                    show_log=True,
                    use_dilation=True,
                    det_db_score_mode='slow',
                    use_gpu=True  # Explicitly enable GPU
                )
                
            # Pre-initialize EasyOCR with GPU
            if not hasattr(utils_module, 'reader') or utils_module.reader is None:
                print("Initializing EasyOCR on GPU...")
                import easyocr
                
                # Configure EasyOCR to use GPU with optimized settings
                utils_module.reader = easyocr.Reader(
                    ['en'], 
                    gpu=True,  # Explicitly enable GPU
                    quantize=False  # Disable quantization for better performance on GPU
                )
        
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
        
        # Run object detection and captioning
        print("Running object detection and captioning...")
        detection_start = time.time()
        
        # Use get_som_labeled_img for both detection and captioning
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
        
        # Get GPU memory usage statistics
        gpu_info = {
            'gpu_name': torch.cuda.get_device_name(0),
            'memory_allocated_mb': torch.cuda.memory_allocated() / (1024 * 1024),
            'memory_reserved_mb': torch.cuda.memory_reserved() / (1024 * 1024),
            'max_memory_allocated_mb': torch.cuda.max_memory_allocated() / (1024 * 1024),
            'max_memory_reserved_mb': torch.cuda.max_memory_reserved() / (1024 * 1024),
        }
        print(f"GPU Memory Usage: {gpu_info['memory_allocated_mb']:.1f}MB allocated, "
              f"{gpu_info['max_memory_allocated_mb']:.1f}MB max")
        
        # Total processing time
        timing['total'] = time.time() - process_start
        print(f"Total processing time: {timing['total']:.2f}s")
        
        return {
            'image_base64': dino_labled_img,
            'parsed_content': parsed_content_text,
            'elements': parsed_content_list,
            'timing': timing,
            'gpu_info': gpu_info
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
        
        # Display GPU information if available
        if 'gpu_info' in result and result['gpu_info']:
            gpu_info = result['gpu_info']
            print("\nGPU Metrics:")
            print(f"  - GPU: {gpu_info.get('gpu_name', 'unknown')}")
            print(f"  - Memory Allocated: {gpu_info.get('memory_allocated_mb', 0):.1f} MB")
            print(f"  - Memory Reserved: {gpu_info.get('memory_reserved_mb', 0):.1f} MB")
            print(f"  - Peak Memory Used: {gpu_info.get('max_memory_allocated_mb', 0):.1f} MB")
    
    except Exception as e:
        print(f"Error processing image: {str(e)}")
        sys.exit(1)

