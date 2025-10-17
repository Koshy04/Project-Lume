import os
import torch
import mss
import easyocr
import logging
import numpy as np
import traceback
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration
from typing import Optional, List, Dict, Tuple

class VisionInput:
    """Vision module for screen capture, OCR, and image captioning."""
    _instance = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super(VisionInput, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self, languages: List[str] = ['en'], device: str = 'cpu'):
        if self._initialized:
            return
            
        self.device = device
        self.logger = logging.getLogger(__name__)
        self.ocr_reader = None
        self.processor = None
        self.model = None

        print(f"Initializing Vision System on device: '{self.device}'...")
        try:
            self.ocr_reader = easyocr.Reader(languages)
            self.logger.info(f"OCR reader initialized for languages: {languages}")
        except Exception as e:
            self.logger.error(f"Failed to initialize OCR reader: {e}")

        try:
            model_id = "Salesforce/blip-image-captioning-base"
            self.processor = BlipProcessor.from_pretrained(model_id)
            self.model = BlipForConditionalGeneration.from_pretrained(model_id)
            if self.device == 'cuda' and torch.cuda.is_available():
                self.model.to('cuda')
            self.logger.info("Image captioning model initialized.")
        except Exception as e:
            self.logger.error(f"Failed to initialize image captioning model: {e}")
        
        self._initialized = True

    def get_monitors(self) -> List[Dict]:
        """
        Get current monitor information for debugging.
        
        Returns:
            List of monitor dictionaries
        """
        try:
            with mss.mss() as sct:
                monitors = sct.monitors
                return monitors
        except Exception as e:
            self.logger.error(f"Failed to get monitors: {e}, {traceback.format_exc()}")
            return []

    def capture_screenshot(self, monitor_index: int = 1, save_path: Optional[str] = None) -> Optional[Image.Image]:
        """
        Capture a screenshot of the specified monitor.
        
        Args:
            monitor_index: Index of the monitor to capture (default: 1)
            save_path: Optional path to save the screenshot
            
        Returns:
            PIL Image object or None if failed
        """
        try:
            with mss.mss() as sct:
                # Get monitor info
                monitors = sct.monitors
                self.logger.debug(f"Available monitors: {len(monitors)}")
                self.logger.debug(f"Requested monitor index: {monitor_index}")
                
                # Validate monitor index
                if monitor_index < 0 or monitor_index >= len(monitors):
                    self.logger.warning(f"Monitor index {monitor_index} not available. Available monitors: 0-{len(monitors)-1}")
                    # Try to find a valid monitor (skip monitor 0 which is usually "all monitors")
                    if len(monitors) > 1:
                        monitor_index = 1  # Default to first actual monitor
                    else:
                        monitor_index = 0
                
                self.logger.debug(f"Using monitor index: {monitor_index}")
                self.logger.debug(f"Monitor info: {monitors[monitor_index]}")
                
                # Capture screenshot
                screenshot = sct.grab(monitors[monitor_index])
                img = Image.frombytes("RGB", (screenshot.width, screenshot.height), screenshot.rgb)
                
                # Save if path provided
                if save_path:
                    img.save(save_path)
                    self.logger.info(f"Screenshot saved to: {save_path}")
                
                return img
                
        except Exception as e:
            self.logger.error(f"Failed to capture screenshot: {e}\n{traceback.format_exc()}")
            return None

    def perform_ocr(self, image: Image.Image, confidence_threshold: float = 0.5, scale_factor: float = 0.5, save_scaled_image: bool = False, scaled_image_path: str = None) -> List[Dict]:
        """
        Perform OCR on the given image.
        
        Args:
            image: PIL Image object
            confidence_threshold: Minimum confidence for text detection
            scale_factor: Factor to scale down the image (0.5 = half size) for faster processing
            save_scaled_image: Whether to save the scaled image used for OCR
            scaled_image_path: Path to save the scaled image if save_scaled_image is True
            
        Returns:
            List of dictionaries containing text, bounding box, and confidence
        """
        if self.ocr_reader is None:
            self.logger.error("OCR reader not initialized")
            return []
        
        try:
            # Scale down the image for faster OCR processing
            original_size = image.size
            if scale_factor != 1.0:
                new_width = int(original_size[0] * scale_factor)
                new_height = int(original_size[1] * scale_factor)
                scaled_image = image.resize((new_width, new_height), Image.Resampling.LANCZOS)
                self.logger.info(f"Scaled image from {original_size} to {scaled_image.size} for OCR processing")
                
                # Save scaled image if requested
                if save_scaled_image and scaled_image_path:
                    scaled_image.save(scaled_image_path)
                    self.logger.info(f"Scaled image saved to: {scaled_image_path}")
            else:
                scaled_image = image
            
            # Convert PIL image to numpy array for EasyOCR
            image_array = np.array(scaled_image)
            
            # Perform OCR on the numpy array
            results = self.ocr_reader.readtext(image_array, decoder='beamsearch')
            
            # Filter results by confidence threshold and scale bounding boxes back to original size
            filtered_results = []
            for bbox, text, conf in results:
                if conf >= confidence_threshold:
                    # Scale bounding box coordinates back to original image size
                    if scale_factor != 1.0:
                        scaled_bbox = []
                        for point in bbox:
                            scaled_point = (int(point[0] / scale_factor), int(point[1] / scale_factor))
                            scaled_bbox.append(scaled_point)
                        bbox = scaled_bbox
                    
                    filtered_results.append({
                        'text': text,
                        'bbox': bbox,
                        'confidence': conf
                    })
            
            self.logger.info(f"OCR completed: {len(filtered_results)} text regions detected")
            return filtered_results
            
        except Exception as e:
            self.logger.error(f"Failed to perform OCR: {e}")
            return []

    def generate_caption(self, image: Image.Image) -> Optional[str]:
        """Generate a descriptive caption for the given image."""
        if not self.processor or not self.model:
            self.logger.error("Image captioning model not initialized")
            return None
        try:
            # Convert image to RGB if needed
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            inputs = self.processor(image, return_tensors="pt")
            if self.device == 'cuda' and torch.cuda.is_available():
                inputs = {k: v.to('cuda') for k, v in inputs.items()}
            
            with torch.no_grad():
                out = self.model.generate(**inputs)
            
            caption = self.processor.decode(out[0], skip_special_tokens=True)
            self.logger.info(f"Caption generated: {caption}")
            return caption
        except Exception as e:
            self.logger.error(f"Failed to generate caption: {e}")
            return None

    def get_detected_text(self, image: Optional[Image.Image] = None, ocr_results: Optional[List[Dict]] = None) -> str:
        """
        Get all detected text as a single string.
        
        Args:
            image: PIL Image object (for backward compatibility)
            ocr_results: List of OCR result dictionaries (preferred method)
            
        Returns:
            Combined text string
        """
        # Use OCR results directly
        if ocr_results is not None:
            return ' '.join([result['text'] for result in ocr_results])

        # Perform OCR on image
        if image is not None and self.ocr_reader:
            try:
                image_np = np.array(image.convert('RGB'))
                results = self.ocr_reader.readtext(image_np, detail=0, paragraph=True)
                return ' '.join(results)
            except Exception as e:
                self.logger.error(f"Failed to perform OCR: {e}")
                return ""
        
        return ""

    def get_text_with_positions(self, ocr_results: List[Dict]) -> List[Tuple[str, Tuple]]:
        """
        Get text with their bounding box positions.
        
        Args:
            ocr_results: List of OCR result dictionaries
            
        Returns:
            List of tuples containing (text, bbox)
        """
        return [(result['text'], result['bbox']) for result in ocr_results]

    def process_screen(self, monitor_index: int = 1, save_screenshot: bool = False, 
                      screenshot_path: str = None, confidence_threshold: float = 0.5, 
                      ocr_scale_factor: float = 0.5, skip_ocr: bool = False) -> Dict:
        """
        Complete screen processing: capture screenshot, perform OCR, and generate caption.
        
        Args:
            monitor_index: Index of the monitor to capture
            save_screenshot: Whether to save the screenshot
            screenshot_path: Path to save screenshot if save_screenshot is True
            confidence_threshold: Minimum confidence for OCR
            ocr_scale_factor: Factor to scale down image for OCR processing (0.5 = half size)
            skip_ocr: Whether to skip OCR processing and only generate caption
            
        Returns:
            Dictionary containing screenshot, OCR results, and caption
        """
        result = {
            'screenshot': None,
            'ocr_results': [],
            'caption': None,
            'success': False
        }
        
        # Capture screenshot
        screenshot = self.capture_screenshot(monitor_index, screenshot_path if save_screenshot else None)
        
        if screenshot is None:
            self.logger.error("Failed to capture screenshot")
            return result
        
        result['screenshot'] = screenshot
        
        # Perform OCR unless skipped
        if not skip_ocr:
            # Create path for scaled image if screenshot is being saved
            scaled_image_path = None
            if save_screenshot and screenshot_path:
                # Create a scaled version filename
                base_name, ext = os.path.splitext(screenshot_path)
                scaled_image_path = f"{base_name}_scaled{ext}"
            
            # Perform OCR with scaled image for faster processing
            ocr_results = self.perform_ocr(screenshot, confidence_threshold, ocr_scale_factor, 
                                         save_scaled_image=save_screenshot, scaled_image_path=scaled_image_path)
            result['ocr_results'] = ocr_results
        else:
            # Skip OCR processing - return empty results
            result['ocr_results'] = []
            self.logger.info("OCR processing skipped")
        
        # Generate caption
        caption = self.generate_caption(screenshot)
        result['caption'] = caption
        
        result['success'] = True
        self.logger.info("Screen processing completed successfully")
        
        return result


# Example usage
if __name__ == "__main__":
    import time
    
    # Initialize vision input
    start_time = time.time()
    vision_input = VisionInput()
    
    # Get monitor information
    monitors = vision_input.get_monitors()
    print(f"Available monitors: {monitors}")
    print(f"Vision input initialized in {time.time() - start_time:.2f} seconds")
    
    # Process screen with scaled OCR for faster processing
    start_time = time.time()
    result = vision_input.process_screen(
        monitor_index=1,
        save_screenshot=True,
        screenshot_path="screen_capture.png",
        confidence_threshold=0.5,
        ocr_scale_factor=0.5  # Scale image to 50% for faster OCR
    )
    print(f"Screen processed in {time.time() - start_time:.2f} seconds")
    
    if result['success']:
        print("Screenshot captured successfully")
        detected_text = vision_input.get_detected_text(ocr_results=result['ocr_results'])
        print(f"Detected text: {detected_text}")
        print(f"Image caption: {result['caption']}")
        
        # Get text with positions
        text_positions = vision_input.get_text_with_positions(result['ocr_results'])
        print(f"Found {len(text_positions)} text regions with positions")
    else:
        print("Failed to process screen")