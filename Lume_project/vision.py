import os
import torch
import mss
import easyocr
import logging
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

    def capture_screenshot(self, monitor_index: int = 1) -> Optional[Image.Image]:
        """Captures a screenshot of the specified monitor."""
        try:
            with mss.mss() as sct:
                monitors = sct.monitors
                if not (0 < monitor_index < len(monitors)):
                    self.logger.warning(f"Monitor index {monitor_index} is invalid. Defaulting to primary.")
                    monitor_index = 1 if len(monitors) > 1 else 0

                screenshot = sct.grab(monitors[monitor_index])
                return Image.frombytes("RGB", screenshot.size, screenshot.rgb)
        except Exception as e:
            self.logger.error(f"Failed to capture screenshot: {e}\n{traceback.format_exc()}")
            return None

    def generate_caption(self, image: Image.Image) -> Optional[str]:
        """Generates a descriptive caption for the given image."""
        if not self.processor or not self.model:
            return None
        try:
            image = image.convert('RGB')
            inputs = self.processor(image, return_tensors="pt")
            if self.device == 'cuda' and torch.cuda.is_available():
                inputs = {k: v.to('cuda') for k, v in inputs.items()}
            
            with torch.no_grad():
                out = self.model.generate(**inputs)
            
            return self.processor.decode(out[0], skip_special_tokens=True)
        except Exception as e:
            self.logger.error(f"Failed to generate caption: {e}")
            return None

    def get_detected_text(self, image: Image.Image) -> str:
        """Performs OCR and returns all detected text as a single string."""
        if not self.ocr_reader:
            return ""
        try:
            image_array = memoryview(image.tobytes())
            results = self.ocr_reader.readtext(image, detail=0, paragraph=True)
            return ' '.join(results)
        except Exception as e:
            self.logger.error(f"Failed to perform OCR: {e}")
            return ""