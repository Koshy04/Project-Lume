import os
import torch
import mss
import easyocr
import numpy as np
import traceback
import config 
from PIL import Image, ImageEnhance, ImageFilter
from transformers import Blip2Processor, Blip2ForConditionalGeneration, BitsAndBytesConfig
from typing import Optional, List, Dict, Tuple
from src.log.custom_logger import logger

class VisionInput:
    """Enhanced vision module for screen capture, OCR, and image captioning."""
    _instance = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super(VisionInput, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self, languages: Optional[List[str]] = None, device: Optional[str] = None, 
                 gpu_acceleration: Optional[bool] = None):
        if self._initialized:
            return
            
        self.device = device if device is not None else config.BLIP2_DEVICE
        self.gpu_acceleration = gpu_acceleration if gpu_acceleration is not None else config.OCR_GPU_ACCELERATION
        _languages = languages if languages is not None else config.VISION_LANGUAGES
        
        self.ocr_reader = None
        self.processor = None
        self.model = None

        logger.info(f"Initializing VisionInput on device: '{self.device}'...")
        
        try:
            self.ocr_reader = easyocr.Reader(_languages, gpu=self.gpu_acceleration and torch.cuda.is_available())
            logger.info(f"OCR reader initialized for {_languages}, GPU: {self.gpu_acceleration and torch.cuda.is_available()}")
        except Exception as e: logger.error(f"Failed to initialize OCR reader: {e}")

        try:
            model_id = config.BLIP_MODEL_ID
            self.processor = Blip2Processor.from_pretrained(model_id)
            if self.device == 'cuda' and torch.cuda.is_available():
                quantization_config = BitsAndBytesConfig(
                    load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True)
                self.model = Blip2ForConditionalGeneration.from_pretrained(model_id, quantization_config=quantization_config, device_map="auto")
            else:
                self.model = Blip2ForConditionalGeneration.from_pretrained(model_id, use_fast=True)
            logger.info(f"Image captioning model initialized: {model_id}")
        except Exception as e: logger.error(f"Failed to initialize image captioning model: {e}")
        
        self._initialized = True

    def unload_model(self):
        """Releases model resources from memory and clears CUDA cache."""
        if not self._initialized:
            logger.info("Vision models are already unloaded.")
            return

        logger.info("Unloading vision models from memory...")
        del self.model
        del self.processor
        del self.ocr_reader
        self.model = None
        self.processor = None
        self.ocr_reader = None
        
        if self.device == 'cuda' and torch.cuda.is_available():
            torch.cuda.empty_cache()
            logger.info("CUDA cache cleared.")
        
        self._initialized = False

    def get_monitors(self) -> List[Dict]:
        """Get current monitor information for debugging."""
        try:
            with mss.mss() as sct: return sct.monitors
        except Exception as e:
            logger.error(f"Failed to get monitors: {e}, {traceback.format_exc()}"); return []

    def preprocess_image(self, image: Image.Image, **kwargs) -> Image.Image:
        """Preprocess image to improve OCR accuracy."""
        try:
            processed = image.copy().convert('RGB')
            if kwargs.get('enhance_contrast', 1.0) != 1.0: processed = ImageEnhance.Contrast(processed).enhance(kwargs['enhance_contrast'])
            if kwargs.get('enhance_sharpness', 1.0) != 1.0: processed = ImageEnhance.Sharpness(processed).enhance(kwargs['enhance_sharpness'])
            if kwargs.get('enhance_brightness', 1.0) != 1.0: processed = ImageEnhance.Brightness(processed).enhance(kwargs['enhance_brightness'])
            if kwargs.get('denoise', False): processed = processed.filter(ImageFilter.MedianFilter(size=3))
            if kwargs.get('binarize', False):
                processed = processed.convert('L').point(lambda x: 255 if x > kwargs.get('binarize_threshold', 128) else 0, mode='1').convert('RGB')
            return processed
        except Exception as e: logger.error(f"Failed to preprocess image: {e}"); return image

    def capture_screenshot(self, monitor_index: int = 1, save_path: Optional[str] = None, region: Optional[Dict[str, int]] = None) -> Optional[Image.Image]:
        """Capture a screenshot of the specified monitor or region."""
        try:
            with mss.mss() as sct:
                monitors = sct.monitors
                if not (0 <= monitor_index < len(monitors)):
                    logger.warning(f"Monitor index {monitor_index} invalid. Defaulting to 1 if available, else 0.")
                    monitor_index = 1 if len(monitors) > 1 else 0
                
                monitor = region or monitors[monitor_index]
                screenshot = sct.grab(monitor)
                img = Image.frombytes("RGB", (screenshot.width, screenshot.height), screenshot.rgb)
                if save_path: img.save(save_path); logger.info(f"Screenshot saved to: {save_path}")
                return img
        except Exception as e: logger.error(f"Failed to capture screenshot: {e}\n{traceback.format_exc()}"); return None

    def perform_ocr(self, image: Image.Image, **kwargs) -> List[Dict]:
        """Perform OCR on the given image with advanced options."""
        if not self.ocr_reader: logger.error("OCR reader not initialized"); return []
        
        conf_thresh = kwargs.get('confidence_threshold', config.OCR_CONFIDENCE_THRESHOLD)
        scale = kwargs.get('scale_factor', config.OCR_SCALE_FACTOR)
        
        try:
            processed_image = self.preprocess_image(image.copy(), **(kwargs.get('preprocessing_options') or config.OCR_DEFAULT_PREPROCESSING_OPTIONS)) if kwargs.get('preprocess', config.OCR_APPLY_PREPROCESSING) else image.copy()
            
            if scale != 1.0: processed_image = processed_image.resize((int(image.width * scale), int(image.height * scale)), Image.Resampling.LANCZOS)
            
            results = self.ocr_reader.readtext(np.array(processed_image), detail=config.OCR_DETAIL_LEVEL, paragraph=config.OCR_PARAGRAPH_MODE, batch_size=config.OCR_BATCH_SIZE)
            
            if config.OCR_DETAIL_LEVEL == 0: return [{'text': text} for text in results]
            
            filtered = []
            for bbox, text, conf in results:
                if conf >= conf_thresh:
                    scaled_bbox = [(int(p[0] / scale), int(p[1] / scale)) for p in bbox] if scale != 1.0 else bbox
                    filtered.append({'text': text.strip(), 'bbox': scaled_bbox, 'confidence': float(conf), 'center': (int(sum(p[0] for p in scaled_bbox)/4), int(sum(p[1] for p in scaled_bbox)/4))})
            return filtered
        except Exception as e: logger.error(f"Failed to perform OCR: {e}\n{traceback.format_exc()}"); return []

    def generate_caption(self, image: Image.Image, **kwargs) -> Optional[str]:
        """Generate a descriptive caption for the given image."""
        if not self.processor or not self.model:
            logger.error("Caption generation failed: Vision model or processor is not available (likely unloaded).")
            return None
        
        prompt = kwargs.get('prompt') or "A detailed and comprehensive description of the image, including all visible objects, text, and the overall scene context is:"
        
        try:
            inputs = self.processor(image.convert('RGB'), text=prompt, return_tensors="pt")
            if self.device == 'cuda' and torch.cuda.is_available(): inputs = {k: v.to('cuda') for k, v in inputs.items()}
            
            with torch.no_grad():
                out = self.model.generate(**inputs, max_length=kwargs.get('max_length', config.CAPTION_MAX_LENGTH), num_beams=kwargs.get('num_beams', config.CAPTION_NUM_BEAMS), temperature=kwargs.get('temperature', config.CAPTION_TEMPERATURE), do_sample=kwargs.get('temperature') is not None)
            
            if out is None:
                logger.error("Caption generation failed: model.generate() returned None.")
                return None

            return self.processor.decode(out[0], skip_special_tokens=True).strip()
        except Exception as e: 
            logger.error(f"Failed to generate caption: {e}"); return None

    def get_detected_text(self, ocr_results: List[Dict], sorted_by_position: bool = True) -> str:
        """Get all detected text as a single string from OCR results."""
        if sorted_by_position and ocr_results:
            ocr_results = sorted(ocr_results, key=lambda x: (x.get('center', (0,0))[1], x.get('center', (0,0))[0]))
        return ' '.join([result['text'] for result in ocr_results])

    def process_screen(self, monitor_index: Optional[int] = None, **kwargs) -> Dict:
        """Complete screen processing: capture, OCR, and caption."""
        result = {'screenshot': None, 'ocr_results': [], 'caption': None, 'success': False, 'text': ''}
        
        # Pass region kwarg through to the capture function
        screenshot = self.capture_screenshot(monitor_index or config.DEFAULT_MONITOR, region=kwargs.get('region'))
        if screenshot is None: logger.error("Failed to capture screenshot"); return result
        
        result['screenshot'] = screenshot
        if not kwargs.get('skip_ocr'):
            result['ocr_results'] = self.perform_ocr(screenshot, **kwargs)
            result['text'] = self.get_detected_text(result['ocr_results'])
        
        if not kwargs.get('skip_caption'):
            result['caption'] = self.generate_caption(screenshot, **kwargs)
            
        result['success'] = True
        return result