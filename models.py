"""
AI Model initialization and management.
Loads SigLIP2 (embeddings), RT-DETR (object detection), and BLIP2 (captioning).
"""

import torch
from transformers import (
    AutoProcessor,
    AutoModel,
    AutoImageProcessor,
    AutoModelForObjectDetection,
    Blip2ForConditionalGeneration,
)

from config import EMBEDDING_MODEL_ID, DETECTION_MODEL_ID, CAPTION_MODEL_ID


def get_inference_device():
    """
    Auto-detect the best available device for inference.
    Returns: device
    """
    if torch.cuda.is_available():
        print("Using CUDA (NVIDIA GPU)")
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        print("Using MPS (Apple Silicon GPU)")
        return torch.device("mps")
    print("No GPU detected. Using CPU (float32).")
    return torch.device("cpu")


class AIModels:
    """
    Container class for all AI models used in the photo management system.
    """
    
    def __init__(self):
        self.device = get_inference_device()
        self.embed_processor = None
        self.embed_model = None
        self.detect_processor = None
        self.detect_model = None
        self.caption_processor = None
        self.caption_model = None
        
    def load_all_models(self):
        """
        Load all three SOTA models: SigLIP2, RT-DETR, and BLIP2.
        Uses device_map="auto" for efficient VRAM distribution.
        """
        print("\n=== Loading AI Models ===")
        
        # Model 1: SigLIP2 for embeddings (text + image)
        print(f"\n[1/3] Loading Embedding Model: {EMBEDDING_MODEL_ID}")
        self.embed_processor = AutoProcessor.from_pretrained(EMBEDDING_MODEL_ID, use_fast=True)
        self.embed_model = AutoModel.from_pretrained(
            EMBEDDING_MODEL_ID,
            dtype="auto",
            device_map="auto"
        ).eval()
        print("✓ Embedding model loaded successfully")
        
        # Model 2: RT-DETR for object detection
        print(f"\n[2/3] Loading Object Detection Model: {DETECTION_MODEL_ID}")
        self.detect_processor = AutoImageProcessor.from_pretrained(DETECTION_MODEL_ID, use_fast=True)
        self.detect_model = AutoModelForObjectDetection.from_pretrained(
            DETECTION_MODEL_ID,
            dtype="auto",
            device_map="auto"
        ).eval()
        print("✓ Object detection model loaded successfully")
        
        # Model 3: BLIP2 for generative captioning
        print(f"\n[3/3] Loading Caption Generation Model: {CAPTION_MODEL_ID}")
        self.caption_processor = AutoProcessor.from_pretrained(CAPTION_MODEL_ID, use_fast=True)
        self.caption_model = Blip2ForConditionalGeneration.from_pretrained(
            CAPTION_MODEL_ID,
            dtype="auto",
            device_map="auto"
        ).eval()
        print("✓ Caption generation model loaded successfully")
        
        print("\n=== All models loaded successfully ===\n")


# Global instance
_models_instance = None


def get_models():
    """
    Get or create the global AIModels instance.
    This ensures models are only loaded once.
    """
    global _models_instance
    if _models_instance is None:
        _models_instance = AIModels()
        _models_instance.load_all_models()
    return _models_instance
