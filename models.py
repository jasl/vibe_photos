"""
AI Model initialization and management (v2).
Loads SigLIP2 (embeddings), RT-DETR (object detection), Qwen2-VL (captioning), and POS tagger (tag extraction).
"""

import torch
from transformers import (
    AutoProcessor,
    AutoModel,
    AutoImageProcessor,
    AutoModelForObjectDetection,
    Qwen2VLForConditionalGeneration,
    pipeline,
)

from config import EMBEDDING_MODEL_ID, DETECTION_MODEL_ID, CAPTION_MODEL_ID, TAGGER_MODEL_ID


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
        self.pos_tagger = None
        
    def load_all_models(self):
        """
        Load all SOTA models (v2): SigLIP2, RT-DETR, Qwen2-VL, and POS tagger.
        Uses device_map="auto" for efficient VRAM distribution.
        """
        print("\n=== Loading AI Models (v2) ===")
        
        # Model 1: SigLIP2 for embeddings (text + image)
        print(f"\n[1/4] Loading Embedding Model: {EMBEDDING_MODEL_ID}")
        self.embed_processor = AutoProcessor.from_pretrained(EMBEDDING_MODEL_ID, use_fast=True)
        self.embed_model = AutoModel.from_pretrained(
            EMBEDDING_MODEL_ID,
            dtype="auto",
            device_map="auto"
        ).eval()
        print("✓ Embedding model loaded successfully")
        
        # Model 2: RT-DETR for object detection
        print(f"\n[2/4] Loading Object Detection Model: {DETECTION_MODEL_ID}")
        self.detect_processor = AutoImageProcessor.from_pretrained(DETECTION_MODEL_ID, use_fast=True)
        self.detect_model = AutoModelForObjectDetection.from_pretrained(
            DETECTION_MODEL_ID,
            dtype="auto",
            device_map="auto"
        ).eval()
        print("✓ Object detection model loaded successfully")
        
        # Model 3: Qwen2-VL for generative captioning (upgraded from BLIP2)
        print(f"\n[3/4] Loading Caption Generation Model: {CAPTION_MODEL_ID}")
        self.caption_processor = AutoProcessor.from_pretrained(
            CAPTION_MODEL_ID,
            trust_remote_code=True
        )
        self.caption_model = Qwen2VLForConditionalGeneration.from_pretrained(
            CAPTION_MODEL_ID,
            dtype="auto",
            device_map="auto",
            trust_remote_code=True
        ).eval()
        print("✓ Caption generation model loaded successfully (Qwen2-VL)")
        
        # Model 4: POS Tagger for tag extraction
        print(f"\n[4/4] Loading POS Tagger for Tag Extraction: {TAGGER_MODEL_ID}")
        self.pos_tagger = pipeline(
            "token-classification",
            model=TAGGER_MODEL_ID,
            device=self.device
        )
        print("✓ POS tagger loaded successfully")
        
        print("\n=== All models loaded successfully (v2) ===\n")


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
