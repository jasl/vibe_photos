"""
Configuration constants for the AI Photo Management System (v2).
"""

import os

# --- Model Configuration (v3.2) ---
# State-of-the-Art model checkpoints

# CRITICAL: Use SigLIP-Giant for ALL embeddings (consistency fix)
EMBEDDING_MODEL_ID = "google/siglip2-giant-opt-patch16-384"  # Upgraded from so400m
DETECTION_MODEL_ID = "PekingU/rtdetr_r101vd"
CAPTION_MODEL_ID = "Qwen/Qwen2-VL-7B-Instruct"  # Upgraded from BLIP2
TAGGER_MODEL_ID = "vblagoje/bert-english-uncased-finetuned-pos"  # For tag extraction

# Model parameters
EMBEDDING_DIM = 1152  # Fixed dimension for SigLIP2 embeddings

# Similarity thresholds (v2)
VISUAL_SIMILARITY_THRESHOLD = 0.98  # For visual duplicate detection (very strict)
FACE_SIMILARITY_THRESHOLD = 0.92  # For face recognition (more lenient)
SEMANTIC_SIMILARITY_THRESHOLD = 0.95  # For merging groups with similar captions

# --- Data Paths ---
PHOTOS_DIR = "/home/jasl/datasets/my_photos"
DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
FAISS_IMAGE_INDEX_PATH = os.path.join(DATA_DIR, "photo_library.index")
FAISS_FACE_INDEX_PATH = os.path.join(DATA_DIR, "face_library.index")
SQLITE_DB_PATH = os.path.join(DATA_DIR, "photo_metadata.db")

# --- Processing Configuration ---
SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".gif", ".webp", ".JPG", ".JPEG", ".PNG", ".GIF", ".WEBP"}
DETECTION_THRESHOLD = 0.9  # Confidence threshold for object detection
MAX_CAPTION_TOKENS = 100  # Max tokens for caption generation (increased for Qwen)

# --- Flask Configuration ---
FLASK_HOST = "0.0.0.0"
FLASK_PORT = 5000
FLASK_DEBUG = True

# Ensure data directory exists
os.makedirs(DATA_DIR, exist_ok=True)
