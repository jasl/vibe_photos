"""
Configuration constants for the AI Photo Management System.
"""

import os

# --- Model Configuration ---
# State-of-the-Art model checkpoints
EMBEDDING_MODEL_ID = "google/siglip2-so400m-patch14-384"
DETECTION_MODEL_ID = "PekingU/rtdetr_r101vd"
CAPTION_MODEL_ID = "Salesforce/blip2-opt-2.7b"

# Model parameters
EMBEDDING_DIM = 1152  # Fixed dimension for SigLIP2 embeddings
SIMILARITY_THRESHOLD = 0.98  # Threshold for duplicate detection (0.98+ = very similar)

# --- Data Paths ---
PHOTOS_DIR = "/home/jasl/datasets/my_photos"
DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
FAISS_INDEX_PATH = os.path.join(DATA_DIR, "photo_library.index")
SQLITE_DB_PATH = os.path.join(DATA_DIR, "photo_metadata.db")

# --- Processing Configuration ---
SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".gif", ".webp", ".JPG", ".JPEG", ".PNG", ".GIF", ".WEBP"}
DETECTION_THRESHOLD = 0.9  # Confidence threshold for object detection
MAX_CAPTION_TOKENS = 50  # Max tokens for caption generation

# --- Flask Configuration ---
FLASK_HOST = "0.0.0.0"
FLASK_PORT = 5000
FLASK_DEBUG = True

# Ensure data directory exists
os.makedirs(DATA_DIR, exist_ok=True)
