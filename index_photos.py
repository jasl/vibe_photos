#!/usr/bin/env python3
"""
Fast Ingest Script (Tier 1) - V3 Cascade Architecture
Quickly processes photos with lightweight models for duplicate detection and queuing.

Uses ~12GB VRAM:
- SigLIP-so400m (1B) for embeddings
- BLIP2-opt-2.7b (2.7B) for fast captions

Workflow:
1. Extract EXIF metadata (timestamp, GPS)
2. Visual deduplication (image embeddings)
3. Semantic deduplication (caption embeddings)
4. Queue unique photos for deep processing
"""

import os
import sys
import torch
import faiss
import numpy as np
from datetime import datetime
from pathlib import Path
from tqdm import tqdm
from PIL import Image
from PIL.ExifTags import TAGS, GPSTAGS

from transformers import (
    AutoProcessor,
    AutoModel,
    Blip2ForConditionalGeneration,
    Blip2Processor
)

from config import (
    PHOTOS_DIR,
    SUPPORTED_EXTENSIONS,
    VISUAL_SIMILARITY_THRESHOLD,
    SEMANTIC_SIMILARITY_THRESHOLD,
    EMBEDDING_MODEL_ID,
    EMBEDDING_DIM
)
from database import get_database, CAPTION_INDEX_PATH

# Fast caption model for tier 1
FAST_CAPTION_MODEL_ID = "Salesforce/blip2-opt-2.7b"


def get_inference_device_and_dtype():
    """Auto-detect device and dtype."""
    if torch.cuda.is_available():
        print("Using CUDA (NVIDIA GPU) - float16")
        return torch.device("cuda"), torch.float16
    if torch.backends.mps.is_available():
        print("Using MPS (Apple Silicon GPU) - float16")
        return torch.device("mps"), torch.float16
    print("Using CPU - float32")
    return torch.device("cpu"), torch.float32


class FastIngestProcessor:
    """Handles fast tier-1 ingest with EXIF extraction and dual deduplication."""
    
    def __init__(self):
        self.device, self.dtype = get_inference_device_and_dtype()
        self.db = get_database()
        self.embed_processor = None
        self.embed_model = None
        self.caption_processor = None
        self.caption_model = None
        
    def load_models(self):
        """Load lightweight models for fast ingest."""
        print("\n=== Loading Tier-1 (Fast) Models ===")
        
        # Model 1: SigLIP-so400m for embeddings
        print(f"\n[1/2] Loading Embedding Model: {EMBEDDING_MODEL_ID}")
        self.embed_processor = AutoProcessor.from_pretrained(EMBEDDING_MODEL_ID)
        self.embed_model = AutoModel.from_pretrained(
            EMBEDDING_MODEL_ID,
            dtype="auto",
            device_map="auto"
        ).eval()
        print("✓ Embedding model loaded")
        
        # Model 2: BLIP2 for fast captions
        print(f"\n[2/2] Loading Fast Caption Model: {FAST_CAPTION_MODEL_ID}")
        self.caption_processor = Blip2Processor.from_pretrained(FAST_CAPTION_MODEL_ID)
        self.caption_model = Blip2ForConditionalGeneration.from_pretrained(
            FAST_CAPTION_MODEL_ID,
            dtype="auto",
            device_map="auto"
        ).eval()
        print("✓ Fast caption model loaded")
        
        print("\n=== Tier-1 models ready ===\n")
        
    def get_exif_data(self, image: Image.Image, image_path: str) -> dict:
        """
        Extract EXIF metadata: timestamp and GPS coordinates.
        Falls back to file creation time if EXIF unavailable.
        """
        timestamp = None
        latitude = None
        longitude = None
        
        try:
            # Fallback: File creation time
            ctime = os.path.getctime(image_path)
            timestamp = datetime.fromtimestamp(ctime).isoformat()
            
            # Try to get EXIF data
            exif_data = image._getexif()
            if exif_data:
                # Get shooting time (DateTimeOriginal)
                dt_original = exif_data.get(36867)  # 36867 = DateTimeOriginal
                if dt_original:
                    try:
                        dt_obj = datetime.strptime(dt_original, '%Y:%m:%d %H:%M:%S')
                        timestamp = dt_obj.isoformat()
                    except:
                        pass
                
                # Get GPS info
                gps_info = exif_data.get(34853)  # 34853 = GPSInfo
                if gps_info:
                    try:
                        lat_dms = gps_info.get(2)  # Latitude
                        lat_ref = gps_info.get(1)  # N or S
                        lon_dms = gps_info.get(4)  # Longitude
                        lon_ref = gps_info.get(3)  # E or W
                        
                        if lat_dms and lat_ref and lon_dms and lon_ref:
                            # Convert DMS to decimal
                            latitude = float(lat_dms[0]) + float(lat_dms[1])/60 + float(lat_dms[2])/3600
                            if lat_ref in ['S', 's']:
                                latitude *= -1
                                
                            longitude = float(lon_dms[0]) + float(lon_dms[1])/60 + float(lon_dms[2])/3600
                            if lon_ref in ['W', 'w']:
                                longitude *= -1
                    except:
                        pass
                        
        except Exception as e:
            pass  # Use defaults
            
        return {
            "timestamp": timestamp,
            "latitude": latitude,
            "longitude": longitude
        }
        
    @torch.no_grad()
    def get_image_embedding(self, image: Image.Image) -> np.ndarray:
        """Create L2-normalized image embedding."""
        inputs = self.embed_processor(
            images=image,
            return_tensors="pt"
        ).to(self.device, self.dtype)
        
        image_features = self.embed_model.get_image_features(**inputs)
        vector = image_features.cpu().numpy().astype(np.float32)
        faiss.normalize_L2(vector)
        return vector
        
    @torch.no_grad()
    def get_text_embedding(self, text: str) -> np.ndarray:
        """Create L2-normalized text embedding."""
        inputs = self.embed_processor(
            text=[text],
            return_tensors="pt",
            padding=True,
            truncation=True
        ).to(self.device, self.dtype)
        
        text_features = self.embed_model.get_text_features(**inputs)
        vector = text_features.cpu().numpy().astype(np.float32)
        faiss.normalize_L2(vector)
        return vector
        
    @torch.no_grad()
    def get_fast_caption(self, image: Image.Image) -> str:
        """Generate quick caption with BLIP2."""
        inputs = self.caption_processor(
            image,
            return_tensors="pt"
        ).to(self.device, self.dtype)
        
        generated_ids = self.caption_model.generate(**inputs, max_new_tokens=50)
        caption = self.caption_processor.batch_decode(
            generated_ids,
            skip_special_tokens=True
        )[0].strip()
        
        return caption
        
    def ingest_new_image(self, image_path: str) -> bool:
        """
        Fast ingest workflow with dual deduplication.
        
        Returns:
            True if successful, False if failed
        """
        # Check if already processed
        self.db.cursor.execute(
            "SELECT group_id FROM images WHERE image_path = ?",
            (image_path,)
        )
        if self.db.cursor.fetchone():
            return True  # Already processed
            
        try:
            image = Image.open(image_path).convert("RGB")
        except Exception as e:
            print(f"  > FAILED to load: {e}")
            return False
            
        # Stage 1: Extract EXIF metadata
        exif_data = self.get_exif_data(image, image_path)
        
        # Stage 2: Visual deduplication
        image_embedding = self.get_image_embedding(image)
        
        if self.db.faiss_image_index.ntotal > 0:
            scores, faiss_ids = self.db.faiss_image_index.search(image_embedding, k=1)
            visual_score = scores[0][0]
            matched_image_faiss_id = faiss_ids[0][0]
        else:
            visual_score = 0.0
            matched_image_faiss_id = -1
            
        if visual_score > VISUAL_SIMILARITY_THRESHOLD:
            # Visual duplicate found
            print(f"  > Visual Duplicate (score: {visual_score:.4f})")
            self.db.cursor.execute(
                "SELECT group_id, canonical_path FROM images i JOIN image_groups g ON i.group_id = g.group_id WHERE i.faiss_image_id = ?",
                (int(matched_image_faiss_id),)
            )
            result = self.db.cursor.fetchone()
            if result:
                group_id = result[0]
                duplicate_of_path = result[1]
                # V3.2: Store duplicate debugging information
                self.db.cursor.execute(
                    "INSERT OR REPLACE INTO images (image_path, group_id, faiss_image_id, duplicate_of_path, duplicate_score) VALUES (?, ?, ?, ?, ?)",
                    (image_path, group_id, None, duplicate_of_path, visual_score)
                )
                self.db.db_conn.commit()
                print(f"  > Added to group {group_id}, duplicate of {Path(duplicate_of_path).name}")
                return True
                
        # Stage 3: Semantic deduplication
        print(f"  > Visually Unique (score: {visual_score:.4f}). Generating fast caption...")
        fast_caption = self.get_fast_caption(image)
        print(f"  > Fast Caption: '{fast_caption}'")
        
        caption_embedding = self.get_text_embedding(fast_caption)
        
        if self.db.faiss_caption_index.ntotal > 0:
            scores, faiss_ids = self.db.faiss_caption_index.search(caption_embedding, k=1)
            semantic_score = scores[0][0]
            matched_caption_faiss_id = faiss_ids[0][0]
        else:
            semantic_score = 0.0
            matched_caption_faiss_id = -1
            
        if semantic_score > SEMANTIC_SIMILARITY_THRESHOLD:
            # Semantic duplicate found
            print(f"  > Semantic Duplicate (score: {semantic_score:.4f})")
            # IMPORTANT: Caption index and image index share the same IDs
            # The matched_caption_faiss_id corresponds to the same position in image_index
            # Both indexes were added with the same ID when the group was created
            self.db.cursor.execute(
                "SELECT group_id, canonical_path FROM images i JOIN image_groups g ON i.group_id = g.group_id WHERE i.faiss_image_id = ?",
                (int(matched_caption_faiss_id),)
            )
            result = self.db.cursor.fetchone()
            if result:
                group_id = result[0]
                duplicate_of_path = result[1]
                # V3.2: Store duplicate debugging information
                self.db.cursor.execute(
                    "INSERT OR REPLACE INTO images (image_path, group_id, faiss_image_id, duplicate_of_path, duplicate_score) VALUES (?, ?, ?, ?, ?)",
                    (image_path, group_id, None, duplicate_of_path, semantic_score)
                )
                self.db.db_conn.commit()
                print(f"  > Added to group {group_id}, semantically similar to {Path(duplicate_of_path).name}")
                return True
            else:
                # If no match found, this is an index desync issue
                # Treat as unique to avoid wrong group assignment
                print(f"  > WARNING: Caption index match but no corresponding image index entry")
                semantic_score = 0.0  # Force creation of new group
                
        # Stage 4: New unique image - create group and queue
        print(f"  > Semantically Unique (score: {semantic_score:.4f}). Queuing for deep processing...")
        
        new_group_id = f"group_{self.db.faiss_image_index.ntotal}"
        new_faiss_image_id = self.db.faiss_image_index.ntotal
        
        # Add to FAISS indexes
        self.db.faiss_image_index.add(image_embedding)
        self.db.faiss_caption_index.add(caption_embedding)
        
        # Add to images table
        self.db.cursor.execute(
            "INSERT OR REPLACE INTO images (image_path, group_id, faiss_image_id) VALUES (?, ?, ?)",
            (image_path, new_group_id, int(new_faiss_image_id))
        )
        
        # Create image group with fast caption and EXIF data
        self.db.cursor.execute(
            """
            INSERT OR REPLACE INTO image_groups 
            (group_id, canonical_path, fast_caption, processing_status, taken_at_timestamp, gps_latitude, gps_longitude)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                new_group_id,
                image_path,
                fast_caption,
                'QUEUED',
                exif_data["timestamp"],
                exif_data["latitude"],
                exif_data["longitude"]
            )
        )
        
        # Add to processing queue
        self.db.cursor.execute(
            "INSERT OR REPLACE INTO processing_queue (group_id) VALUES (?)",
            (new_group_id,)
        )
        
        self.db.db_conn.commit()
        print(f"  > Created Group {new_group_id}, queued for deep processing")
        return True


def find_all_images(root_dir: str) -> list:
    """Recursively find all image files."""
    image_paths = []
    
    print(f"\nScanning for images in: {root_dir}")
    
    for root, dirs, files in os.walk(root_dir):
        for file in files:
            if Path(file).suffix in SUPPORTED_EXTENSIONS:
                full_path = os.path.join(root, file)
                image_paths.append(full_path)
                
    print(f"Found {len(image_paths)} images")
    return image_paths


def main():
    """Main fast ingest workflow."""
    print("\n" + "="*60)
    print("AI Photo Management - Fast Ingest (Tier 1)")
    print("="*60)
    
    if not os.path.exists(PHOTOS_DIR):
        print(f"\nERROR: Photos directory not found: {PHOTOS_DIR}")
        sys.exit(1)
        
    # Find all images
    image_paths = find_all_images(PHOTOS_DIR)
    
    if not image_paths:
        print("\nNo images found!")
        sys.exit(1)
        
    # Initialize processor
    print("\nInitializing fast ingest processor...")
    processor = FastIngestProcessor()
    processor.load_models()
    
    # Check which images are already processed
    print("\nChecking for already processed images...")
    processor.db.cursor.execute("SELECT image_path FROM images")
    processed_paths = {row[0] for row in processor.db.cursor.fetchall()}
    
    images_to_process = [p for p in image_paths if p not in processed_paths]
    
    print(f"\nTotal images: {len(image_paths)}")
    print(f"Already processed: {len(processed_paths)}")
    print(f"To process: {len(images_to_process)}")
    
    if not images_to_process:
        print("\nAll images already processed!")
        stats = processor.db.get_stats()
        print(f"\nStats:")
        print(f"  Total images: {stats['total_images']}")
        print(f"  Unique groups: {stats['unique_groups']}")
        print(f"  Queued for deep processing: {stats['queued_for_processing']}")
        return
        
    # Process images
    print("\nProcessing images (fast tier)...")
    print("(This should be 5-10x faster than v2)\n")
    
    success_count = 0
    error_count = 0
    
    with tqdm(total=len(images_to_process), unit="img") as pbar:
        for image_path in images_to_process:
            pbar.set_description(f"Ingesting {Path(image_path).name[:30]}")
            
            success = processor.ingest_new_image(image_path)
            
            if success:
                success_count += 1
            else:
                error_count += 1
                
            pbar.update(1)
            
            # Save progress every 100 images
            if success_count % 100 == 0:
                processor.db.save_databases()
                
    # Final save
    print("\nSaving databases...")
    processor.db.save_databases()
    
    # Print stats
    print("\n" + "="*60)
    print("Fast Ingest Complete!")
    print("="*60)
    
    stats = processor.db.get_stats()
    
    print(f"\nProcessing Summary:")
    print(f"  Successfully processed: {success_count}")
    print(f"  Errors/Skipped: {error_count}")
    
    print(f"\nDatabase Statistics:")
    print(f"  Total images: {stats['total_images']}")
    print(f"  Unique groups: {stats['unique_groups']}")
    print(f"  Duplicate images: {stats['duplicate_images']}")
    print(f"  Queued for deep processing: {stats['queued_for_processing']}")
    print(f"  Completed deep processing: {stats['completed_processing']}")
    
    print(f"\nNext Steps:")
    print(f"  Run deep processing: uv run process_queue.py")
    print(f"  Check queue status: uv run check_queue.py")
    print("="*60 + "\n")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nIngest interrupted by user.")
        print("Saving progress...")
        db = get_database()
        db.save_databases()
        print("Progress saved.")
        sys.exit(0)
    except Exception as e:
        print(f"\n\nFATAL ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

