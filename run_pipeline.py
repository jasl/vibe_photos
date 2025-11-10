#!/usr/bin/env python3
"""
Unified AI Photo Management Pipeline (v3.3)

Combines three processing stages:
1. Fast Ingest (Tier 1): Visual + semantic deduplication, EXIF extraction
2. Deep Processing (Tier 2): Qwen3-VL captions, object detection, face recognition
3. Semantic Merge: Merge groups with similar captions

Usage:
    python run_pipeline.py                    # Run all stages
    python run_pipeline.py --ingest-only      # Only run fast ingest
    python run_pipeline.py --process-only     # Only run deep processing
    python run_pipeline.py --merge-only       # Only run semantic merge
    python run_pipeline.py --no-merge         # Run ingest + process (skip merge)
"""

import os
import sys
import argparse
import torch
import faiss
import numpy as np
import json
import cv2
from datetime import datetime
from pathlib import Path
from tqdm import tqdm
from PIL import Image

from transformers import (
    AutoProcessor,
    AutoModel,
    AutoImageProcessor,
    AutoModelForObjectDetection,
    Qwen3VLForConditionalGeneration,
    Blip2ForConditionalGeneration,
    Blip2Processor
)

from config import (
    PHOTOS_DIR,
    SUPPORTED_EXTENSIONS,
    VISUAL_SIMILARITY_THRESHOLD,
    SEMANTIC_SIMILARITY_THRESHOLD,
    EMBEDDING_MODEL_ID,
    EMBEDDING_DIM,
    DETECTION_MODEL_ID,
    CAPTION_MODEL_ID,
    DETECTION_THRESHOLD,
    MAX_CAPTION_TOKENS,
    FACE_SIMILARITY_THRESHOLD
)
from database import get_database

# Fast caption model for tier 1
FAST_CAPTION_MODEL_ID = "Salesforce/blip2-opt-2.7b"

# Try to import spacy
try:
    import spacy
    SPACY_AVAILABLE = True
except ImportError:
    SPACY_AVAILABLE = False


def get_inference_device_and_dtype():
    """Auto-detect device and dtype (v3.3: bfloat16 on CUDA for quality)."""
    if torch.cuda.is_available():
        print("Using CUDA (NVIDIA GPU) - bfloat16")
        return torch.device("cuda"), torch.bfloat16
    if torch.backends.mps.is_available():
        print("Using MPS (Apple Silicon GPU) - float16")
        return torch.device("mps"), torch.float16
    print("Using CPU - float32")
    return torch.device("cpu"), torch.float32


class UnifiedPipeline:
    """
    Unified pipeline that handles all processing stages.
    Efficiently loads and reuses models across stages.
    """
    
    def __init__(self):
        self.device, self.dtype = get_inference_device_and_dtype()
        self.db = get_database()
        
        # Tier 1 models (fast ingest)
        self.embed_processor = None
        self.embed_model = None
        self.fast_caption_processor = None
        self.fast_caption_model = None
        
        # Tier 2 models (deep processing)
        self.detect_processor = None
        self.detect_model = None
        self.deep_caption_processor = None
        self.deep_caption_model = None
        self.nlp = None
        self.face_cascade = None
        
    def load_tier1_models(self):
        """Load lightweight models for fast ingest."""
        print("\n=== Loading Tier-1 Models (Fast Ingest) ===")
        
        # Model 1: SigLIP2-Giant for embeddings
        print(f"\n[1/2] Loading Embedding Model: {EMBEDDING_MODEL_ID}")
        self.embed_processor = AutoProcessor.from_pretrained(EMBEDDING_MODEL_ID, use_fast=True)
        self.embed_model = AutoModel.from_pretrained(
            EMBEDDING_MODEL_ID,
            dtype=self.dtype,
            device_map="auto"
        ).eval()
        print("✓ Embedding model loaded")
        
        # Model 2: BLIP2 for fast captions
        print(f"\n[2/2] Loading Fast Caption Model: {FAST_CAPTION_MODEL_ID}")
        self.fast_caption_processor = Blip2Processor.from_pretrained(FAST_CAPTION_MODEL_ID, use_fast=True)
        self.fast_caption_model = Blip2ForConditionalGeneration.from_pretrained(
            FAST_CAPTION_MODEL_ID,
            dtype=self.dtype,
            device_map="auto"
        ).eval()
        print("✓ Fast caption model loaded")
        
        print("\n=== Tier-1 models ready ===\n")
        
    def load_tier2_models(self):
        """Load heavyweight models for deep processing."""
        print("\n=== Loading Tier-2 Models (Deep Processing) ===")
        
        # Reuse embedding model if already loaded
        if self.embed_model is None:
            print(f"\n[1/5] Loading Embedding Model: {EMBEDDING_MODEL_ID}")
            self.embed_processor = AutoProcessor.from_pretrained(EMBEDDING_MODEL_ID, use_fast=True)
            self.embed_model = AutoModel.from_pretrained(
                EMBEDDING_MODEL_ID,
                dtype=self.dtype,
                device_map="auto"
            ).eval()
            print("✓ Embedding model loaded")
        else:
            print(f"\n[1/5] Reusing already loaded embedding model")
        
        # Model 2: RT-DETR for object detection
        print(f"\n[2/5] Loading Object Detection Model: {DETECTION_MODEL_ID}")
        self.detect_processor = AutoImageProcessor.from_pretrained(DETECTION_MODEL_ID, use_fast=True)
        self.detect_model = AutoModelForObjectDetection.from_pretrained(
            DETECTION_MODEL_ID,
            dtype=self.dtype,
            device_map="auto"
        ).eval()
        print("✓ Object detection model loaded")
        
        # Model 3: Qwen3-VL for deep captioning
        print(f"\n[3/5] Loading Deep Caption Model: {CAPTION_MODEL_ID}")
        self.deep_caption_processor = AutoProcessor.from_pretrained(
            CAPTION_MODEL_ID,
            trust_remote_code=True
        )
        self.deep_caption_model = Qwen3VLForConditionalGeneration.from_pretrained(
            CAPTION_MODEL_ID,
            dtype=self.dtype,
            device_map="auto",
            trust_remote_code=True
        ).eval()
        print("✓ Deep caption model loaded (Qwen3-VL-8B-Thinking)")
        
        # Model 4: spaCy for tag extraction
        print(f"\n[4/5] Loading Tag Extraction Model (spaCy)")
        if SPACY_AVAILABLE:
            try:
                self.nlp = spacy.load("en_core_web_sm")
                print("✓ spaCy model loaded")
            except:
                print("✗ spaCy model not found. Run: python -m spacy download en_core_web_sm")
                self.nlp = None
        else:
            print("✗ spaCy not available. Install with: pip install spacy")
            self.nlp = None
        
        # Model 5: OpenCV for face detection
        print(f"\n[5/5] Loading Face Detector (OpenCV)")
        cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        self.face_cascade = cv2.CascadeClassifier(cascade_path)
        if self.face_cascade.empty():
            print("✗ Failed to load face cascade")
            self.face_cascade = None
        else:
            print("✓ Face detector loaded")
        
        print("\n=== Tier-2 models ready ===\n")
        
    # ========== TIER 1: FAST INGEST ==========
    
    def get_exif_data(self, image: Image.Image, image_path: str) -> dict:
        """Extract EXIF metadata: timestamp and GPS coordinates."""
        timestamp = None
        latitude = None
        longitude = None
        
        try:
            ctime = os.path.getctime(image_path)
            timestamp = datetime.fromtimestamp(ctime).isoformat()
            
            exif_data = image._getexif()
            if exif_data:
                dt_original = exif_data.get(36867)  # DateTimeOriginal
                if dt_original:
                    try:
                        dt_obj = datetime.strptime(dt_original, '%Y:%m:%d %H:%M:%S')
                        timestamp = dt_obj.isoformat()
                    except:
                        pass
                
                gps_info = exif_data.get(34853)  # GPSInfo
                if gps_info:
                    try:
                        lat_dms = gps_info.get(2)
                        lat_ref = gps_info.get(1)
                        lon_dms = gps_info.get(4)
                        lon_ref = gps_info.get(3)
                        
                        if lat_dms and lat_ref and lon_dms and lon_ref:
                            latitude = float(lat_dms[0]) + float(lat_dms[1])/60 + float(lat_dms[2])/3600
                            if lat_ref in ['S', 's']:
                                latitude *= -1
                            longitude = float(lon_dms[0]) + float(lon_dms[1])/60 + float(lon_dms[2])/3600
                            if lon_ref in ['W', 'w']:
                                longitude *= -1
                    except:
                        pass
        except:
            pass
            
        return {"timestamp": timestamp, "latitude": latitude, "longitude": longitude}
    
    @torch.no_grad()
    def get_image_embedding(self, image: Image.Image) -> np.ndarray:
        """Create L2-normalized image embedding."""
        inputs = self.embed_processor(images=image, return_tensors="pt").to(self.device)
        image_features = self.embed_model.get_image_features(**inputs)
        vector = image_features.cpu().float().numpy().astype(np.float32)
        # Ensure 2D shape (1, EMBEDDING_DIM) for FAISS
        if vector.ndim == 1:
            vector = vector.reshape(1, -1)
        elif vector.shape[0] != 1:
            vector = vector.reshape(1, -1)
        # Verify dimension matches
        assert vector.shape[1] == EMBEDDING_DIM, f"Embedding dimension mismatch: got {vector.shape[1]}, expected {EMBEDDING_DIM}"
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
        ).to(self.device)
        text_features = self.embed_model.get_text_features(**inputs)
        vector = text_features.cpu().float().numpy().astype(np.float32)
        # Ensure 2D shape (1, dim) for FAISS
        if vector.ndim == 1:
            vector = vector.reshape(1, -1)
        faiss.normalize_L2(vector)
        return vector
    
    @torch.no_grad()
    def get_fast_caption(self, image: Image.Image) -> str:
        """Generate quick caption with BLIP2."""
        inputs = self.fast_caption_processor(image, return_tensors="pt").to(self.device, self.dtype)
        generated_ids = self.fast_caption_model.generate(**inputs, max_new_tokens=50)
        caption = self.fast_caption_processor.batch_decode(
            generated_ids,
            skip_special_tokens=True
        )[0].strip()
        return caption
    
    def ingest_image(self, image_path: str) -> bool:
        """Fast ingest workflow with dual deduplication."""
        # Check if already processed
        self.db.cursor.execute("SELECT group_id FROM images WHERE image_path = ?", (image_path,))
        if self.db.cursor.fetchone():
            return True
            
        try:
            image = Image.open(image_path).convert("RGB")
        except Exception as e:
            return False
            
        exif_data = self.get_exif_data(image, image_path)
        image_embedding = self.get_image_embedding(image)
        
        # Visual deduplication
        if self.db.faiss_image_index.ntotal > 0:
            scores, faiss_ids = self.db.faiss_image_index.search(image_embedding, k=1)
            visual_score = scores[0][0]
            matched_image_faiss_id = faiss_ids[0][0]
        else:
            visual_score = 0.0
            
        if visual_score > VISUAL_SIMILARITY_THRESHOLD:
            self.db.cursor.execute(
                "SELECT i.group_id, g.canonical_path FROM images i JOIN image_groups g ON i.group_id = g.group_id WHERE i.faiss_image_id = ?",
                (int(matched_image_faiss_id),)
            )
            result = self.db.cursor.fetchone()
            if result:
                group_id, duplicate_of_path = result[0], result[1]
                self.db.cursor.execute(
                    "INSERT OR REPLACE INTO images (image_path, group_id, faiss_image_id, duplicate_of_path, duplicate_score) VALUES (?, ?, ?, ?, ?)",
                    (image_path, group_id, None, duplicate_of_path, visual_score)
                )
                self.db.db_conn.commit()
                return True
        
        # Semantic deduplication
        fast_caption = self.get_fast_caption(image)
        caption_embedding = self.get_text_embedding(fast_caption)
        
        if self.db.faiss_caption_index.ntotal > 0:
            scores, faiss_ids = self.db.faiss_caption_index.search(caption_embedding, k=1)
            semantic_score = scores[0][0]
            matched_caption_faiss_id = faiss_ids[0][0]
        else:
            semantic_score = 0.0
            
        if semantic_score > SEMANTIC_SIMILARITY_THRESHOLD:
            self.db.cursor.execute(
                "SELECT i.group_id, g.canonical_path FROM images i JOIN image_groups g ON i.group_id = g.group_id WHERE i.faiss_image_id = ?",
                (int(matched_caption_faiss_id),)
            )
            result = self.db.cursor.fetchone()
            if result:
                group_id, duplicate_of_path = result[0], result[1]
                self.db.cursor.execute(
                    "INSERT OR REPLACE INTO images (image_path, group_id, faiss_image_id, duplicate_of_path, duplicate_score) VALUES (?, ?, ?, ?, ?)",
                    (image_path, group_id, None, duplicate_of_path, semantic_score)
                )
                self.db.db_conn.commit()
                return True
        
        # New unique image - create group and queue
        new_group_id = f"group_{self.db.faiss_image_index.ntotal}"
        new_faiss_image_id = self.db.faiss_image_index.ntotal
        
        self.db.faiss_image_index.add(image_embedding)
        self.db.faiss_caption_index.add(caption_embedding)
        
        self.db.cursor.execute(
            "INSERT OR REPLACE INTO images (image_path, group_id, faiss_image_id) VALUES (?, ?, ?)",
            (image_path, new_group_id, int(new_faiss_image_id))
        )
        
        self.db.cursor.execute(
            """INSERT OR REPLACE INTO image_groups 
               (group_id, canonical_path, fast_caption, processing_status, taken_at_timestamp, gps_latitude, gps_longitude)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (new_group_id, image_path, fast_caption, 'QUEUED', 
             exif_data["timestamp"], exif_data["latitude"], exif_data["longitude"])
        )
        
        self.db.cursor.execute("INSERT OR REPLACE INTO processing_queue (group_id) VALUES (?)", (new_group_id,))
        self.db.db_conn.commit()
        return True
    
    def run_ingest(self):
        """Run the fast ingest stage."""
        print("\n" + "="*70)
        print("STAGE 1: Fast Ingest (Visual + Semantic Deduplication)")
        print("="*70)
        
        if not os.path.exists(PHOTOS_DIR):
            print(f"\nERROR: Photos directory not found: {PHOTOS_DIR}")
            return False
        
        # Find all images
        print(f"\nScanning for images in: {PHOTOS_DIR}")
        image_paths = []
        for root, dirs, files in os.walk(PHOTOS_DIR):
            for file in files:
                if Path(file).suffix in SUPPORTED_EXTENSIONS:
                    image_paths.append(os.path.join(root, file))
        
        print(f"Found {len(image_paths)} images")
        
        if not image_paths:
            print("No images found!")
            return False
        
        # Check which are already processed
        self.db.cursor.execute("SELECT image_path FROM images")
        processed_paths = {row[0] for row in self.db.cursor.fetchall()}
        images_to_process = [p for p in image_paths if p not in processed_paths]
        
        print(f"Already processed: {len(processed_paths)}")
        print(f"To process: {len(images_to_process)}")
        
        if not images_to_process:
            print("\nAll images already processed!")
            return True
        
        # Load tier 1 models
        self.load_tier1_models()
        
        # Process images
        print("\nProcessing images...")
        success_count = 0
        
        with tqdm(total=len(images_to_process), unit="img") as pbar:
            for image_path in images_to_process:
                pbar.set_description(f"Ingesting {Path(image_path).name[:30]}")
                if self.ingest_image(image_path):
                    success_count += 1
                pbar.update(1)
                
                if success_count % 100 == 0:
                    self.db.save_databases()
        
        self.db.save_databases()
        
        stats = self.db.get_stats()
        print(f"\n✓ Ingest complete: {success_count} images processed")
        print(f"  Total unique groups: {stats['unique_groups']}")
        print(f"  Queued for deep processing: {stats['queued_for_processing']}")
        return True
    
    # ========== TIER 2: DEEP PROCESSING ==========
    
    def extract_tags_from_caption(self, caption: str) -> list:
        """Extract tags using spaCy POS tagging."""
        tags_list = []
        if self.nlp:
            try:
                doc = self.nlp(caption)
                for token in doc:
                    if token.pos_ in ['NOUN', 'PROPN'] and len(token.text) > 1:
                        tags_list.append(token.lemma_.lower())
                tags_list = list(set(tags_list))
            except Exception as e:
                pass
        return tags_list
    
    @torch.no_grad()
    def get_or_create_person_group(self, person_embedding: np.ndarray) -> str:
        """Face grouping and recognition."""
        if self.db.faiss_face_index.ntotal > 0:
            scores, faiss_ids = self.db.faiss_face_index.search(person_embedding, k=1)
            similarity_score = scores[0][0]
            matched_faiss_id = faiss_ids[0][0]
        else:
            similarity_score = 0.0
            
        if similarity_score > FACE_SIMILARITY_THRESHOLD:
            self.db.cursor.execute(
                "SELECT person_group_id FROM person_groups WHERE representative_faiss_id = ?",
                (int(matched_faiss_id),)
            )
            result = self.db.cursor.fetchone()
            if result:
                return result[0]
        
        # New person
        new_faiss_id = self.db.faiss_face_index.ntotal
        self.db.faiss_face_index.add(person_embedding)
        new_person_group_id = f"person_{new_faiss_id}"
        default_name = f"Person {new_faiss_id + 1}"
        
        self.db.cursor.execute(
            "INSERT INTO person_groups (person_group_id, name, representative_faiss_id) VALUES (?, ?, ?)",
            (new_person_group_id, default_name, int(new_faiss_id))
        )
        self.db.db_conn.commit()
        return new_person_group_id
    
    @torch.no_grad()
    def process_group(self, image_path: str, pil_image: Image.Image) -> dict:
        """Run deep processing on a unique image."""
        ai_results = {}
        
        # Convert to OpenCV format for face detection
        cv_image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
        gray_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
        
        # Deep Caption with Qwen3-VL
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": "Describe this image in detail."}
                ]
            }
        ]
        
        text_prompt = self.deep_caption_processor.apply_chat_template(messages, add_generation_prompt=True)
        inputs = self.deep_caption_processor(
            text=[text_prompt],
            images=[pil_image],
            return_tensors="pt"
        ).to(self.device)
        
        generated_ids = self.deep_caption_model.generate(**inputs, max_new_tokens=MAX_CAPTION_TOKENS)
        generated_text = self.deep_caption_processor.batch_decode(
            generated_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False
        )[0]
        
        caption = generated_text.split("assistant")[-1].strip() if "assistant" in generated_text.lower() else generated_text.strip()
        ai_results["generated_caption"] = caption
        
        # Tag Extraction
        tags = self.extract_tags_from_caption(caption)
        ai_results["extracted_tags_json"] = json.dumps(tags)
        
        # Object Detection
        detect_inputs = self.detect_processor(images=pil_image, return_tensors="pt").to(self.device)
        outputs = self.detect_model(**detect_inputs)
        target_sizes = torch.tensor([pil_image.size[::-1]], device=self.device)
        detections = self.detect_processor.post_process_object_detection(
            outputs,
            threshold=DETECTION_THRESHOLD,
            target_sizes=target_sizes
        )[0]
        
        detected_objects = []
        for score, label_id, box in zip(detections["scores"], detections["labels"], detections["boxes"]):
            detected_objects.append({
                "label": self.detect_model.config.id2label[label_id.item()],
                "score": score.item(),
                "box": box.tolist()
            })
        
        ai_results["detected_objects_json"] = json.dumps(detected_objects)
        
        # Face Detection and Recognition
        if self.face_cascade is not None:
            faces = self.face_cascade.detectMultiScale(gray_image, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
            for (x, y, w, h) in faces:
                try:
                    face_crop = pil_image.crop((x, y, x+w, y+h))
                    face_embedding = self.get_image_embedding(face_crop)
                    person_group_id = self.get_or_create_person_group(face_embedding)
                    self.db.cursor.execute(
                        "INSERT INTO face_appearances (image_path, person_group_id, box_json) VALUES (?, ?, ?)",
                        (image_path, person_group_id, json.dumps([x, y, w, h]))
                    )
                except:
                    pass
            self.db.db_conn.commit()
        
        return ai_results
    
    def run_deep_processing(self):
        """Run the deep processing stage."""
        print("\n" + "="*70)
        print("STAGE 2: Deep Processing (Captions, Objects, Faces)")
        print("="*70)
        
        # Get queue
        self.db.cursor.execute("SELECT group_id FROM processing_queue")
        queue = self.db.cursor.fetchall()
        
        print(f"\nFound {len(queue)} groups to process")
        
        if len(queue) == 0:
            print("Queue is empty!")
            return True
        
        # Load tier 2 models
        self.load_tier2_models()
        
        # Process queue
        success_count = 0
        error_count = 0
        
        with tqdm(total=len(queue), unit="group") as pbar:
            for (group_id,) in queue:
                pbar.set_description(f"Processing {group_id}")
                
                self.db.cursor.execute(
                    "SELECT canonical_path FROM image_groups WHERE group_id = ?",
                    (group_id,)
                )
                path_result = self.db.cursor.fetchone()
                
                if not path_result:
                    error_count += 1
                    pbar.update(1)
                    continue
                
                try:
                    image_path = path_result[0]
                    pil_image = Image.open(image_path).convert("RGB")
                    ai_results = self.process_group(image_path, pil_image)
                    
                    self.db.cursor.execute(
                        """UPDATE image_groups 
                           SET generated_caption = ?, detected_objects_json = ?, 
                               extracted_tags_json = ?, processing_status = 'COMPLETED'
                           WHERE group_id = ?""",
                        (ai_results["generated_caption"], ai_results["detected_objects_json"],
                         ai_results["extracted_tags_json"], group_id)
                    )
                    
                    self.db.cursor.execute("DELETE FROM processing_queue WHERE group_id = ?", (group_id,))
                    self.db.db_conn.commit()
                    success_count += 1
                    
                except Exception as e:
                    self.db.cursor.execute(
                        "UPDATE image_groups SET processing_status = 'FAILED' WHERE group_id = ?",
                        (group_id,)
                    )
                    self.db.db_conn.commit()
                    error_count += 1
                
                pbar.update(1)
        
        self.db.save_databases()
        
        print(f"\n✓ Deep processing complete: {success_count} groups processed")
        if error_count > 0:
            print(f"  Errors: {error_count}")
        return True
    
    # ========== TIER 3: SEMANTIC MERGE ==========
    
    @torch.no_grad()
    def get_text_embeddings_batch(self, text_list: list) -> np.ndarray:
        """Generate embeddings for a batch of text."""
        inputs = self.embed_processor(
            text=text_list,
            return_tensors="pt",
            padding=True,
            truncation=True
        ).to(self.device)
        
        text_features = self.embed_model.get_text_features(**inputs)
        vector = text_features.cpu().float().numpy().astype(np.float32)
        # Ensure 2D shape (batch_size, dim) for FAISS
        if vector.ndim == 1:
            vector = vector.reshape(1, -1)
        faiss.normalize_L2(vector)
        return vector
    
    def run_semantic_merge(self):
        """Run the semantic merge stage."""
        print("\n" + "="*70)
        print("STAGE 3: Semantic Merge (Consolidate Similar Groups)")
        print("="*70)
        
        # Get all groups with captions
        self.db.cursor.execute(
            "SELECT group_id, generated_caption FROM image_groups WHERE processing_status = 'COMPLETED' AND generated_caption IS NOT NULL"
        )
        rows = self.db.cursor.fetchall()
        
        if len(rows) < 2:
            print("\nNot enough groups to merge.")
            return True
        
        group_ids = [row[0] for row in rows]
        captions = [row[1] for row in rows if row[1] and row[1].strip()]
        
        if len(captions) < 2:
            print("\nNot enough valid captions to merge.")
            return True
        
        print(f"\nFound {len(captions)} groups with captions")
        
        # Load embedding model if not already loaded
        if self.embed_model is None:
            print("\nLoading embedding model...")
            self.embed_processor = AutoProcessor.from_pretrained(EMBEDDING_MODEL_ID)
            self.embed_model = AutoModel.from_pretrained(
                EMBEDDING_MODEL_ID,
                dtype=self.dtype,
                device_map="auto"
            ).eval()
        
        # Create embeddings
        print("Creating caption embeddings...")
        caption_embeddings = self.get_text_embeddings_batch(captions)
        
        # Build FAISS index
        caption_index = faiss.IndexFlatIP(EMBEDDING_DIM)
        caption_index.add(caption_embeddings)
        
        # Search for duplicates
        print("Searching for semantic duplicates...")
        k = 5
        all_scores, all_indices = caption_index.search(caption_embeddings, k=k)
        
        merged_groups = set()
        total_merges = 0
        
        for i in range(len(all_indices)):
            primary_group_id = group_ids[i]
            if primary_group_id in merged_groups:
                continue
            
            for j in range(1, k):
                neighbor_index = all_indices[i][j]
                neighbor_score = all_scores[i][j]
                
                if neighbor_index < 0 or neighbor_index >= len(group_ids):
                    continue
                
                secondary_group_id = group_ids[neighbor_index]
                
                if secondary_group_id in merged_groups or primary_group_id == secondary_group_id:
                    continue
                
                if neighbor_score > SEMANTIC_SIMILARITY_THRESHOLD:
                    try:
                        self.db.cursor.execute(
                            "UPDATE images SET group_id = ? WHERE group_id = ?",
                            (primary_group_id, secondary_group_id)
                        )
                        self.db.cursor.execute(
                            "DELETE FROM image_groups WHERE group_id = ?",
                            (secondary_group_id,)
                        )
                        self.db.db_conn.commit()
                        merged_groups.add(secondary_group_id)
                        total_merges += 1
                    except:
                        self.db.db_conn.rollback()
        
        print(f"\n✓ Semantic merge complete: {total_merges} groups merged")
        return True
    
    def run_full_pipeline(self, skip_merge=False):
        """Run the complete pipeline."""
        print("\n" + "="*70)
        print("AI Photo Management - Full Pipeline (v3.3)")
        print("="*70)
        
        # Stage 1: Fast Ingest
        if not self.run_ingest():
            print("\n✗ Pipeline failed at ingest stage")
            return False
        
        # Stage 2: Deep Processing
        if not self.run_deep_processing():
            print("\n✗ Pipeline failed at deep processing stage")
            return False
        
        # Stage 3: Semantic Merge (optional)
        if not skip_merge:
            if not self.run_semantic_merge():
                print("\n✗ Pipeline failed at semantic merge stage")
                return False
        
        # Final stats
        print("\n" + "="*70)
        print("Pipeline Complete!")
        print("="*70)
        
        stats = self.db.get_stats()
        print(f"\nFinal Statistics:")
        print(f"  Total images: {stats['total_images']}")
        print(f"  Unique groups: {stats['unique_groups']}")
        print(f"  Duplicate images: {stats['duplicate_images']}")
        print(f"  Unique persons: {stats.get('unique_persons', 0)}")
        print(f"  Completed processing: {stats['completed_processing']}")
        print(f"  Queued for processing: {stats['queued_for_processing']}")
        
        print(f"\nNext step: Start web interface with 'uv run web_app.py'")
        print("="*70 + "\n")
        
        return True


def main():
    """Main entry point with argument parsing."""
    parser = argparse.ArgumentParser(
        description='AI Photo Management Pipeline (v3.3)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_pipeline.py                 # Run full pipeline
  python run_pipeline.py --ingest-only   # Only run fast ingest
  python run_pipeline.py --process-only  # Only run deep processing
  python run_pipeline.py --merge-only    # Only run semantic merge
  python run_pipeline.py --no-merge      # Run ingest + process (skip merge)
        """
    )
    
    parser.add_argument('--ingest-only', action='store_true', help='Only run fast ingest stage')
    parser.add_argument('--process-only', action='store_true', help='Only run deep processing stage')
    parser.add_argument('--merge-only', action='store_true', help='Only run semantic merge stage')
    parser.add_argument('--no-merge', action='store_true', help='Skip semantic merge stage')
    
    args = parser.parse_args()
    
    # Create pipeline
    pipeline = UnifiedPipeline()
    
    try:
        # Run requested stages
        if args.ingest_only:
            pipeline.run_ingest()
        elif args.process_only:
            pipeline.run_deep_processing()
        elif args.merge_only:
            pipeline.run_semantic_merge()
        else:
            # Run full pipeline
            pipeline.run_full_pipeline(skip_merge=args.no_merge)
            
    except KeyboardInterrupt:
        print("\n\nPipeline interrupted by user.")
        print("Saving progress...")
        pipeline.db.save_databases()
        print("Progress saved.")
        sys.exit(0)
    except Exception as e:
        print(f"\n\nFATAL ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

