#!/usr/bin/env python3
"""
Deep Processing Script (Tier 2) - V3.3 Cascade Architecture
Processes queued photos with heavyweight models for detailed analysis.

Uses ~24GB VRAM:
- Qwen3-VL-8B-Thinking (8B) for deep captions
- SigLIP2-Giant (2B) for face embeddings
- RT-DETR-r101vd for object detection
- OpenCV Haar Cascade for face detection
- spaCy for tag extraction

Workflow:
1. Load high-quality models
2. Process each queued group
3. Generate deep caption
4. Extract tags (NOUN, PROPN)
5. Detect objects
6. Detect and recognize faces
7. Update database with results
"""

import os
import sys
import torch
import faiss
import numpy as np
import json
import cv2
from PIL import Image
from typing import Dict, List

from transformers import (
    AutoProcessor,
    AutoModel,
    AutoImageProcessor,
    AutoModelForObjectDetection,
    Qwen3VLForConditionalGeneration
)

from config import (
    EMBEDDING_MODEL_ID,
    EMBEDDING_DIM,
    DETECTION_MODEL_ID,
    CAPTION_MODEL_ID,
    DETECTION_THRESHOLD,
    MAX_CAPTION_TOKENS,
    FACE_SIMILARITY_THRESHOLD
)
from database import get_database

# Try to import spacy
try:
    import spacy
    SPACY_AVAILABLE = True
except ImportError:
    SPACY_AVAILABLE = False
    print("Warning: spacy not installed. Install with: pip install spacy && python -m spacy download en_core_web_sm")


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


class DeepProcessor:
    """Handles deep tier-2 processing with heavyweight models."""
    
    def __init__(self):
        self.device, self.dtype = get_inference_device_and_dtype()
        self.db = get_database()
        
        # Model components
        self.embed_processor = None
        self.embed_model = None
        self.detect_processor = None
        self.detect_model = None
        self.caption_processor = None
        self.caption_model = None
        self.nlp = None
        self.face_cascade = None
        
    def load_models(self):
        """Load heavyweight models for deep processing."""
        print("\n=== Loading Tier-2 (Deep) Models ===")
        
        # Model 1: SigLIP2-Giant for embeddings (faces)
        print(f"\n[1/5] Loading Embedding Model: {EMBEDDING_MODEL_ID}")
        self.embed_processor = AutoProcessor.from_pretrained(EMBEDDING_MODEL_ID, use_fast=True)
        self.embed_model = AutoModel.from_pretrained(
            EMBEDDING_MODEL_ID,
            dtype=self.dtype,
            device_map="auto"
        ).eval()
        print("✓ Embedding model loaded")
        
        # Model 2: RT-DETR for object detection
        print(f"\n[2/5] Loading Object Detection Model: {DETECTION_MODEL_ID}")
        self.detect_processor = AutoImageProcessor.from_pretrained(DETECTION_MODEL_ID, use_fast=True)
        self.detect_model = AutoModelForObjectDetection.from_pretrained(
            DETECTION_MODEL_ID,
            dtype=self.dtype,
            device_map="auto"
        ).eval()
        print("✓ Object detection model loaded")
        
        # Model 3: Qwen3-VL for deep captioning (v3.3)
        print(f"\n[3/5] Loading Deep Caption Model: {CAPTION_MODEL_ID}")
        self.caption_processor = AutoProcessor.from_pretrained(
            CAPTION_MODEL_ID,
            trust_remote_code=True
        )
        self.caption_model = Qwen3VLForConditionalGeneration.from_pretrained(
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
            print("✗ spaCy not available")
            self.nlp = None
        
        # Model 5: OpenCV for face detection
        print(f"\n[5/5] Loading Face Detector (OpenCV Haar Cascade)")
        cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        self.face_cascade = cv2.CascadeClassifier(cascade_path)
        if self.face_cascade.empty():
            print("✗ Failed to load face cascade")
            self.face_cascade = None
        else:
            print("✓ Face detector loaded")
        
        print("\n=== Tier-2 models ready ===\n")
        
    @torch.no_grad()
    def get_embedding(self, image: Image.Image) -> np.ndarray:
        """
        Create a 1152-dimension L2-normalized embedding for any image.
        Works for both full images and face crops.
        """
        inputs = self.embed_processor(
            images=image,
            return_tensors="pt"
        ).to(self.device)
        
        image_features = self.embed_model.get_image_features(**inputs)
        vector = image_features.cpu().float().numpy().astype(np.float32)
        # Ensure 2D shape (1, dim) for FAISS
        if vector.ndim == 1:
            vector = vector.reshape(1, -1)
        faiss.normalize_L2(vector)
        
        return vector
        
    def extract_tags_from_caption(self, caption: str) -> List[str]:
        """
        Extract searchable tags from caption using spaCy POS tagging.
        Extracts NOUN and PROPN tokens.
        """
        tags_list = []
        
        if self.nlp:
            try:
                doc = self.nlp(caption)
                for token in doc:
                    if token.pos_ in ['NOUN', 'PROPN'] and len(token.text) > 1:
                        # Use lemma for root form
                        tags_list.append(token.lemma_.lower())
                tags_list = list(set(tags_list))  # De-duplicate
            except Exception as e:
                print(f"  > Tag extraction failed: {e}")
        else:
            print("  > spaCy not available, skipping tag extraction")
            
        return tags_list
        
    @torch.no_grad()
    def get_or_create_person_group(self, person_embedding: np.ndarray) -> str:
        """
        V3.3: Face grouping and recognition.
        Searches the face index for a match. If found, returns existing person_group_id.
        If not, creates a new person group.
        """
        # Search face index for similar faces
        if self.db.faiss_face_index.ntotal > 0:
            scores, faiss_ids = self.db.faiss_face_index.search(person_embedding, k=1)
            similarity_score = scores[0][0]
            matched_faiss_id = faiss_ids[0][0]
        else:
            similarity_score = 0.0
            
        # Check if this is a known person
        if similarity_score > FACE_SIMILARITY_THRESHOLD:
            self.db.cursor.execute(
                "SELECT person_group_id FROM person_groups WHERE representative_faiss_id = ?",
                (int(matched_faiss_id),)
            )
            result = self.db.cursor.fetchone()
            if result:
                person_group_id = result[0]
                print(f"    > Recognized person: {person_group_id} (similarity: {similarity_score:.4f})")
                return person_group_id
                
        # This is a new person
        new_faiss_id = self.db.faiss_face_index.ntotal
        self.db.faiss_face_index.add(person_embedding)
        
        new_person_group_id = f"person_{new_faiss_id}"
        default_name = f"Person {new_faiss_id + 1}"
        
        self.db.cursor.execute(
            "INSERT INTO person_groups (person_group_id, name, representative_faiss_id) VALUES (?, ?, ?)",
            (new_person_group_id, default_name, int(new_faiss_id))
        )
        self.db.db_conn.commit()
        
        print(f"    > New person detected: {new_person_group_id}")
        return new_person_group_id
        
    @torch.no_grad()
    def run_deep_processing_pipeline(
        self,
        image_path: str,
        pil_image: Image.Image
    ) -> Dict[str, str]:
        """
        Run the full SOTA analysis on a unique image (v3.3).
        Includes: deep caption (Qwen3-VL), tag extraction, object detection, face recognition.
        """
        print(f"  > Running deep processing pipeline...")
        ai_results = {}
        
        # Convert PIL Image to OpenCV format for face detection
        cv_image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
        gray_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
        
        # --- Step 1: Deep Caption with Qwen3-VL ---
        print(f"  > Generating deep caption with Qwen3-VL...")
        
        # Prepare chat template for Qwen3-VL
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": "Describe this image in detail."}
                ]
            }
        ]
        
        # Apply chat template
        text_prompt = self.caption_processor.apply_chat_template(
            messages,
            add_generation_prompt=True
        )
        
        # Process image and text
        inputs = self.caption_processor(
            text=[text_prompt],
            images=[pil_image],
            return_tensors="pt"
        ).to(self.device)
        
        # Generate caption
        generated_ids = self.caption_model.generate(
            **inputs,
            max_new_tokens=MAX_CAPTION_TOKENS
        )
        
        generated_text = self.caption_processor.batch_decode(
            generated_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False
        )[0]
        
        # Extract assistant's response
        if "assistant" in generated_text.lower():
            caption = generated_text.split("assistant")[-1].strip()
        else:
            caption = generated_text.strip()
            
        ai_results["generated_caption"] = caption
        print(f"  > Deep Caption: '{caption}'")
        
        # --- Step 2: Tag Extraction with spaCy ---
        tags = self.extract_tags_from_caption(caption)
        ai_results["extracted_tags_json"] = json.dumps(tags)
        print(f"  > Extracted Tags: {tags}")
        
        # --- Step 3: Object Detection with RT-DETR ---
        print(f"  > Running object detection...")
        detect_inputs = self.detect_processor(
            images=pil_image,
            return_tensors="pt"
        ).to(self.device)
        
        outputs = self.detect_model(**detect_inputs)
        
        # Post-process to get bounding boxes and labels
        target_sizes = torch.tensor([pil_image.size[::-1]], device=self.device)
        detections = self.detect_processor.post_process_object_detection(
            outputs,
            threshold=DETECTION_THRESHOLD,
            target_sizes=target_sizes
        )[0]
        
        detected_objects = []
        
        for score, label_id, box in zip(
            detections["scores"],
            detections["labels"],
            detections["boxes"]
        ):
            label = self.detect_model.config.id2label[label_id.item()]
            detection_data = {
                "label": label,
                "score": score.item(),
                "box": box.tolist()
            }
            detected_objects.append(detection_data)
        
        ai_results["detected_objects_json"] = json.dumps(detected_objects)
        print(f"  > Detected {len(detected_objects)} objects")
        
        # --- Step 4: Face Detection and Recognition (v3.3) ---
        if self.face_cascade is not None:
            print(f"  > Running face detection...")
            faces = self.face_cascade.detectMultiScale(
                gray_image,
                scaleFactor=1.1,
                minNeighbors=5,
                minSize=(30, 30)
            )
            print(f"  > Found {len(faces)} faces with OpenCV")
            
            for (x, y, w, h) in faces:
                try:
                    # Crop the exact face from the PIL image
                    face_crop = pil_image.crop((x, y, x+w, y+h))
                    
                    # Embed using SigLIP2-Giant
                    face_embedding = self.get_embedding(face_crop)
                    
                    # Get or create the unique ID for this person
                    person_group_id = self.get_or_create_person_group(face_embedding)
                    
                    # Store this appearance in face_appearances table
                    self.db.cursor.execute(
                        "INSERT INTO face_appearances (image_path, person_group_id, box_json) VALUES (?, ?, ?)",
                        (image_path, person_group_id, json.dumps([x, y, w, h]))
                    )
                    print(f"    > Linked face at [{x},{y}] to {person_group_id}")
                    
                except Exception as e:
                    print(f"  > Face embedding failed: {e}")
            
            self.db.db_conn.commit()  # Commit face appearances
        else:
            print(f"  > Face detection skipped (OpenCV not available)")
        
        return ai_results
        
    def process_queue(self):
        """
        Main processing loop for the queue.
        Processes all groups in the processing_queue table.
        """
        print("\n=== Starting Deep Processing Queue ===")
        
        # Get all queued groups
        self.db.cursor.execute("SELECT group_id FROM processing_queue")
        queue = self.db.cursor.fetchall()
        
        print(f"Found {len(queue)} groups to process")
        
        if len(queue) == 0:
            print("Queue is empty!")
            return
        
        success_count = 0
        error_count = 0
        
        for (group_id,) in queue:
            print(f"\n--- Processing Group: {group_id} ---")
            
            # Get canonical path for this group
            self.db.cursor.execute(
                "SELECT canonical_path FROM image_groups WHERE group_id = ?",
                (group_id,)
            )
            path_result = self.db.cursor.fetchone()
            
            if not path_result:
                print(f"  > ERROR: No canonical path found for {group_id}. Skipping.")
                error_count += 1
                continue
                
            try:
                image_path = path_result[0]
                pil_image = Image.open(image_path).convert("RGB")
                
                # Run deep processing
                ai_results = self.run_deep_processing_pipeline(image_path, pil_image)
                
                # Update image_groups with results
                self.db.cursor.execute(
                    """
                    UPDATE image_groups 
                    SET generated_caption = ?, 
                        detected_objects_json = ?, 
                        extracted_tags_json = ?,
                        processing_status = 'COMPLETED'
                    WHERE group_id = ?
                    """,
                    (
                        ai_results["generated_caption"],
                        ai_results["detected_objects_json"],
                        ai_results["extracted_tags_json"],
                        group_id
                    )
                )
                
                # Remove from processing queue
                self.db.cursor.execute(
                    "DELETE FROM processing_queue WHERE group_id = ?",
                    (group_id,)
                )
                
                self.db.db_conn.commit()
                success_count += 1
                print(f"  > SUCCESS: Group {group_id} completed")
                
            except Exception as e:
                print(f"  > FAILED: {e}")
                import traceback
                traceback.print_exc()
                
                # Mark as failed
                self.db.cursor.execute(
                    "UPDATE image_groups SET processing_status = 'FAILED' WHERE group_id = ?",
                    (group_id,)
                )
                self.db.db_conn.commit()
                error_count += 1
        
        # Save databases
        print("\n=== Saving Databases ===")
        self.db.save_databases()
        
        print("\n" + "="*60)
        print("Deep Processing Complete!")
        print("="*60)
        print(f"\nProcessing Summary:")
        print(f"  Successfully processed: {success_count}")
        print(f"  Errors/Failed: {error_count}")
        print("="*60 + "\n")


def main():
    """Main deep processing workflow."""
    print("\n" + "="*60)
    print("AI Photo Management - Deep Processing (Tier 2)")
    print("="*60)
    
    # Initialize processor
    print("\nInitializing deep processor...")
    processor = DeepProcessor()
    processor.load_models()
    
    # Process the queue
    processor.process_queue()
    
    print("\nDeep processing workflow complete.")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nProcessing interrupted by user.")
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

