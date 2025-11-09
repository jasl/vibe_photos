"""
Image processing pipeline with embedding extraction, object detection, and captioning (v3.2).
Implements:
- Solution 1: Face grouping and recognition (with OpenCV + face_appearances)
- Solution 3: Tag extraction from captions (with Spacy)
- Solution 4: Upgraded Qwen2-VL captioning
- Caching logic to avoid redundant AI processing on duplicate images
"""

import torch
import faiss
import numpy as np
import json
import cv2
from PIL import Image
from typing import Dict, Optional, List

try:
    import spacy
    SPACY_AVAILABLE = True
except ImportError:
    SPACY_AVAILABLE = False
    print("Warning: Spacy not installed. Install with: pip install spacy && python -m spacy download en_core_web_sm")

from config import (
    VISUAL_SIMILARITY_THRESHOLD,
    FACE_SIMILARITY_THRESHOLD,
    DETECTION_THRESHOLD,
    MAX_CAPTION_TOKENS
)
from models import get_models
from database import get_database


class ImageProcessor:
    """
    Handles the v3.2 image processing workflow:
    1. Create embedding (fingerprint)
    2. Check for duplicates
    3. Run expensive AI pipeline only for unique images
    4. Face grouping for persons (with OpenCV)
    5. Tag extraction from captions (with Spacy)
    """
    
    def __init__(self):
        self.models = get_models()
        self.db = get_database()
        
        # V3.2: Load OpenCV Haar Cascade for face detection
        cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        self.face_cascade = cv2.CascadeClassifier(cascade_path)
        
        # V3.2: Load Spacy model for better tag extraction
        if SPACY_AVAILABLE:
            try:
                self.nlp = spacy.load("en_core_web_sm")
            except:
                print("Spacy model not found. Run: python -m spacy download en_core_web_sm")
                self.nlp = None
        else:
            self.nlp = None
        
    @torch.no_grad()
    def get_embedding(self, image: Image.Image, model, processor) -> np.ndarray:
        """
        Create a 1152-dimension L2-normalized embedding for any image (full or crop).
        Flexible function that works for both full images and face crops.
        
        Args:
            image: PIL Image in RGB format
            model: The embedding model to use
            processor: The processor for the model
            
        Returns:
            L2-normalized numpy array of shape (1, 1152)
        """
        inputs = processor(
            images=image,
            return_tensors="pt"
        ).to(self.models.device)
        
        # Get image features (embedding)
        image_features = model.get_image_features(**inputs)
        
        # Convert to float32 for FAISS and normalize
        vector = image_features.cpu().numpy().astype(np.float32)
        faiss.normalize_L2(vector)
        
        return vector
        
    def extract_tags_from_caption(self, caption: str) -> List[str]:
        """
        Solution 3: Extract searchable tags from caption using POS tagging.
        V3.2: Prefers Spacy for better accuracy, falls back to BERT.
        
        Args:
            caption: Generated caption text
            
        Returns:
            List of extracted tags (nouns)
        """
        tags_list = []
        
        # V3.2: Try Spacy first (better quality)
        if self.nlp:
            try:
                doc = self.nlp(caption)
                for token in doc:
                    if token.pos_ in ['NOUN', 'PROPN'] and len(token.text) > 1:
                        tags_list.append(token.text.lower())
                tags_list = list(set(tags_list))
                return tags_list
            except Exception as e:
                print(f"  > Spacy extraction failed: {e}, falling back to BERT")
        
        # Fallback to BERT POS tagger
        try:
            pos_results = self.models.pos_tagger(caption)
            
            # The model outputs Universal Dependencies tags: 'NOUN', 'PROPN'
            for entity in pos_results:
                if entity['entity'] in ['NOUN', 'PROPN']:
                    tag = entity['word'].replace("##", "").strip()
                    if tag and len(tag) > 1:
                        tags_list.append(tag.lower())
                        
            tags_list = list(set(tags_list))
            
        except Exception as e:
            print(f"  > POS Tagger failed: {e}")
            
        return tags_list
        
    @torch.no_grad()
    def get_or_create_person_group(self, person_embedding: np.ndarray) -> str:
        """
        Solution 1: Face grouping and recognition.
        Searches the face index for a match. If found, returns existing person_group_id.
        If not, creates a new person group.
        
        Args:
            person_embedding: Face embedding vector
            
        Returns:
            person_group_id (string)
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
                "SELECT person_group_id FROM person_groups WHERE faiss_face_id = ?",
                (int(matched_faiss_id),)
            )
            result = self.db.cursor.fetchone()
            if result:
                person_group_id = result[0]
                print(f"  > Recognized person: {person_group_id} (similarity: {similarity_score:.4f})")
                return person_group_id
                
        # This is a new person
        new_faiss_id = self.db.faiss_face_index.ntotal
        self.db.faiss_face_index.add(person_embedding)
        
        new_person_group_id = f"person_{new_faiss_id}"
        default_name = f"Person {new_faiss_id + 1}"
        
        self.db.cursor.execute(
            "INSERT INTO person_groups (person_group_id, name, faiss_face_id) VALUES (?, ?, ?)",
            (new_person_group_id, default_name, int(new_faiss_id))
        )
        self.db.db_conn.commit()
        
        print(f"  > New person detected: {new_person_group_id}")
        return new_person_group_id
        
    @torch.no_grad()
    def run_expensive_ai_pipeline(
        self,
        image: Image.Image,
        new_group_id: str
    ) -> Dict[str, str]:
        """
        Run the full SOTA analysis on a unique image (v2).
        Includes: caption generation (Qwen2-VL), object detection, face grouping, and tag extraction.
        
        Args:
            image: PIL Image in RGB format
            new_group_id: Unique identifier for this image group
            
        Returns:
            Dictionary with 'generated_caption', 'detected_objects_json', and 'extracted_tags_json'
        """
        print(f"  > Running EXPENSIVE AI pipeline (v2) for new group: {new_group_id}")
        ai_results = {}
        
        # Solution 4: Generate Caption with Qwen2-VL
        print(f"  > Generating caption with Qwen2-VL...")
        
        # Prepare chat template for Qwen2-VL
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": "Describe this image in detail."}
                ]
            }
        ]
        
        # Process with Qwen2-VL
        text_prompt = self.models.caption_processor.apply_chat_template(
            messages,
            add_generation_prompt=True
        )
        
        inputs = self.models.caption_processor(
            text=[text_prompt],
            images=[image],
            return_tensors="pt"
        ).to(self.models.device)
        
        generated_ids = self.models.caption_model.generate(
            **inputs,
            max_new_tokens=MAX_CAPTION_TOKENS
        )
        
        generated_text = self.models.caption_processor.batch_decode(
            generated_ids,
            skip_special_tokens=True
        )[0]
        
        # Extract just the assistant's response
        if "assistant" in generated_text:
            caption = generated_text.split("assistant")[-1].strip()
        else:
            caption = generated_text.strip()
            
        ai_results["generated_caption"] = caption
        print(f"  > Generated Caption: '{caption}'")
        
        # Solution 3: Extract Tags from Caption
        tags = self.extract_tags_from_caption(caption)
        ai_results["extracted_tags_json"] = json.dumps(tags)
        print(f"  > Extracted Tags: {tags}")
        
        # Object Detection with Face Grouping (Solution 1)
        detect_inputs = self.models.detect_processor(
            images=image,
            return_tensors="pt"
        ).to(self.models.device)
        
        outputs = self.models.detect_model(**detect_inputs)
        
        # Post-process to get bounding boxes and labels
        target_sizes = torch.tensor([image.size[::-1]], device=self.models.device)
        detections = self.models.detect_processor.post_process_object_detection(
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
            label = self.models.detect_model.config.id2label[label_id.item()]
            detection_data = {
                "label": label,
                "score": score.item(),
                "box": box.tolist()
            }
            
            # Solution 1: Face Recognition for "person" objects
            if label == "person":
                try:
                    # Crop the person from the image
                    box_coords = [int(c) for c in box.tolist()]
                    person_crop = image.crop((box_coords[0], box_coords[1], box_coords[2], box_coords[3]))
                    
                    # Create face embedding
                    person_embedding = self.get_embedding(
                        person_crop,
                        self.models.embed_model,
                        self.models.embed_processor
                    )
                    
                    # Get or create person group
                    person_group_id = self.get_or_create_person_group(person_embedding)
                    detection_data["person_group_id"] = person_group_id
                    
                except Exception as e:
                    print(f"  > Face embedding failed: {e}")
                    
            detected_objects.append(detection_data)
        
        ai_results["detected_objects_json"] = json.dumps(detected_objects)
        print(f"  > Detected {len(detected_objects)} objects (with face grouping)")
        
        return ai_results
        
    def process_new_image(self, image_path: str) -> bool:
        """
        Main processing function for a new image file (v2).
        Implements: (1) Embed → (2) Search → (3) Cache/Process logic
        
        Args:
            image_path: Full path to the image file
            
        Returns:
            True if successful, False if failed
        """
        # Check if already processed
        self.db.cursor.execute(
            "SELECT group_id FROM images WHERE image_path = ?",
            (image_path,)
        )
        if self.db.cursor.fetchone():
            print(f"  > Already processed: {image_path}")
            return True
            
        try:
            # Load image
            image = Image.open(image_path).convert("RGB")
        except Exception as e:
            print(f"  > FAILED to load image: {e}")
            return False
            
        # Step 1: Create Fingerprint (Embedding)
        embedding = self.get_embedding(
            image,
            self.models.embed_model,
            self.models.embed_processor
        )
        
        # Step 2: Check for Duplicates (Cache Check)
        similarity_score = 0.0
        matched_faiss_id = -1
        
        if self.db.faiss_image_index.ntotal > 0:
            # Search FAISS for the 1 nearest neighbor
            scores, faiss_ids = self.db.faiss_image_index.search(embedding, k=1)
            similarity_score = scores[0][0]
            matched_faiss_id = faiss_ids[0][0]
            
        # Step 3: Branching Logic
        if similarity_score > VISUAL_SIMILARITY_THRESHOLD:
            # CACHE HIT (Near-Duplicate)
            print(f"  > STATUS: Near-Duplicate Found (Score: {similarity_score:.4f})")
            
            # Get the group_id from the matched image
            self.db.cursor.execute(
                "SELECT group_id FROM images WHERE faiss_image_id = ?",
                (int(matched_faiss_id),)
            )
            result = self.db.cursor.fetchone()
            
            if not result:
                print("  > ERROR: FAISS-DB mismatch. Treating as new image.")
                # Fall through to cache miss logic
                similarity_score = 0.0
            else:
                matched_group_id = result[0]
                print(f"  > ACTION: Reusing results from Group: {matched_group_id}")
                
                # Add this new image to the DB, linking to existing group
                self.db.cursor.execute(
                    "INSERT OR REPLACE INTO images (image_path, group_id, faiss_image_id) VALUES (?, ?, ?)",
                    (image_path, matched_group_id, None)
                )
                self.db.db_conn.commit()
                return True
                
        if similarity_score <= VISUAL_SIMILARITY_THRESHOLD:
            # CACHE MISS (Unique Photo)
            print(f"  > STATUS: Unique Photo (Top score: {similarity_score:.4f})")
            
            # Create a new unique ID for this group
            new_group_id = f"group_{self.db.faiss_image_index.ntotal}"
            
            # Add this photo's vector to the FAISS index
            new_faiss_id = self.db.faiss_image_index.ntotal
            self.db.faiss_image_index.add(embedding)
            
            # Add this image to the images table
            self.db.cursor.execute(
                "INSERT OR REPLACE INTO images (image_path, group_id, faiss_image_id) VALUES (?, ?, ?)",
                (image_path, new_group_id, int(new_faiss_id))
            )
            
            # Run the expensive AI pipeline (v2)
            ai_results = self.run_expensive_ai_pipeline(image, new_group_id)
            
            # Store the AI results in the image_groups table
            self.db.cursor.execute(
                "INSERT OR REPLACE INTO image_groups (group_id, canonical_path, generated_caption, detected_objects_json, extracted_tags_json) VALUES (?, ?, ?, ?, ?)",
                (
                    new_group_id,
                    image_path,
                    ai_results["generated_caption"],
                    ai_results["detected_objects_json"],
                    ai_results["extracted_tags_json"]
                )
            )
            
            self.db.db_conn.commit()
            print(f"  > ACTION: New results saved to Group: {new_group_id}")
            return True
            
        return False


# Global instance
_processor_instance = None


def get_processor():
    """
    Get or create the global ImageProcessor instance.
    """
    global _processor_instance
    if _processor_instance is None:
        _processor_instance = ImageProcessor()
    return _processor_instance
