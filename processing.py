"""
Image processing pipeline with embedding extraction, object detection, and captioning.
Implements caching logic to avoid redundant AI processing on duplicate images.
"""

import torch
import faiss
import numpy as np
import json
from PIL import Image
from typing import Dict, Optional

from config import (
    SIMILARITY_THRESHOLD,
    DETECTION_THRESHOLD,
    MAX_CAPTION_TOKENS
)
from models import get_models
from database import get_database


class ImageProcessor:
    """
    Handles the three-step image processing workflow:
    1. Create embedding (fingerprint)
    2. Check for duplicates
    3. Run expensive AI pipeline only for unique images
    """
    
    def __init__(self):
        self.models = get_models()
        self.db = get_database()
        
    @torch.no_grad()
    def get_image_embedding(self, image: Image.Image) -> np.ndarray:
        """
        Create a 1152-dimension L2-normalized embedding for an image.
        This is the lightweight operation run on every photo.
        
        Args:
            image: PIL Image in RGB format
            
        Returns:
            L2-normalized numpy array of shape (1, 1152)
        """
        inputs = self.models.embed_processor(
            images=image,
            return_tensors="pt"
        ).to(self.models.device)
        
        # Get image features (embedding)
        image_features = self.models.embed_model.get_image_features(**inputs)
        
        # Convert to float32 for FAISS and normalize
        vector = image_features.cpu().numpy().astype(np.float32)
        faiss.normalize_L2(vector)
        
        return vector
        
    @torch.no_grad()
    def run_expensive_ai_pipeline(
        self,
        image: Image.Image,
        new_group_id: str
    ) -> Dict[str, str]:
        """
        Run the full SOTA analysis on a unique image.
        This includes caption generation and object detection.
        
        Args:
            image: PIL Image in RGB format
            new_group_id: Unique identifier for this image group
            
        Returns:
            Dictionary with 'generated_caption' and 'detected_objects_json'
        """
        print(f"  > Running EXPENSIVE AI pipeline for new group: {new_group_id}")
        ai_results = {}
        
        # Step 1: Generate Caption
        caption_inputs = self.models.caption_processor(
            image,
            return_tensors="pt"
        ).to(self.models.device)
        
        generated_ids = self.models.caption_model.generate(
            **caption_inputs,
            max_new_tokens=MAX_CAPTION_TOKENS
        )
        
        caption = self.models.caption_processor.batch_decode(
            generated_ids,
            skip_special_tokens=True
        )[0].strip()
        
        ai_results["generated_caption"] = caption
        print(f"  > Generated Caption: '{caption}'")
        
        # Step 2: Object Detection
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
        
        detected_objects = [
            {
                "label": self.models.detect_model.config.id2label[label.item()],
                "score": score.item(),
                "box": box.tolist()
            }
            for score, label, box in zip(
                detections["scores"],
                detections["labels"],
                detections["boxes"]
            )
        ]
        
        ai_results["detected_objects_json"] = json.dumps(detected_objects)
        print(f"  > Detected {len(detected_objects)} objects")
        
        return ai_results
        
    def process_new_image(self, image_path: str) -> bool:
        """
        Main processing function for a new image file.
        Implements: (1) Embed → (2) Search → (3) Cache/Process logic
        
        Args:
            image_path: Full path to the image file
            
        Returns:
            True if successful, False if failed
        """
        try:
            # Load image
            image = Image.open(image_path).convert("RGB")
        except Exception as e:
            print(f"  > FAILED to load image: {e}")
            return False
            
        # Step 1: Create Fingerprint (Embedding)
        embedding = self.get_image_embedding(image)
        
        # Step 2: Check for Duplicates (Cache Check)
        similarity_score = 0.0
        matched_faiss_id = -1
        
        if self.db.faiss_index.ntotal > 0:
            # Search FAISS for the 1 nearest neighbor
            scores, faiss_ids = self.db.faiss_index.search(embedding, k=1)
            similarity_score = scores[0][0]
            matched_faiss_id = faiss_ids[0][0]
            
        # Step 3: Branching Logic
        if similarity_score > SIMILARITY_THRESHOLD:
            # CACHE HIT (Near-Duplicate)
            print(f"  > STATUS: Near-Duplicate Found (Score: {similarity_score:.4f})")
            
            # Get the group_id from the matched image
            self.db.cursor.execute(
                "SELECT group_id FROM images WHERE faiss_id = ?",
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
                    "INSERT OR REPLACE INTO images (image_path, group_id, faiss_id) VALUES (?, ?, ?)",
                    (image_path, matched_group_id, None)
                )
                self.db.db_conn.commit()
                return True
                
        if similarity_score <= SIMILARITY_THRESHOLD:
            # CACHE MISS (Unique Photo)
            print(f"  > STATUS: Unique Photo (Top score: {similarity_score:.4f})")
            
            # Create a new unique ID for this group
            new_group_id = f"group_{self.db.faiss_index.ntotal}"
            
            # Add this photo's vector to the FAISS index
            new_faiss_id = self.db.faiss_index.ntotal
            self.db.faiss_index.add(embedding)
            
            # Add this image to the images table
            self.db.cursor.execute(
                "INSERT OR REPLACE INTO images (image_path, group_id, faiss_id) VALUES (?, ?, ?)",
                (image_path, new_group_id, int(new_faiss_id))
            )
            
            # Run the expensive AI pipeline
            ai_results = self.run_expensive_ai_pipeline(image, new_group_id)
            
            # Store the AI results in the image_groups table
            self.db.cursor.execute(
                "INSERT OR REPLACE INTO image_groups (group_id, canonical_path, generated_caption, detected_objects_json) VALUES (?, ?, ?, ?)",
                (
                    new_group_id,
                    image_path,
                    ai_results["generated_caption"],
                    ai_results["detected_objects_json"]
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
