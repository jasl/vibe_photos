# AI Photo Management: Implementation Plan (v2.0)

This document outlines the technical implementation steps to build an advanced AI-powered photo management system.
This plan is an extension of the previous design, incorporating four new SOTA (State-of-the-Art) features while adhering to a 16GB VRAM hardware budget (e.g., M4 Pro / GeForce 5070-class).

Core Objectives:

1. Face Grouping (Solution 1): Go beyond generic "person" tags. Detect, extract, and group unique faces.
2. Semantic Merging (Solution 2): Implement a maintenance workflow to find and merge visually distinct groups that have highly similar (or identical) text captions, correcting for errors caused by lighting or angle changes.
3. Tag Extraction (Solution 3): Automatically parse the generative caption to extract key nouns (e.g., "pizza," "laptop") to create a searchable tag list.
4. Model Upgrade (Solution 4): Upgrade the generative caption model to Qwen/Qwen3-VL-8B-Thinking-FP8 to provide richer, more detailed descriptions.

Technical Corrections Applied:

1. FAISS Library: All pip install commands will use faiss-cpu, which is the currently maintained standard, instead of the deprecated faiss-gpu.
2. Dtype Precision: The torch_dtype parameter in from_pretrained() will be explicitly set to torch.float16 (DTYPE). This is a mandatory step. Using "auto" would load the models in their default float32 precision, which would exceed the 16GB VRAM budget and cause an Out-of-Memory (OOM) error.

## 1. Installation and Dependencies

This file (requirements.txt or pip install commands) sets up the environment.

```bash
# Install core libraries
pip install transformers torch pillow requests accelerate
# Install the maintained FAISS CPU library
pip install faiss-cpu
# Install numpy and sqlite for data storage
pip install numpy sqlite3
```

Note: `accelerate` is required for `device_map="auto"`, which intelligently distributes model layers across your GPU and CPU RAM, making it possible to run this multi-model stack within 16GB of VRAM.

## 2. Core Component Initialization

This script (`initialize.py` or similar) loads all models and databases into memory.

```python
import torch
import requests
import faiss
import numpy as np
import sqlite3
import json
from PIL import Image
from transformers import (
    AutoImageProcessor, 
    AutoModel, 
    AutoModelForObjectDetection,
    AutoModelForCausalLM, # For Qwen3-VL
    AutoProcessor,
    pipeline # For the POS tagger
)
from transformers.image_utils import load_image

# --- 2.1. Global Configuration ---
# Threshold for visual de-duplication (very strict)
VISUAL_SIMILARITY_THRESHOLD = 0.98 
# Threshold for face recognition (more lenient)
FACE_SIMILARITY_THRESHOLD = 0.92
# Threshold for merging groups based on text (strict)
SEMANTIC_SIMILARITY_THRESHOLD = 0.95

# --- 2.2. SOTA Model Checkpoints ---
# 1. Embedding Model (SigLIP): For image/face fingerprints and text search [10, 11, 12, 13, 14]
EMBEDDING_MODEL_ID = "google/siglip2-so400m-patch14-384"
EMBEDDING_DIM = 1152 # Fixed dimension for SigLIP2-so400m [15, 16, 17, 18]

# 2. Object Detection (RT-DETR): SOTA for accuracy in transformers [19, 20]
DETECTION_MODEL_ID = "PekingU/rtdetr_r101vd"

# 3. Captioning Model (Qwen3-VL): Upgraded for rich descriptions [21]
CAPTION_MODEL_ID = "Qwen/Qwen3-VL-8B-Thinking-FP8"

# 4. Tag Extraction Model (BERT): Lightweight POS-tagger
TAGGER_MODEL_ID = "vblagoje/bert-english-uncased-finetuned-pos"

# --- 2.3. Hardware Device Setup ---
def get_inference_device_and_dtype():
    """Detects and returns the optimal device and dtype."""
    if torch.cuda.is_available():
        print("Using CUDA (NVIDIA GPU). Setting DTYPE to float16.")
        # Explicitly setting float16 is REQUIRED to fit models in 16GB VRAM.
        return torch.device("cuda"), torch.float16
    if torch.backends.mps.is_available(): # For Apple M-series [22, 23, 24, 25]
        print("Using MPS (Apple Silicon GPU). Setting DTYPE to float16.")
        return torch.device("mps"), torch.float16
    print("No GPU detected. Using CPU. Setting DTYPE to float32.")
    return torch.device("cpu"), torch.float32

DEVICE, DTYPE = get_inference_device_and_dtype()

# --- 2.4. Load AI Models ---
# We use device_map="auto" and torch_dtype=DTYPE to load these large models
# efficiently within our 16GB VRAM budget. [26, 9, 27]

print(f"Loading Model 1: Embedding ({EMBEDDING_MODEL_ID})")
embed_processor = AutoProcessor.from_pretrained(EMBEDDING_MODEL_ID)
embed_model = AutoModel.from_pretrained(
    EMBEDDING_MODEL_ID,
    torch_dtype=DTYPE,
    device_map="auto"
).eval()

print(f"Loading Model 2: Object Detection ({DETECTION_MODEL_ID})")
detect_processor = AutoImageProcessor.from_pretrained(DETECTION_MODEL_ID)
detect_model = AutoModelForObjectDetection.from_pretrained(
    DETECTION_MODEL_ID,
    torch_dtype=DTYPE,
    device_map="auto"
).eval()

print(f"Loading Model 3: Captioning ({CAPTION_MODEL_ID})")
caption_processor = AutoProcessor.from_pretrained(CAPTION_MODEL_ID, trust_remote_code=True)
caption_model = AutoModelForCausalLM.from_pretrained(
    CAPTION_MODEL_ID,
    torch_dtype=DTYPE,
    device_map="auto",
    trust_remote_code=True
).eval()

print(f"Loading Model 4: Tag Extraction ({TAGGER_MODEL_ID})")
# This model is small and can be loaded directly to the device
pos_tagger = pipeline(
    "token-classification", 
    model=TAGGER_MODEL_ID, 
    device=DEVICE
)

print("All models loaded successfully.")

# --- 2.5. Vector Database (FAISS) Initialization ---
# Index 1: For full-image visual fingerprints
try:
    faiss_image_index = faiss.read_index("photo_library.index")
    print(f"Loaded existing FAISS image index ({faiss_image_index.ntotal} vectors).")
except:
    faiss_image_index = faiss.IndexFlatIP(EMBEDDING_DIM)
    print("Created new FAISS image index.")

# Index 2: For cropped face fingerprints (Solution 1)
try:
    faiss_face_index = faiss.read_index("face_library.index")
    print(f"Loaded existing FAISS face index ({faiss_face_index.ntotal} vectors).")
except:
    faiss_face_index = faiss.IndexFlatIP(EMBEDDING_DIM)
    print("Created new FAISS face index.")

# --- 2.6. Metadata Database (SQLite) Initialization ---
db_conn = sqlite3.connect("photo_metadata.db")
cursor = db_conn.cursor()

# Table 1: Stores every individual image file
cursor.execute("""
CREATE TABLE IF NOT EXISTS images (
    image_path TEXT PRIMARY KEY,
    group_id TEXT,
    faiss_image_id INTEGER
)
""")

# Table 2: Stores AI results for each *unique* group
cursor.execute("""
CREATE TABLE IF NOT EXISTS image_groups (
    group_id TEXT PRIMARY KEY,
    canonical_path TEXT,
    generated_caption TEXT,
    detected_objects_json TEXT,
    extracted_tags_json TEXT  -- NEW (Solution 3)
)
""")

# Table 3: Stores each unique person (Solution 1)
cursor.execute("""
CREATE TABLE IF NOT EXISTS person_groups (
    person_group_id TEXT PRIMARY KEY,
    name TEXT, -- User-editable name (e.g., "Mom", "John Doe")
    faiss_face_id INTEGER
)
""")
db_conn.commit()
print("SQLite database connected and tables ensured.")
```

## 3. Core AI Workflow (Functions)

These functions perform the heavy lifting. They are called by the main ingest workflow.

```python
@torch.no_grad()
def get_embedding(image, model, processor):
    """
    Generates a SOTA feature vector for any PIL image (full or crop).
    Uses the SigLIP2 model.[28, 11]
    """
    inputs = processor(images=image, return_tensors="pt").to(DEVICE, DTYPE)
    
    image_features = model.get_image_features(**inputs["pixel_values"])
    
    # Convert to FAISS-compatible format (CPU, float32, L2-normalized)
    vector = image_features.cpu().numpy().astype(np.float32)
    faiss.normalize_L2(vector) # [29]
    return vector

@torch.no_grad()
def get_or_create_person_group(person_embedding):
    """
    Solution 1: Face Grouping.
    Searches the face index. If a match is found, returns the existing ID.
    If not, creates a new person group and returns the new ID.
    """
    global faiss_face_index
    
    if faiss_face_index.ntotal > 0:
        scores, faiss_ids = faiss_face_index.search(person_embedding, k=1)
        similarity_score = scores
        matched_faiss_id = faiss_ids
    else:
        similarity_score = 0.0

    if similarity_score > FACE_SIMILARITY_THRESHOLD:
        # HIT: This is a known person
        cursor.execute("SELECT person_group_id FROM person_groups WHERE faiss_face_id =?", (int(matched_faiss_id),))
        result = cursor.fetchone()
        if result:
            return result
    
    # MISS: This is a new person
    new_faiss_id = faiss_face_index.ntotal
    faiss_face_index.add(person_embedding)
    
    new_person_group_id = f"person_{new_faiss_id}"
    default_name = f"Person {new_faiss_id + 1}" 
    
    cursor.execute("INSERT INTO person_groups (person_group_id, name, faiss_face_id) VALUES (?,?,?)",
                   (new_person_group_id, default_name, int(new_faiss_id)))
    db_conn.commit()
    
    return new_person_group_id


@torch.no_grad()
def run_expensive_ai_pipeline(image, new_group_id):
    """
    (UPDATED)
    Runs the full analysis suite on a *unique* image.
    Integrates all new solutions.
    """
    print(f"  > Running EXPENSIVE AI pipeline for new group: {new_group_id}")
    ai_results = {}

    # --- 3.1: Solution 4 (Caption Upgrade) ---
    # Use Qwen3-VL for rich, generative description [21]
    prompt_messages =}]
    inputs = caption_processor.apply_chat_template(prompt_messages, images=[image], return_tensors="pt").to(DEVICE)
    
    generated_ids = caption_model.generate(inputs, max_new_tokens=100)
    caption = caption_processor.batch_decode(generated_ids, skip_special_tokens=True).strip()
    ai_results["generated_caption"] = caption
    print(f"  > (Qwen3-VL) Caption: '{caption}'")

    # --- 3.2: Solution 3 (Tag Extraction) ---
    # Parse the new caption to extract nouns as searchable tags
    tags_list =
    try:
        pos_results = pos_tagger(caption)
        # Extract all Nouns (NN, NNS) and Proper Nouns (NNP)
        for entity in pos_results:
            if entity['entity'] in:
                tag = entity['word'].replace("##", "").strip()
                if tag:
                    tags_list.append(tag)
        tags_list = list(set(tags_list)) # De-duplicate
    except Exception as e:
        print(f"  > POS Tagger failed: {e}")
        
    ai_results["extracted_tags_json"] = json.dumps(tags_list)
    print(f"  > (Tagger) Extracted Tags: {tags_list}")

    # --- 3.3: Solution 1 (Face/Object Detection) ---
    detect_inputs = detect_processor(images=image, return_tensors="pt").to(DEVICE)
    outputs = detect_model(**detect_inputs)
    
    target_sizes = torch.tensor([image.size[::-1]], device=DEVICE)
    detections = detect_processor.post_process_object_detection(
        outputs, 
        threshold=0.9, # High-confidence
        target_sizes=target_sizes
    ) # 
    
    detected_objects_list =
    
    for score, label_id, box in zip(detections["scores"], detections["labels"], detections["boxes"]):
        label = detect_model.config.id2label[label_id.item()]
        detection_data = {
            "label": label,
            "score": score.item(),
            "box": box.tolist()
        }

        # --- Face Recognition Logic ---
        if label == "person":
            try:
                box_int = [int(coord) for coord in box.tolist()]
                person_crop = image.crop((box_int, box_int[1], box_int[2], box_int[3]))
                
                # Create a "fingerprint" for the detected person
                person_embedding = get_embedding(person_crop, embed_model, embed_processor)
                
                # Get or create their unique ID
                person_group_id = get_or_create_person_group(person_embedding)
                detection_data["person_group_id"] = person_group_id
                
            except Exception as e:
                print(f"  > Face embedding failed: {e}")

        detected_objects_list.append(detection_data)
        
    ai_results["detected_objects_json"] = json.dumps(detected_objects_list)
    print(f"  > (RT-DETR) Detected {len(detected_objects_list)} objects (with face grouping).")

    return ai_results
```

## 4. Ingest Workflow (Main Executable)

This is the main script to run when adding new photos. It implements the caching logic to avoid re-processing duplicates.

```python
# Main executable script: ingest_photos.py

def process_new_image(image_path):
    """
    (UPDATED)
    Main processing function for a new image file.
    Implements the (1) Embed -> (2) Search -> (3) Cache/Process logic.
    """
    print(f"\n--- Processing new file: {image_path} ---")
    
    # Check if already processed
    cursor.execute("SELECT group_id FROM images WHERE image_path =?", (image_path,))
    if cursor.fetchone():
        print("  > STATUS: Already processed. Skipping.")
        return

    try:
        image = load_image(image_path).convert("RGB")
    except Exception as e:
        print(f"  > FAILED to load image. Skipping. Error: {e}")
        return

    # 1. Create Fingerprint (Embedding)
    embedding = get_embedding(image, embed_model, embed_processor)

    # 2. Check for Visual Duplicates (Cache Check)
    if faiss_image_index.ntotal > 0:
        scores, faiss_ids = faiss_image_index.search(embedding, k=1)
        similarity_score = scores
        matched_faiss_id = faiss_ids
    else:
        similarity_score = 0.0
        matched_faiss_id = -1

    # 3. Branching Logic
    if similarity_score > VISUAL_SIMILARITY_THRESHOLD:
        # --- CACHE HIT (Near-Duplicate) ---
        print(f"  > STATUS: Cache Hit (Visual Similarity: {similarity_score:.4f})")
        
        cursor.execute("SELECT group_id FROM images WHERE faiss_image_id =?", (int(matched_faiss_id),))
        result = cursor.fetchone()
        
        if not result:
            print("  > ERROR: FAISS-DB mismatch. Treating as new image.")
            similarity_score = 0.0 # Force Cache Miss
        else:
            matched_group_id = result
            print(f"  > ACTION: Reusing results from Group: {matched_group_id}")
            
            # Add this new photo to `images` table, linking to the *existing* group
            cursor.execute("INSERT OR REPLACE INTO images (image_path, group_id, faiss_image_id) VALUES (?,?,?)",
                           (image_path, matched_group_id, None))
            db_conn.commit()
            return # Finished. Skipped expensive AI.

    if similarity_score <= VISUAL_SIMILARITY_THRESHOLD:
        # --- CACHE MISS (Unique Photo) ---
        print(f"  > STATUS: Unique Photo (Top score: {similarity_score:.4f}).")
        
        # 1. Create new unique group ID
        new_group_id = f"group_{faiss_image_index.ntotal}"
        
        # 2. Add vector to FAISS index
        new_faiss_id = faiss_image_index.ntotal
        faiss_image_index.add(embedding)
        
        # 3. Add this image to `images` table as the "canonical" image
        cursor.execute("INSERT OR REPLACE INTO images (image_path, group_id, faiss_image_id) VALUES (?,?,?)",
                       (image_path, new_group_id, int(new_faiss_id)))
        
        # 4. Run the full, expensive AI pipeline
        ai_results = run_expensive_ai_pipeline(image, new_group_id)
        
        # 5. Store the *new* AI results in the `image_groups` table
        cursor.execute(
            "INSERT OR REPLACE INTO image_groups (group_id, canonical_path, generated_caption, detected_objects_json, extracted_tags_json) VALUES (?,?,?,?,?)",
            (
                new_group_id,
                image_path,
                ai_results["generated_caption"],
                ai_results["detected_objects_json"],
                ai_results["extracted_tags_json"] # Store new tags
            )
        )
        db_conn.commit()
        print(f"  > ACTION: New results saved to Group: {new_group_id}")

# --- Example Execution ---
if __name__ == "__main__":
    # Example: processing a directory
    import glob
    my_photo_directory = "/path/to/your/photos"
    image_paths = glob.glob(f"{my_photo_directory}/*.jpg")
    
    for path in image_paths:
        process_new_image(path)
        
    # Save the updated FAISS indexes
    faiss.write_index(faiss_image_index, "photo_library.index")
    faiss.write_index(faiss_face_index, "face_library.index")
    print("\n--- Ingest complete. All indexes saved. ---")
    
    db_conn.close()
```

## 5. Maintenance Workflow: Semantic Merging (Solution 2)

This is a separate script (run_maintenance.py) to run periodically (e.g., nightly) to clean up the database by merging groups that have similar captions.

```python
# File: run_maintenance.py

import torch
import faiss
import numpy as np
import sqlite3
from transformers import AutoProcessor, AutoModel

# --- 1. Load Required Components ---
EMBEDDING_MODEL_ID = "google/siglip2-so400m-patch14-384"
EMBEDDING_DIM = 1152 # [15, 16, 17, 18]
SEMANTIC_SIMILARITY_THRESHOLD = 0.95 # Your tuning parameter
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

print("Loading Embedding Model (for text)...")
embed_processor = AutoProcessor.from_pretrained(EMBEDDING_MODEL_ID)
embed_model = AutoModel.from_pretrained(
    EMBEDDING_MODEL_ID,
    torch_dtype=torch.float16 if DEVICE == "cuda" else torch.float32,
    device_map="auto"
).eval()

db_conn = sqlite3.connect("photo_metadata.db")
cursor = db_conn.cursor()

@torch.no_grad()
def get_text_embedding(text_list):
    """Generates SigLIP embeddings for a batch of text captions."""
    inputs = embed_processor(text=text_list, return_tensors="pt", padding=True, truncation=True).to(DEVICE)
    text_features = embed_model.get_text_features(**inputs) #
    
    vector = text_features.cpu().numpy().astype(np.float32)
    faiss.normalize_L2(vector)
    return vector

def merge_groups(primary_group_id, secondary_group_id):
    """Consolidates two groups in the database."""
    print(f"  > MERGING: {secondary_group_id} -> {primary_group_id}")
    try:
        # 1. Re-assign all images from the secondary group to the primary group
        cursor.execute("UPDATE images SET group_id =? WHERE group_id =?", (primary_group_id, secondary_group_id))
        
        # 2. Delete the now-redundant secondary group entry
        cursor.execute("DELETE FROM image_groups WHERE group_id =?", (secondary_group_id,))
        
        db_conn.commit()
    except Exception as e:
        print(f"  > MERGE FAILED: {e}")
        db_conn.rollback()

def run_semantic_merge_task():
    print("\n--- Starting Semantic Merge Maintenance Task ---")
    
    # 1. Get all unique captions and their group IDs
    cursor.execute("SELECT group_id, generated_caption FROM image_groups")
    rows = cursor.fetchall()
    if len(rows) < 2:
        print("Not enough groups to merge. Exiting.")
        return

    group_ids = [row for row in rows]
    captions = [row[1] for row in rows]
    
    print(f"Creating text embeddings for {len(captions)} unique captions...")
    
    # 2. Create embeddings for all captions
    # (This should be batched if len(captions) is very large)
    caption_embeddings = get_text_embedding(captions)
    
    # 3. Build a temporary FAISS index for captions
    caption_index = faiss.IndexFlatIP(EMBEDDING_DIM)
    caption_index.add(caption_embeddings)
    
    print("Searching caption index for semantic duplicates...")
    
    # 4. Search the index for nearest neighbors (k=5)
    k = 5 
    all_scores, all_indices = caption_index.search(caption_embeddings, k=k)
    
    merged_groups = set()
    total_merges = 0
    
    # 5. Iterate and apply merge logic
    for i in range(len(all_indices)):
        primary_group_id = group_ids[i]
        
        if primary_group_id in merged_groups:
            continue
            
        # Check neighbors (skip j=0, which is itself)
        for j in range(1, k):
            neighbor_index = all_indices[i][j]
            neighbor_score = all_scores[i][j]
            
            secondary_group_id = group_ids[neighbor_index]
            
            if secondary_group_id in merged_groups or primary_group_id == secondary_group_id:
                continue

            # 6. Decision: Merge if semantically similar enough
            if neighbor_score > SEMANTIC_SIMILARITY_THRESHOLD:
                merge_groups(primary_group_id, secondary_group_id)
                merged_groups.add(secondary_group_id) # Mark as merged
                total_merges += 1

    print(f"--- Semantic Merge Complete. Total merges: {total_merges} ---")

if __name__ == "__main__":
    run_semantic_merge_task()
    db_conn.close()
```

## 6. Search Workflow (For Application UI)

This is the final piece: the search function your application will call.

```python
# File: search.py
# (Load models and DBs as in Section 2)

@torch.no_grad()
def search_photo_library(text_query, top_k=5):
    """
    (UPDATED)
    Performs a semantic search of the photo library using a text query.
    """
    print(f"\n--- Searching for: '{text_query}' ---")
    
    # 1. Create Text Embedding for the query
    inputs = embed_processor(text=[text_query], return_tensors="pt", padding=True, truncation=True).to(DEVICE)
    text_embedding = embed_model.get_text_features(**inputs)
    
    # 2. Normalize and prepare for FAISS
    vector = text_embedding.cpu().numpy().astype(np.float32)
    faiss.normalize_L2(vector)
    
    # 3. Search the IMAGE index (faiss_image_index)
    scores, faiss_ids = faiss_image_index.search(vector, k=top_k)
    
    # 4. Retrieve Full Results from SQLite DB
    print(f"--- Top {top_k} Results ---")
    faiss_id_list = [int(i) for i in faiss_ids]
    
    # Build a query to get all results in one go
    query_placeholders = ",".join("?" * len(faiss_id_list))
    cursor.execute(
        f"SELECT i.image_path, i.group_id, g.generated_caption, g.extracted_tags_json "
        f"FROM images i "
        f"JOIN image_groups g ON i.group_id = g.group_id "
        f"WHERE i.faiss_id IN ({query_placeholders})",
        faiss_id_list
    )
    
    # Create a map to re-order results based on FAISS score
    results_map = {row: (row[1], row[2], row[3]) for row in cursor.fetchall()}

    # Print ordered results
    for i, (faiss_id, score) in enumerate(zip(faiss_id_list, scores)):
        cursor.execute("SELECT image_path FROM images WHERE faiss_id =?", (faiss_id,))
        image_path_tuple = cursor.fetchone()
        
        if image_path_tuple:
            image_path = image_path_tuple
            if image_path in results_map:
                group_id, caption, tags = results_map[image_path]
                print(f"Rank {i+1} (Similarity: {score:.4f})")
                print(f"  Path: {image_path}")
                print(f"  Group: {group_id}")
                print(f"  Caption: {caption}")
                print(f"  Tags: {json.loads(tags)}\n")
            else:
                # This can happen if the image is a duplicate (no faiss_id)
                # A full implementation would look up the group_id instead
                print(f"Rank {i+1} (FAISS ID {faiss_id}) - Result not found (is it a duplicate?)")
```
