# AI Photo Management: Implementation Plan

This document outlines the complete technical workflow for building the AI-powered photo management application.

## 1. Setup and Dependencies

Install the required Python libraries. `accelerate` is required for smart model distribution on your hardware (`device_map="auto"`). `faiss-gpu` is for high-speed vector search.python pip install transformers torch pillow requests faiss-gpu accelerate

## 2. Core Component Initialization

This script loads all selected State-of-the-Art (SOTA) models and initializes the database components. Your 16GB VRAM budget supports loading these models in `float16` for high performance.

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
    Blip2ForConditionalGeneration,
    AutoProcessor
)
from transformers.image_utils import load_image

# --- 2.1. Global Configuration ---
# Set a high similarity threshold for finding "near-duplicates"
# 1.0 is identical. 0.98+ is a very close match (e.g., burst photos).
SIMILARITY_THRESHOLD = 0.98

# Define SOTA model checkpoints
EMBEDDING_MODEL_ID = "google/siglip2-so400m-patch14-384"
DETECTION_MODEL_ID = "PekingU/rtdetr_r101vd"
CAPTION_MODEL_ID = "Salesforce/blip2-opt-2.7b"

# Define embedding dimension for SigLIP2 model
# This is a fixed value from the model's configuration
EMBEDDING_DIM = 1152 #

# --- 2.2. Hardware Device Setup ---
# Auto-detect best available device (NVIDIA CUDA, Apple MPS, or CPU)
def get_inference_device():
    if torch.cuda.is_available():
        print("Using CUDA (NVIDIA GPU)")
        return torch.device("cuda"), torch.float16
    if torch.backends.mps.is_available():
        print("Using MPS (Apple Silicon GPU)")
        return torch.device("mps"), torch.float16
    print("No GPU detected. Using CPU (float32).")
    return torch.device("cpu"), torch.float32

DEVICE, DTYPE = get_inference_device_and_dtype()

# --- 2.3. Load AI Models (Quality-First) ---
# Use device_map="auto" and float16 for efficient 16GB VRAM usage

print(f"Loading Model 1: Embedding (SOTA: {EMBEDDING_MODEL_ID})")
embed_processor = AutoProcessor.from_pretrained(EMBEDDING_MODEL_ID)
embed_model = AutoModel.from_pretrained(
    EMBEDDING_MODEL_ID,
    torch_dtype=DTYPE,
    device_map="auto"
).eval()

print(f"Loading Model 2: Object Detection (SOTA: {DETECTION_MODEL_ID})")
detect_processor = AutoImageProcessor.from_pretrained(DETECTION_MODEL_ID)
detect_model = AutoModelForObjectDetection.from_pretrained(
    DETECTION_MODEL_ID,
    torch_dtype=DTYPE,
    device_map="auto"
).eval()

print(f"Loading Model 3: Generative Captioning (SOTA: {CAPTION_MODEL_ID})")
caption_processor = AutoProcessor.from_pretrained(CAPTION_MODEL_ID)
caption_model = Blip2ForConditionalGeneration.from_pretrained(
    CAPTION_MODEL_ID,
    torch_dtype=DTYPE,
    device_map="auto"
).eval()

print("All models loaded successfully.")

# --- 2.4. Vector Database (FAISS) Initialization ---
# Using IndexFlatIP (Inner Product). When vectors are L2-normalized,
# Inner Product is mathematically equivalent to Cosine Similarity.
try:
    faiss_index = faiss.read_index("photo_library.index")
    print(f"Loaded existing FAISS index with {faiss_index.ntotal} vectors.")
except:
    faiss_index = faiss.IndexFlatIP(EMBEDDING_DIM)
    print("Created new FAISS index.")

# --- 2.5. Metadata Database (SQLite) Initialization ---
db_conn = sqlite3.connect("photo_metadata.db")
cursor = db_conn.cursor()

# Table 1: Stores info for every single image file
cursor.execute("""
CREATE TABLE IF NOT EXISTS images (
    image_path TEXT PRIMARY KEY,
    group_id TEXT,
    faiss_id INTEGER
)
""")

# Table 2: Stores the AI results for each *unique* group
cursor.execute("""
CREATE TABLE IF NOT EXISTS image_groups (
    group_id TEXT PRIMARY KEY,
    canonical_path TEXT,
    generated_caption TEXT,
    detected_objects_json TEXT
)
""")
db_conn.commit()
print("SQLite database connected and tables ensured.")
```

## 3. Ingest & Caching Workflow (Main Logic)

This is the primary workflow that runs for every new photo added to the library.

### 3.1. Step 1: Create Image "Fingerprint" (Embedding)

This function creates the 1152-dimension vector for any given image. This is the only AI task run on every photo.

```python
@torch.no_grad()
def get_image_embedding(image):
    """
    Takes a PIL image, runs it through the SigLIP2 embedding model,
    and returns a 1152-dimension L2-normalized NumPy vector.
    """
    inputs = embed_processor(images=image, return_tensors="pt").to(DEVICE, DTYPE)
    
    # Get the image features (embedding)
    image_features = embed_model.get_image_features(**inputs["pixel_values"])
    
    # Move to CPU, convert to float32 for FAISS, and normalize
    vector = image_features.cpu().numpy().astype(np.float32)
    faiss.normalize_L2(vector) #
    return vector
```

### 3.2. Step 2: The "Expensive" AI Pipeline (For Unique Photos Only)

This function runs the heavy models (Detection, Captioning) and is only called if the photo is unique.

```python
@torch.no_grad()
def run_expensive_ai_pipeline(image, new_group_id):
    """
    Runs the full SOTA analysis on a unique image.
    This function performs your "Classify First" step.
    """
    print(f"  > Running EXPENSIVE AI pipeline for new group: {new_group_id}")
    ai_results = {}

    # --- 3.2.1: Generative Captioning (The "Classify First" Step) ---
    # This generates the descriptive text you requested.
    caption_inputs = caption_processor(image, return_tensors="pt").to(DEVICE, DTYPE)
    generated_ids = caption_model.generate(**caption_inputs, max_new_tokens=50)
    caption = caption_processor.batch_decode(generated_ids, skip_special_tokens=True).strip()
    ai_results["generated_caption"] = caption
    print(f"  > Generated Caption: '{caption}'")

    # --- 3.2.2: Object Detection ---
    inputs = detect_processor(images=image, return_tensors="pt").to(DEVICE)
    outputs = detect_model(**inputs)
    
    # Post-process to get bounding boxes and labels
    target_sizes = torch.tensor([image.size[::-1]], device=DEVICE)
    detections = detect_processor.post_process_object_detection(
        outputs, 
        threshold=0.9, # High-confidence threshold
        target_sizes=target_sizes
    ) #

    detected_objects = [
        {
            "label": detect_model.config.id2label[label.item()],
            "score": score.item(),
            "box": box.tolist()
        }
        for score, label, box in zip(detections["scores"], detections["labels"], detections["boxes"])
    ]
    
    ai_results["detected_objects_json"] = json.dumps(detected_objects)
    print(f"  > Detected {len(detected_objects)} objects.")

    return ai_results
```

### 3.3. Step 3: Main Processing Function (with Caching)

This function ties everything together. It checks for duplicates and decides whether to run the expensive pipeline or reuse cached results.

```python
def process_new_image(image_path):
    """
    Main processing function for a new image file.
    Implements the (1) Embed -> (2) Search -> (3) Cache/Process logic.
    """
    print(f"\n--- Processing new file: {image_path} ---")
    
    try:
        image = load_image(image_path).convert("RGB")
    except Exception as e:
        print(f"  > FAILED to load image. Skipping. Error: {e}")
        return

    # 1. Create Fingerprint (Embedding)
    embedding = get_image_embedding(image)

    # 2. Check for Duplicates (Cache Check)
    if faiss_index.ntotal > 0:
        # Search FAISS for the 1 nearest neighbor
        scores, faiss_ids = faiss_index.search(embedding, k=1)
        similarity_score = scores
        matched_faiss_id = faiss_ids
    else:
        similarity_score = 0.0
        matched_faiss_id = -1

    # 3. Branching Logic
    if similarity_score > SIMILARITY_THRESHOLD:
        # --- CACHE HIT (Near-Duplicate) ---
        print(f"  > STATUS: Near-Duplicate Found (Score: {similarity_score:.4f})")
        
        # 1. Get the group_id from the image we matched
        cursor.execute("SELECT group_id FROM images WHERE faiss_id =?", (int(matched_faiss_id),))
        result = cursor.fetchone()
        if not result:
            print("  > ERROR: FAISS-DB mismatch. Treating as new image.")
            # Fall through to the "CACHE MISS" logic
            similarity_score = 0.0
        else:
            matched_group_id = result
            print(f"  > ACTION: Reusing results from Group: {matched_group_id}")
            
            # 2. Add this new image to the DB, linking it to the *existing* group
            # We don't add its vector to FAISS to avoid cluttering the index
            cursor.execute("INSERT OR REPLACE INTO images (image_path, group_id, faiss_id) VALUES (?,?,?)",
                           (image_path, matched_group_id, None)) #
            db_conn.commit()
            return # Finished. Skipped expensive AI.

    if similarity_score <= SIMILARITY_THRESHOLD:
        # --- CACHE MISS (Unique Photo) ---
        print(f"  > STATUS: Unique Photo (Top score: {similarity_score:.4f})")
        
        # 1. Create a new unique ID for this group
        new_group_id = f"group_{faiss_index.ntotal}"
        
        # 2. Add this photo's vector to the FAISS index
        new_faiss_id = faiss_index.ntotal
        faiss_index.add(embedding)
        
        # 3. Add this image to the `images` table as the "canonical" image
        cursor.execute("INSERT OR REPLACE INTO images (image_path, group_id, faiss_id) VALUES (?,?,?)",
                       (image_path, new_group_id, int(new_faiss_id)))
        
        # 4. Run the "Expensive" AI Pipeline
        ai_results = run_expensive_ai_pipeline(image, new_group_id)
        
        # 5. Store the *new* AI results in the `image_groups` table
        cursor.execute(
            "INSERT OR REPLACE INTO image_groups (group_id, canonical_path, generated_caption, detected_objects_json) VALUES (?,?,?,?)",
            (
                new_group_id,
                image_path,
                ai_results["generated_caption"],
                ai_results["detected_objects_json"]
            )
        )
        db_conn.commit()
        print(f"  > ACTION: New results saved to Group: {new_group_id}")
```

## 4. Search & Retrieval Workflow

This workflow runs when you use your application's search bar. It uses the same embedding model (SigLIP2) to compare your text query against all indexed image "fingerprints."

```python
@torch.no_grad()
def search_photo_library(text_query, top_k=5):
    """
    Performs a semantic search of the photo library using a text query.
    """
    print(f"\n--- Searching for: '{text_query}' ---")
    
    # 1. Create Text Embedding
    # This is the "Zero-Shot" part: we compare text and images in the same space
    inputs = embed_processor(text=[text_query], return_tensors="pt").to(DEVICE)
    text_embedding = embed_model.get_text_features(**inputs) #
    
    # 2. Normalize and prepare for FAISS
    vector = text_embedding.cpu().numpy().astype(np.float32)
    faiss.normalize_L2(vector)
    
    # 3. Search FAISS
    scores, faiss_ids = faiss_index.search(vector, k=top_k)
    
    # 4. Retrieve Results from SQLite DB
    print("--- Top 5 Results ---")
    faiss_id_list = [int(i) for i in faiss_ids]
    
    # Build a query to get all results in one go
    query_placeholders = ",".join("?" * len(faiss_id_list))
    cursor.execute(
        f"SELECT i.image_path, i.group_id, g.generated_caption "
        f"FROM images i "
        f"JOIN image_groups g ON i.group_id = g.group_id "
        f"WHERE i.faiss_id IN ({query_placeholders})",
        faiss_id_list
    )
    
    # Map results for correct ordering
    results_map = {row: (row[1], row[2]) for row in cursor.fetchall()}

    # Print ordered results
    for i, (faiss_id, score) in enumerate(zip(faiss_id_list, scores)):
        cursor.execute("SELECT image_path FROM images WHERE faiss_id =?", (faiss_id,))
        image_path = cursor.fetchone()
        
        if image_path in results_map:
            group_id, caption = results_map[image_path]
            print(f"Rank {i+1} (Score: {score:.4f})")
            print(f"  Path: {image_path}")
            print(f"  Group: {group_id}")
            print(f"  Caption: {caption}\n")
        else:
            print(f"Rank {i+1} (ID {faiss_id}) - DB mapping error")
```

## 5. Full Workflow Execution Example

This block demonstrates how to use the functions defined above.

```python
if __name__ == "__main__":
    
    # --- 5.1. SIMULATE INGESTING NEW PHOTOS ---
    # We will process a list of photos. Note that two paths are identical
    # to simulate a "near-duplicate" for caching.

    # (Paths to example images - replace with your local file paths)
    # A photo of two cats
    CAT_PHOTO = "[http://images.cocodataset.org/val2017/000000039769.jpg](http://images.cocodataset.org/val2017/000000039769.jpg)"
    # A photo of a bear
    BEAR_PHOTO = "[https://huggingface.co/datasets/merve/coco/resolve/main/val2017/000000000285.jpg](https://huggingface.co/datasets/merve/coco/resolve/main/val2017/000000000285.jpg)"

    # Create dummy local files for the demo
    try:
        img_cat_data = requests.get(CAT_PHOTO).content
        with open("cat_photo_1.jpg", "wb") as f: f.write(img_cat_data)
        with open("cat_photo_2_burst.jpg", "wb") as f: f.write(img_cat_data) # Identical photo
            
        img_bear_data = requests.get(BEAR_PHOTO).content
        with open("bear_photo_1.jpg", "wb") as f: f.write(img_bear_data)

        photo_ingest_list =

        for photo_path in photo_ingest_list:
            process_new_image(photo_path)
            
    except Exception as e:
        print(f"Could not download demo images: {e}")


    # --- 5.2. SIMULATE SEARCHING ---
    # Now that the library is indexed, we can search it.
    
    search_photo_library(text_query="a picture of a cat")

    search_photo_library(text_query="a bear standing in a field")

    # --- 5.3. CLEANUP ---
    db_conn.close()
    # Save the FAISS index to disk for next time
    faiss.write_index(faiss_index, "photo_library.index")
    print("\n--- Workflow complete. Index saved. ---")
```

