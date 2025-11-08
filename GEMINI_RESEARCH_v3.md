# AI Photo Management: Implementation Plan (v3 - 24GB VRAM)

This document outlines the complete technical workflow for building the AI-powered photo management system. It uses a "cascade" architecture to ensure fast importing, with deep, high-quality analysis running as a background task.

This plan is divided into six parts:

1. Installation: Libraries needed for the project.
2. Database Schema: The `setup_db.py` script to create the database.
3. Script 1: `ingest.py`: The fast script for adding new photos. It extracts metadata, filters duplicates, and queues unique photos.
4. Script 2: `process_queue.py`: The heavy script for background processing. It uses the 24GB VRAM to run high-quality analysis (face recognition, deep captioning) on the queued photos.
5. Script 3: `maintenance.py`: A script to run periodically to merge groups based on text similarity.
6. Script 4: `search.py`: The backend logic for your application's search bar.

## 1. Installation and Dependencies

Create a `requirements.txt` file. This plan uses `faiss-cpu` as it is the modern, maintained package.

```plain
# Core AI and ML libraries
transformers
torch
accelerate
pillow
# Vector database (maintained version)
faiss-cpu
# Standard utilities
numpy
sqlite3
requests
```

## 2. Database Schema (setup_db.py)

Run this script once to create the `photo_metadata.db` file with the complete schema.

```python
import sqlite3

db_conn = sqlite3.connect("photo_metadata.db")
cursor = db_conn.cursor()

# Table 1: Stores every single image file
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
    fast_caption TEXT,            -- Stores the fast BLIP-2 caption
    generated_caption TEXT,       -- Stores the deep Qwen3-VL caption
    detected_objects_json TEXT,
    extracted_tags_json TEXT,
    processing_status TEXT DEFAULT 'QUEUED', -- Tracks processing state
    taken_at_timestamp TEXT,      -- NEW: Stores ISO 8601 timestamp
    gps_latitude REAL,             -- NEW: Stores decimal latitude
    gps_longitude REAL             -- NEW: Stores decimal longitude
)
""")

# Table 3: Stores each unique person (Solution 1)
cursor.execute("""
CREATE TABLE IF NOT EXISTS person_groups (
    person_group_id TEXT PRIMARY KEY,
    name TEXT, -- User-editable name (e.g., "Mom", "John")
    faiss_face_id INTEGER
)
""")

# Table 4: Queue for deep processing
cursor.execute("""
CREATE TABLE IF NOT EXISTS processing_queue (
    group_id TEXT PRIMARY KEY,
    FOREIGN KEY (group_id) REFERENCES image_groups (group_id)
)
""")
db_conn.commit()
db_conn.close()
print("Database v3 schema is ready.")
```

## 3. Script 1: `ingest.py` (Fast Ingest & Filtering)

This script is your primary tool for adding photos. It is optimized for speed and uses a smaller set of models (~12GB VRAM) to quickly filter duplicates.

Models Loaded:

- `google/siglip2-so400m-patch14-384` (1B params)
- `Salesforce/blip2-opt-2.7b` (2.7B params)

```python
import torch
import faiss
import numpy as np
import sqlite3
import json
import os
from datetime import datetime
from PIL import Image
from PIL.ExifTags import TAGS
from transformers import (
    AutoProcessor, 
    AutoModel,
    Blip2ForConditionalGeneration,
    Blip2Processor
)
from transformers.image_utils import load_image

# --- 1. CONFIGURATION ---
VISUAL_SIMILARITY_THRESHOLD = 0.98 
SEMANTIC_SIMILARITY_THRESHOLD = 0.95 

# Models for THIS SCRIPT (Fast/Lightweight)
EMBEDDING_MODEL_ID = "google/siglip2-so400m-patch14-384"
EMBEDDING_DIM = 1152 #
FAST_CAPTION_MODEL_ID = "Salesforce/blip2-opt-2.7b"

# --- 2. HARDWARE SETUP ---
def get_inference_device_and_dtype():
    if torch.cuda.is_available():
        print("Using CUDA (NVIDIA GPU). Setting DTYPE to float16.")
        return torch.device("cuda"), torch.float16
    if torch.backends.mps.is_available(): # [7, 8, 9, 10, 11, 12, 13, 14]
        print("Using MPS (Apple Silicon GPU). Setting DTYPE to float16.")
        return torch.device("mps"), torch.float16
    print("No GPU detected. Using CPU (float32).")
    return torch.device("cpu"), torch.float32

DEVICE, DTYPE = get_inference_device_and_dtype()
# We explicitly set DTYPE to float16/bfloat16 to fit models in VRAM

# --- 3. LOAD MODELS (Fast Set) ---
print(f"Loading Model 1: Embedding ({EMBEDDING_MODEL_ID})")
embed_processor = AutoProcessor.from_pretrained(EMBEDDING_MODEL_ID)
embed_model = AutoModel.from_pretrained(
    EMBEDDING_MODEL_ID,
    torch_dtype=DTYPE,
    device_map="auto" #
).eval()

print(f"Loading Model 2: Fast Captioner ({FAST_CAPTION_MODEL_ID})")
fast_caption_processor = Blip2Processor.from_pretrained(FAST_CAPTION_MODEL_ID)
fast_caption_model = Blip2ForConditionalGeneration.from_pretrained(
    FAST_CAPTION_MODEL_ID,
    torch_dtype=DTYPE,
    device_map="auto" # [4, 5]
).eval()

# --- 4. LOAD DATABASES ---
print("Connecting to databases and FAISS...")
db_conn = sqlite3.connect("photo_metadata.db")
cursor = db_conn.cursor()

try:
    faiss_image_index = faiss.read_index("photo_library.index")
except:
    faiss_image_index = faiss.IndexFlatIP(EMBEDDING_DIM)

try:
    faiss_caption_index = faiss.read_index("caption_library.index")
except:
    faiss_caption_index = faiss.IndexFlatIP(EMBEDDING_DIM) 

# --- 5. CORE AI & METADATA FUNCTIONS ---

def _convert_gps_to_decimal(dms_coords, direction):
    """Helper function to convert EXIF GPS to decimal degrees."""
    degrees = dms_coords
    minutes = dms_coords[1]
    seconds = dms_coords[2]
    decimal = degrees + (minutes / 60.0) + (seconds / 3600.0)
    if direction in:
        decimal *= -1
    return decimal

def get_exif_data(image_obj, image_path):
    """
    (v3.1 Feature) Extracts Timestamp and GPS data from image EXIF.
    Falls back to file creation time.
    """
    timestamp = None
    latitude = None
    longitude = None

    try:
        # Get Fallback Time (File Creation)
        ctime = os.path.getctime(image_path)
        timestamp = datetime.fromtimestamp(ctime).isoformat()

        exif_data = image_obj._getexif()
        if exif_data:
            # 1. Get Shooting Time
            dt_original = exif_data.get(36867) # 36867 = DateTimeOriginal
            if dt_original:
                dt_obj = datetime.strptime(dt_original, '%Y:%m:%d %H:%M:%S')
                timestamp = dt_obj.isoformat()

            # 2. Get GPS Info
            gps_info = exif_data.get(34853) # 34853 = GPSInfo
            if gps_info:
                lat_dms, lat_ref = gps_info.get(2), gps_info.get(1)
                lon_dms, lon_ref = gps_info.get(4), gps_info.get(3)
                
                if lat_dms and lat_ref and lon_dms and lon_ref:
                    latitude = _convert_gps_to_decimal(lat_dms, lat_ref)
                    longitude = _convert_gps_to_decimal(lon_dms, lon_ref)
    except Exception as e:
        print(f"  > Warning: Could not parse EXIF data ({e}). Using file time.")
        
    return {"timestamp": timestamp, "latitude": latitude, "longitude": longitude}


@torch.no_grad()
def get_image_embedding(image):
    inputs = embed_processor(images=image, return_tensors="pt").to(DEVICE, DTYPE)
    image_features = embed_model.get_image_features(**inputs["pixel_values"]) #
    vector = image_features.cpu().numpy().astype(np.float32)
    faiss.normalize_L2(vector) # [15]
    return vector

@torch.no_grad()
def get_text_embedding(text):
    inputs = embed_processor(text=[text], return_tensors="pt", padding=True, truncation=True).to(DEVICE)
    text_features = embed_model.get_text_features(**inputs)
    vector = text_features.cpu().numpy().astype(np.float32)
    faiss.normalize_L2(vector)
    return vector

@torch.no_grad()
def get_fast_caption(image):
    inputs = fast_caption_processor(image, return_tensors="pt").to(DEVICE, DTYPE)
    generated_ids = fast_caption_model.generate(**inputs, max_new_tokens=50) # [5]
    caption = fast_caption_processor.batch_decode(generated_ids, skip_special_tokens=True).strip() # [4]
    return caption

# --- 6. MAIN INGEST WORKFLOW ---
def ingest_new_image(image_path):
    print(f"\n--- Processing: {image_path} ---")
    
    cursor.execute("SELECT group_id FROM images WHERE image_path =?", (image_path,))
    if cursor.fetchone():
        print("  > STATUS: Already processed. Skipping.")
        return

    try:
        image = Image.open(image_path).convert("RGB")
    except Exception as e:
        print(f"  > FAILED to load. Skipping. Error: {e}")
        return

    # --- STAGE 1: EXIF Extraction ---
    exif_data = get_exif_data(image, image_path)
    print(f"  > Metadata: Time={exif_data['timestamp']}, GPS=({exif_data['latitude']}, {exif_data['longitude']})")

    # --- STAGE 2: Visual De-duplication ---
    image_embedding = get_image_embedding(image)
    if faiss_image_index.ntotal > 0:
        scores, faiss_ids = faiss_image_index.search(image_embedding, k=1)
        visual_score, matched_image_faiss_id = scores, faiss_ids
    else:
        visual_score = 0.0

    if visual_score > VISUAL_SIMILARITY_THRESHOLD:
        print(f"  > STATUS: Visual Duplicate Found (Score: {visual_score:.4f})")
        cursor.execute("SELECT group_id FROM images WHERE faiss_image_id =?", (int(matched_image_faiss_id),))
        result = cursor.fetchone()
        if result:
            group_id = result
            cursor.execute("INSERT OR REPLACE INTO images (image_path, group_id, faiss_image_id) VALUES (?,?,?)",
                           (image_path, group_id, None))
            db_conn.commit()
            print(f"  > ACTION: Added to existing visual group {group_id}.")
        return

    # --- STAGE 3: Fast Caption & Semantic De-duplication ---
    print(f"  > STATUS: Visually Unique (Score: {visual_score:.4f}). Running Fast Caption...")
    fast_caption = get_fast_caption(image)
    caption_embedding = get_text_embedding(fast_caption)
    
    if faiss_caption_index.ntotal > 0:
        scores, faiss_ids = faiss_caption_index.search(caption_embedding, k=1)
        semantic_score, matched_caption_faiss_id = scores, faiss_ids
    else:
        semantic_score = 0.0

    if semantic_score > SEMANTIC_SIMILARITY_THRESHOLD:
        print(f"  > STATUS: Semantic Duplicate Found (Score: {semantic_score:.4f})")
        print(f"  > Fast Caption: '{fast_caption}'")
        cursor.execute("SELECT group_id FROM images WHERE faiss_image_id =?", (int(matched_caption_faiss_id),))
        result = cursor.fetchone()
        if result:
            group_id = result
            cursor.execute("INSERT OR REPLACE INTO images (image_path, group_id, faiss_image_id) VALUES (?,?,?)",
                           (image_path, group_id, None))
            db_conn.commit()
            print(f"  > ACTION: Added to existing semantic group {group_id}.")
        return

    # --- STAGE 4: New Unique Image ---
    print(f"  > STATUS: Semantically Unique (Score: {semantic_score:.4f}).")
    
    new_group_id = f"group_{faiss_image_index.ntotal}"
    new_faiss_image_id = faiss_image_index.ntotal
    
    faiss_image_index.add(image_embedding)
    faiss_caption_index.add(caption_embedding)
    
    cursor.execute("INSERT OR REPLACE INTO images (image_path, group_id, faiss_image_id) VALUES (?,?,?)",
                   (image_path, new_group_id, int(new_faiss_image_id)))
                   
    cursor.execute(
        """
        INSERT OR REPLACE INTO image_groups 
        (group_id, canonical_path, fast_caption, processing_status, taken_at_timestamp, gps_latitude, gps_longitude) 
        VALUES (?,?,?,?,?,?,?)
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
    
    cursor.execute("INSERT OR REPLACE INTO processing_queue (group_id) VALUES (?)", (new_group_id,))
    db_conn.commit()
    print(f"  > ACTION: Created new Group {new_group_id}. Queued for deep processing.")

# --- 7. BATCH EXECUTION ---
if __name__ == "__main__":
    import glob
    my_photo_directory = "./my_photos" # Example directory
    os.makedirs(my_photo_directory, exist_ok=True)
    print(f"Looking for images in {os.path.abspath(my_photo_directory)}")
    
    # Add all.jpg and.png files
    image_paths = glob.glob(f"{my_photo_directory}/*.jpg") + glob.glob(f"{my_photo_directory}/*.png")
    
    for path in image_paths:
        ingest_new_image(path)
        
    faiss.write_index(faiss_image_index, "photo_library.index")
    faiss.write_index(faiss_caption_index, "caption_library.index")
    print("\n--- Ingest complete. All indexes saved. ---")
    
    db_conn.close()
```

## 4. Script 2: `process_queue.py` (Deep Processing - 24GB VRAM)

This is the upgraded script. It runs as a background task. It loads the larger, higher-quality models onto the 24GB GPU to process the queue.

Models Loaded:

- `Qwen/Qwen3-VL-8B-Thinking` (8B params, bfloat16)
- `google/siglip2-giant-opt-patch16-384` (2B params, for faces/text)
- `PekingU/rtdetr_r101vd` (SOTA 56.2 AP)
- `vblagoje/bert-english-uncased-finetuned-pos` (Lightweight tagger)

Est. VRAM: ~24-25GB (Managed by device_map="auto" by offloading to CPU RAM if needed).

```python
import torch
import faiss
import numpy as np
import sqlite3
import json
from PIL import Image
from transformers import (
    AutoImageProcessor, 
    AutoModel, 
    AutoModelForObjectDetection,
    AutoModelForCausalLM,
    AutoProcessor,
    pipeline
)
from transformers.image_utils import load_image

# --- 1. CONFIGURATION ---
FACE_SIMILARITY_THRESHOLD = 0.92

# --- 2. SOTA MODEL CHECKPOINTS (UPGRADED for 24GB) ---
# 1. Embedding Model (UPGRADED): Use 2B param "Giant" model 
EMBEDDING_MODEL_ID = "google/siglip2-giant-opt-patch16-384"
EMBEDDING_DIM = 1152 #

# 2. Object Detection (SOTA)
DETECTION_MODEL_ID = "PekingU/rtdetr_r101vd"

# 3. Deep Captioning (UPGRADED): Use full bfloat16 8B model
CAPTION_MODEL_ID = "Qwen/Qwen3-VL-8B-Thinking" 

# 4. Tag Extraction (Lightweight)
TAGGER_MODEL_ID = "vblagoje/bert-english-uncased-finetuned-pos"

# --- 3. HARDWARE SETUP ---
def get_inference_device_and_dtype():
    if torch.cuda.is_available():
        print("Using CUDA (NVIDIA GPU). Setting DTYPE to bfloat16 for quality.")
        # bfloat16 is preferred for new models like Qwen3 [22]
        return torch.device("cuda"), torch.bfloat16
    if torch.backends.mps.is_available(): #
        print("Using MPS (Apple Silicon GPU). Setting DTYPE to float16.")
        return torch.device("mps"), torch.float16
    print("No GPU detected. Using CPU (float32).")
    return torch.device("cpu"), torch.float32

DEVICE, DTYPE = get_inference_device_and_dtype()

# --- 4. LOAD AI MODELS (Heavy/Upgraded Set) ---
print(f"Loading Model 1: Embedding (UPGRADED: {EMBEDDING_MODEL_ID})")
embed_processor = AutoProcessor.from_pretrained(EMBEDDING_MODEL_ID)
embed_model = AutoModel.from_pretrained(
    EMBEDDING_MODEL_ID,
    torch_dtype=DTYPE,
    device_map="auto" #
).eval()

print(f"Loading Model 2: Object Detection ({DETECTION_MODEL_ID})")
detect_processor = AutoImageProcessor.from_pretrained(DETECTION_MODEL_ID)
detect_model = AutoModelForObjectDetection.from_pretrained(
    DETECTION_MODEL_ID,
    torch_dtype=DTYPE,
    device_map="auto"
).eval()

print(f"Loading Model 3: Deep Captioning (UPGRADED: {CAPTION_MODEL_ID})")
caption_processor = AutoProcessor.from_pretrained(CAPTION_MODEL_ID, trust_remote_code=True)
caption_model = AutoModelForCausalLM.from_pretrained(
    CAPTION_MODEL_ID,
    torch_dtype=DTYPE,
    device_map="auto",
    trust_remote_code=True
).eval()

print(f"Loading Model 4: Tag Extraction ({TAGGER_MODEL_ID})")
pos_tagger = pipeline(
    "token-classification", 
    model=TAGGER_MODEL_ID, 
    device=DEVICE
)

# --- 5. LOAD DATABASES ---
print("Connecting to databases and FAISS...")
db_conn = sqlite3.connect("photo_metadata.db")
cursor = db_conn.cursor()

try:
    faiss_face_index = faiss.read_index("face_library.index")
except:
    faiss_face_index = faiss.IndexFlatIP(EMBEDDING_DIM)

# --- 6. CORE AI FUNCTIONS ---
@torch.no_grad()
def get_embedding(image, model, processor):
    """Uses the UPGRADED SigLIP-Giant model"""
    inputs = processor(images=image, return_tensors="pt").to(DEVICE, DTYPE)
    image_features = model.get_image_features(**inputs["pixel_values"]) #
    vector = image_features.cpu().numpy().astype(np.float32)
    faiss.normalize_L2(vector)
    return vector

@torch.no_grad()
def get_or_create_person_group(person_embedding):
    """Uses the UPGRADED SigLIP-Giant embeddings for faces"""
    global faiss_face_index
    if faiss_face_index.ntotal > 0:
        scores, faiss_ids = faiss_face_index.search(person_embedding, k=1)
        if scores > FACE_SIMILARITY_THRESHOLD:
            cursor.execute("SELECT person_group_id FROM person_groups WHERE faiss_face_id =?", (int(faiss_ids),))
            result = cursor.fetchone()
            if result: return result

    new_faiss_id = faiss_face_index.ntotal
    faiss_face_index.add(person_embedding)
    new_person_group_id = f"person_{new_faiss_id}"
    default_name = f"Person {new_faiss_id + 1}" 
    cursor.execute("INSERT INTO person_groups (person_group_id, name, faiss_face_id) VALUES (?,?,?)",
                   (new_person_group_id, default_name, int(new_faiss_id)))
    db_conn.commit()
    return new_person_group_id

@torch.no_grad()
def run_deep_processing_pipeline(image):
    """Runs the full SOTA analysis suite"""
    ai_results = {}

    # --- 6.1: Deep Captioning (UPGRADED Qwen3-VL) ---
    prompt_messages =}
    ]
    inputs = caption_processor.apply_chat_template(prompt_messages, images=[image], return_tensors="pt").to(DEVICE)
    generated_ids = caption_model.generate(inputs, max_new_tokens=100)
    caption = caption_processor.batch_decode(generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False).strip()
    ai_results["generated_caption"] = caption
    print(f"  > Deep Caption: '{caption}'")

    # --- 6.2: Tag Extraction (from new caption) ---
    tags_list =
    try:
        pos_results = pos_tagger(caption) #
        for entity in pos_results:
            if entity['entity_group'] in: # Common Noun/Proper Noun tags
                tag = entity['word'].replace("##", "").strip()
                if tag: tags_list.append(tag)
        tags_list = list(set(tags_list))
    except Exception as e:
        print(f"  > Tagger failed: {e}")
    ai_results["extracted_tags_json"] = json.dumps(tags_list)
    print(f"  > Extracted Tags: {tags_list}")

    # --- 6.3: Object & Face Detection ---
    detect_inputs = detect_processor(images=image, return_tensors="pt").to(DEVICE)
    outputs = detect_model(**detect_inputs)
    target_sizes = torch.tensor([image.size[::-1]], device=DEVICE)
    detections = detect_processor.post_process_object_detection(outputs, threshold=0.9, target_sizes=target_sizes) #
    
    detected_objects_list =
    for score, label_id, box in zip(detections["scores"], detections["labels"], detections["boxes"]):
        label = detect_model.config.id2label[label_id.item()]
        detection_data = {"label": label, "score": score.item(), "box": box.tolist()}
        if label == "person":
            try:
                box_int = [int(coord) for coord in box.tolist()]
                person_crop = image.crop((box_int, box_int[1], box_int[2], box_int[3]))
                person_embedding = get_embedding(person_crop, embed_model, embed_processor) # Use upgraded embedder
                detection_data["person_group_id"] = get_or_create_person_group(person_embedding)
            except Exception as e:
                print(f"  > Face embedding failed: {e}")
        detected_objects_list.append(detection_data)
        
    ai_results["detected_objects_json"] = json.dumps(detected_objects_list)
    print(f"  > Detected {len(detected_objects_list)} objects (with face grouping).")
    return ai_results

# --- 7. MAIN QUEUE PROCESSING LOOP ---
if __name__ == "__main__":
    print("--- Starting Deep Processing Queue (24GB VRAM) ---")
    
    cursor.execute("SELECT group_id FROM processing_queue")
    queue = cursor.fetchall()
    print(f"Found {len(queue)} images to process.")
    
    for (group_id,) in queue:
        print(f"\nProcessing Group: {group_id}")
        cursor.execute("SELECT canonical_path FROM image_groups WHERE group_id =?", (group_id,))
        path_result = cursor.fetchone()
        
        if not path_result:
            print(f"  > ERROR: No canonical path found for {group_id}. Skipping.")
            continue
            
        try:
            image_path = path_result
            image = load_image(image_path).convert("RGB")
            
            ai_results = run_deep_processing_pipeline(image)
            
            cursor.execute(
                """
                UPDATE image_groups 
                SET generated_caption =?, 
                    detected_objects_json =?, 
                    extracted_tags_json =?,
                    processing_status = 'COMPLETED'
                WHERE group_id =?
                """,
                (
                    ai_results["generated_caption"],
                    ai_results["detected_objects_json"],
                    ai_results["extracted_tags_json"],
                    group_id
                )
            )
            
            cursor.execute("DELETE FROM processing_queue WHERE group_id =?", (group_id,))
            db_conn.commit()
            print(f"  > SUCCESS: Group {group_id} processed and updated.")
            
        except Exception as e:
            print(f"  > FAILED processing for {group_id}: {e}")
            cursor.execute("UPDATE image_groups SET processing_status = 'FAILED' WHERE group_id =?", (group_id,))
            db_conn.commit()

    faiss.write_index(faiss_face_index, "face_library.index")
    print("\n--- Deep processing complete. Face index saved. ---")
    db_conn.close()
```

## 5. Script 3: maintenance.py (Semantic Merge)

This script is for Solution 2. It is upgraded to use the new `google/siglip2-giant-opt-patch16-384` model for embedding the high-quality captions,
ensuring the most accurate semantic comparison.

```python
# File: run_maintenance.py
import torch
import faiss
import numpy as np
import sqlite3
from transformers import AutoProcessor, AutoModel

# --- 1. Load Required Components ---
# UPGRADED: Use the same SOTA embedder as the deep processing script
EMBEDDING_MODEL_ID = "google/siglip2-giant-opt-patch16-384"
EMBEDDING_DIM = 1152 #
SEMANTIC_SIMILARITY_THRESHOLD = 0.95 
DEVICE, DTYPE = (torch.device("cuda"), torch.bfloat16) if torch.cuda.is_available() else (torch.device("mps"), torch.float16) if torch.backends.mps.is_available() else (torch.device("cpu"), torch.float32)

print("Loading Embedding Model (for text)...")
embed_processor = AutoProcessor.from_pretrained(EMBEDDING_MODEL_ID)
embed_model = AutoModel.from_pretrained(
    EMBEDDING_MODEL_ID,
    torch_dtype=DTYPE,
    device_map="auto"
).eval()

db_conn = sqlite3.connect("photo_metadata.db")
cursor = db_conn.cursor()

@torch.no_grad()
def get_text_embedding(text_list):
    inputs = embed_processor(text=text_list, return_tensors="pt", padding=True, truncation=True).to(DEVICE)
    text_features = embed_model.get_text_features(**inputs)
    vector = text_features.cpu().numpy().astype(np.float32)
    faiss.normalize_L2(vector)
    return vector

def merge_groups(primary_group_id, secondary_group_id):
    print(f"  > MERGING: {secondary_group_id} -> {primary_group_id}")
    try:
        cursor.execute("UPDATE images SET group_id =? WHERE group_id =?", (primary_group_id, secondary_group_id))
        cursor.execute("DELETE FROM image_groups WHERE group_id =?", (secondary_group_id,))
        db_conn.commit()
    except Exception as e:
        print(f"  > MERGE FAILED: {e}")
        db_conn.rollback()

def run_semantic_merge_task():
    print("\n--- Starting Semantic Merge Maintenance Task ---")
    
    # Merge based on the high-quality 'generated_caption'
    cursor.execute("SELECT group_id, generated_caption FROM image_groups WHERE processing_status = 'COMPLETED' AND generated_caption IS NOT NULL")
    rows = cursor.fetchall()
    if len(rows) < 2:
        print("Not enough processed groups to merge. Exiting.")
        return

    group_ids = [row for row in rows]
    captions = [row[1] for row in rows]
    
    print(f"Creating text embeddings for {len(captions)} deep captions...")
    caption_embeddings = get_text_embedding(captions)
    
    caption_index = faiss.IndexFlatIP(EMBEDDING_DIM)
    caption_index.add(caption_embeddings)
    
    print("Searching for semantic duplicates...")
    k = 5 
    all_scores, all_indices = caption_index.search(caption_embeddings, k=k)
    
    merged_groups = set()
    total_merges = 0
    
    for i in range(len(all_indices)):
        primary_group_id = group_ids[i]
        if primary_group_id in merged_groups: continue
            
        for j in range(1, k): # Start at 1 (skip self)
            neighbor_index = all_indices[i][j]
            neighbor_score = all_scores[i][j]
            secondary_group_id = group_ids[neighbor_index]
            
            if secondary_group_id in merged_groups or primary_group_id == secondary_group_id:
                continue

            if neighbor_score > SEMANTIC_SIMILARITY_THRESHOLD:
                merge_groups(primary_group_id, secondary_group_id)
                merged_groups.add(secondary_group_id)
                total_merges += 1

    print(f"--- Semantic Merge Complete. Total merges: {total_merges} ---")

if __name__ == "__main__":
    run_semantic_merge_task()
    db_conn.close()
```

## 6. Script 4: search.py (Application UI Backend)

This script is also upgraded to use the `google/siglip2-giant-opt-patch16-384` model.
This is critical for search accuracy, as the text query must be embedded with the same model that created the image and face embeddings.

```python
# File: search.py
import torch
import faiss
import numpy as np
import sqlite3
import json
from transformers import AutoProcessor, AutoModel

# --- 1. Load Required Components ---
# UPGRADED: Use the same SOTA embedder as the deep processing script
EMBEDDING_MODEL_ID = "google/siglip2-giant-opt-patch16-384"
EMBEDDING_DIM = 1152 #

DEVICE, DTYPE = (torch.device("cuda"), torch.bfloat16) if torch.cuda.is_available() else (torch.device("mps"), torch.float16) if torch.backends.mps.is_available() else (torch.device("cpu"), torch.float32)

print("Loading Embedding Model (for search)...")
embed_processor = AutoProcessor.from_pretrained(EMBEDDING_MODEL_ID)
embed_model = AutoModel.from_pretrained(
    EMBEDDING_MODEL_ID,
    torch_dtype=DTYPE,
    device_map="auto"
).eval()

db_conn = sqlite3.connect("photo_metadata.db")
cursor = db_conn.cursor()
faiss_image_index = faiss.read_index("photo_library.index")

# --- 2. Search Function ---
@torch.no_grad()
def get_text_embedding(text_list):
    """Generates SigLIP embeddings for a batch of text queries."""
    inputs = embed_processor(text=text_list, return_tensors="pt", padding=True, truncation=True).to(DEVICE)
    text_features = embed_model.get_text_features(**inputs)
    vector = text_features.cpu().numpy().astype(np.float32)
    faiss.normalize_L2(vector)
    return vector

@torch.no_grad()
def search_photo_library(text_query, top_k=5):
    """
    Performs a semantic search of the photo library using a text query.
    """
    print(f"\n--- Searching for: '{text_query}' ---")
    
    # 1. Create Text Embedding for the query
    vector = get_text_embedding([text_query])
    
    # 2. Search the IMAGE index
    scores, faiss_ids = faiss_image_index.search(vector, k=top_k)
    
    # 3. Retrieve Full Results from SQLite DB
    print(f"--- Top {top_k} Results ---")
    faiss_id_list = [int(i) for i in faiss_ids]
    scores_list = scores

    # Map FAISS ID to full group data
    results =
    for i, faiss_id in enumerate(faiss_id_list):
        cursor.execute(
            """
            SELECT 
                i.image_path, i.group_id, 
                g.generated_caption, g.extracted_tags_json,
                g.taken_at_timestamp, g.gps_latitude, g.gps_longitude
            FROM images i
            JOIN image_groups g ON i.group_id = g.group_id
            WHERE i.faiss_image_id =?
            """,
            (faiss_id,)
        )
        row = cursor.fetchone()
        if row:
            image_path, group_id, caption, tags, timestamp, lat, lon = row
            print(f"Rank {i+1} (Similarity: {scores_list[i]:.4f})")
            print(f"  Path: {image_path}")
            print(f"  Group: {group_id}")
            print(f"  Timestamp: {timestamp}")
            print(f"  GPS: ({lat}, {lon})")
            print(f"  Deep Caption: {caption}")
            print(f"  Tags: {json.loads(tags)}\n")

# --- Example Execution ---
if __name__ == "__main__":
    search_photo_library(text_query="a picture of a cat")
    search_photo_library(text_query="a bear in a field")
    db_conn.close()
```
