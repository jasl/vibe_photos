# AI Photo Management: Implementation Plan (v3.3 - Cascade + SOTA Fixes)

This document outlines the final, corrected technical workflow. It fixes the critical similarity bug from v3.2, adds specialist face recognition, and implements debugging support.

## 1. Installation and Dependencies

This `requirements.txt` file sets up the environment.

```bash
# Core AI and ML libraries
pip install transformers torch pillow requests accelerate
# Vector database (maintained version)
pip install faiss-cpu
# Standard utilities
pip install numpy sqlite3
# (NEW) For specialist face detection
pip install opencv-python-headless
# (NEW) For lightweight tag extraction
pip install spacy
# (RUN THIS ONCE) Download the spacy model
python -m spacy download en_core_web_sm
```

## 2. Database Schema (`setup_db.py`)

Run this script once. It is updated with new tables and columns for face recognition and debugging.

```python
import sqlite3

db_conn = sqlite3.connect("photo_metadata.db")
cursor = db_conn.cursor()

# Table 1: Stores every single image file (UPDATED)
cursor.execute("""
CREATE TABLE IF NOT EXISTS images (
    image_path TEXT PRIMARY KEY,
    group_id TEXT,
    faiss_image_id INTEGER,
    -- (NEW) For debugging similarity
    duplicate_of_path TEXT,
    duplicate_score REAL
)
""")

# Table 2: Stores AI results for each *unique* group
cursor.execute("""
CREATE TABLE IF NOT EXISTS image_groups (
    group_id TEXT PRIMARY KEY,
    canonical_path TEXT,
    fast_caption TEXT,
    generated_caption TEXT,
    detected_objects_json TEXT,
    extracted_tags_json TEXT,
    processing_status TEXT DEFAULT 'QUEUED',
    taken_at_timestamp TEXT,
    gps_latitude REAL,
    gps_longitude REAL
)
""")

# Table 3: Stores each unique person (UPDATED)
cursor.execute("""
CREATE TABLE IF NOT EXISTS person_groups (
    person_group_id TEXT PRIMARY KEY,
    name TEXT, -- User-editable name (e.g., "Mom", "John")
    representative_faiss_id INTEGER -- The first FAISS ID added for this person
)
""")

# Table 4: Queue for deep processing
cursor.execute("""
CREATE TABLE IF NOT EXISTS processing_queue (
    group_id TEXT PRIMARY KEY,
    FOREIGN KEY (group_id) REFERENCES image_groups (group_id)
)
""")

# Table 5: Links faces to the photos they appear in (NEW)
cursor.execute("""
CREATE TABLE IF NOT EXISTS face_appearances (
    appearance_id INTEGER PRIMARY KEY AUTOINCREMENT,
    image_path TEXT,
    person_group_id TEXT,
    box_json TEXT, -- Bounding box of the face
    FOREIGN KEY (image_path) REFERENCES images (image_path),
    FOREIGN KEY (person_group_id) REFERENCES person_groups (person_group_id)
)
""")
db_conn.commit()
db_conn.close()
print("Database v3.3 (Cascade + Face-Fix) schema is ready.")
```

## 3. Script 1: `ingest.py` (Fast Ingest & Filtering)

This script is your primary tool for adding photos. It is UPDATED to use the correct SOTA embedding model (SigLIP-Giant)  and to store debugging evidence.
Models Loaded:

- google/siglip2-giant-opt-patch16-384 (2B params)
- Salesforce/blip2-opt-2.7b (2.7B params)

Est. VRAM: ~15.2GB (Fits within 24GB).

```python
import torch
import faiss
import numpy as np
import sqlite3
import json
import os
import sys
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

# --- CRITICAL FIX (v3.3) ---
# Use the *best* embedding model for ALL embedding tasks.
EMBEDDING_MODEL_ID = "google/siglip2-giant-opt-patch16-384"
EMBEDDING_DIM = 1152 #
# Use the FAST caption model for filtering
FAST_CAPTION_MODEL_ID = "Salesforce/blip2-opt-2.7b"

# --- 2. HARDWARE SETUP ---
def get_inference_device_and_dtype():
    if torch.cuda.is_available():
        print("Using CUDA (NVIDIA GPU). Setting DTYPE to bfloat16 for quality.")
        return torch.device("cuda"), torch.bfloat16
    if torch.backends.mps.is_available(): #
        print("Using MPS (Apple Silicon GPU). Setting DTYPE to float16.")
        return torch.device("mps"), torch.float16
    print("No GPU detected. Using CPU (float32).")
    return torch.device("cpu"), torch.float32

DEVICE, DTYPE = get_inference_device_and_dtype()
# We explicitly set DTYPE to float16/bfloat16 to fit models in VRAM [18, 19, 20, 21, 22]

# --- 3. LOAD MODELS (Fast Set, Upgraded Embedder) ---
print(f"Loading Model 1: Embedding (UPGRADED: {EMBEDDING_MODEL_ID})")
embed_processor = AutoProcessor.from_pretrained(EMBEDDING_MODEL_ID)
embed_model = AutoModel.from_pretrained(
    EMBEDDING_MODEL_ID,
    torch_dtype=DTYPE,
    device_map="auto" # [18, 4, 8, 23]
).eval()

print(f"Loading Model 2: Fast Captioner ({FAST_CAPTION_MODEL_ID})")
fast_caption_processor = Blip2Processor.from_pretrained(FAST_CAPTION_MODEL_ID)
fast_caption_model = Blip2ForConditionalGeneration.from_pretrained(
    FAST_CAPTION_MODEL_ID,
    torch_dtype=DTYPE,
    device_map="auto" # [3]
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
        ctime = os.path.getctime(image_path)
        timestamp = datetime.fromtimestamp(ctime).isoformat()

        exif_data = image_obj._getexif()
        if exif_data:
            dt_original = exif_data.get(36867) # 36867 = DateTimeOriginal
            if dt_original:
                dt_obj = datetime.strptime(dt_original, '%Y:%m:%d %H:%M:%S')
                timestamp = dt_obj.isoformat()

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
    """(UPDATED) Uses the new SigLIP-Giant model"""
    inputs = embed_processor(images=image, return_tensors="pt").to(DEVICE, DTYPE)
    image_features = embed_model.get_image_features(**inputs["pixel_values"]) # [24, 5, 25]
    vector = image_features.cpu().numpy().astype(np.float32)
    faiss.normalize_L2(vector) # [26]
    return vector

@torch.no_grad()
def get_text_embedding(text):
    """(UPDATED) Uses the new SigLIP-Giant model"""
    inputs = embed_processor(text=[text], return_tensors="pt", padding=True, truncation=True).to(DEVICE)
    text_features = embed_model.get_text_features(**inputs)
    vector = text_features.cpu().numpy().astype(np.float32)
    faiss.normalize_L2(vector)
    return vector

@torch.no_grad()
def get_fast_caption(image):
    """Uses the BLIP-2 model [27, 28, 29]"""
    inputs = fast_caption_processor(image, return_tensors="pt").to(DEVICE, DTYPE)
    generated_ids = fast_caption_model.generate(**inputs, max_new_tokens=50) # [30]
    caption = fast_caption_processor.batch_decode(generated_ids, skip_special_tokens=True).strip() # [30]
    return caption

# --- 6. MAIN INGEST WORKFLOW (UPDATED) ---
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
    matched_canonical_path = None
    if faiss_image_index.ntotal > 0:
        scores, faiss_ids = faiss_image_index.search(image_embedding, k=1)
        visual_score = scores
        matched_image_faiss_id = faiss_ids
    else:
        visual_score = 0.0

    if visual_score > VISUAL_SIMILARITY_THRESHOLD: #
        print(f"  > STATUS: Visual Duplicate Found (Score: {visual_score:.4f})")
        cursor.execute("SELECT group_id, (SELECT canonical_path FROM image_groups g WHERE g.group_id = i.group_id) FROM images i WHERE faiss_image_id =?", (int(matched_image_faiss_id),))
        result = cursor.fetchone()
        if result:
            group_id, matched_canonical_path = result
            # (NEW) Store debug evidence
            cursor.execute("INSERT OR REPLACE INTO images (image_path, group_id, faiss_image_id, duplicate_of_path, duplicate_score) VALUES (?,?,?,?,?)",
                           (image_path, group_id, None, matched_canonical_path, float(visual_score)))
            db_conn.commit()
            print(f"  > ACTION: Added to existing visual group {group_id}. Storing debug evidence.")
        return

    # --- STAGE 3: Fast Caption & Semantic De-duplication ---
    print(f"  > STATUS: Visually Unique (Score: {visual_score:.4f}). Running Fast Caption...")
    fast_caption = get_fast_caption(image)
    caption_embedding = get_text_embedding(fast_caption)
    
    if faiss_caption_index.ntotal > 0:
        scores, faiss_ids = faiss_caption_index.search(caption_embedding, k=1)
        semantic_score = scores
        matched_caption_faiss_id = faiss_ids
    else:
        semantic_score = 0.0

    if semantic_score > SEMANTIC_SIMILARITY_THRESHOLD: #
        print(f"  > STATUS: Semantic Duplicate Found (Score: {semantic_score:.4f})")
        print(f"  > Fast Caption: '{fast_caption}'")
        cursor.execute("SELECT group_id, (SELECT canonical_path FROM image_groups g WHERE g.group_id = i.group_id) FROM images i WHERE faiss_image_id =?", (int(matched_caption_faiss_id),))
        result = cursor.fetchone()
        if result:
            group_id, matched_canonical_path = result
            # (NEW) Store debug evidence
            cursor.execute("INSERT OR REPLACE INTO images (image_path, group_id, faiss_image_id, duplicate_of_path, duplicate_score) VALUES (?,?,?,?,?)",
                           (image_path, group_id, None, matched_canonical_path, float(semantic_score)))
            db_conn.commit()
            print(f"  > ACTION: Added to existing semantic group {group_id}. Storing debug evidence.")
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
    
    if len(sys.argv) < 2:
        print("Usage: python ingest.py /path/to/your/photos")
        sys.exit(1)
        
    my_photo_directory = sys.argv[1]
    if not os.path.isdir(my_photo_directory):
        print(f"Error: Directory not found: {my_photo_directory}")
        sys.exit(1)
        
    print(f"Starting ingest process for: {os.path.abspath(my_photo_directory)}")
    
    # Recursively find common image types
    image_paths = glob.glob(f"{my_photo_directory}/**/*.jpg", recursive=True) + \
                  glob.glob(f"{my_photo_directory}/**/*.png", recursive=True) + \
                  glob.glob(f"{my_photo_directory}/**/*.jpeg", recursive=True)
    
    print(f"Found {len(image_paths)} images to process.")
    
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
- `spacy.load("en_core_web_sm")` (Lightweight tagger)
- `cv2.CascadeClassifier` (Lightweight face detector)

Est. VRAM: ~24-25GB (Managed by device_map="auto").

```python
import torch
import faiss
import numpy as np
import sqlite3
import json
import cv2 # NEW: For face detection
import spacy # NEW: For POS tagging
from PIL import Image
from transformers import (
    AutoImageProcessor, 
    AutoModel, 
    AutoModelForObjectDetection,
    AutoModelForCausalLM,
    AutoProcessor
)
from transformers.image_utils import load_image
import os

# --- 1. CONFIGURATION ---
FACE_SIMILARITY_THRESHOLD = 0.92

# --- 2. SOTA MODEL CHECKPOINTS (UPGRADED for 24GB) ---
EMBEDDING_MODEL_ID = "google/siglip2-giant-opt-patch16-384" # 
EMBEDDING_DIM = 1152 #
DETECTION_MODEL_ID = "PekingU/rtdetr_r101vd"
CAPTION_MODEL_ID = "Qwen/Qwen3-VL-8B-Thinking" 

# --- 3. HARDWARE SETUP ---
def get_inference_device_and_dtype():
    if torch.cuda.is_available():
        print("Using CUDA (NVIDIA GPU). Setting DTYPE to bfloat16 for quality.")
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

print(f"Loading Model 4: Tag Extraction (spaCy)")
# Use lightweight, fast spaCy for POS tagging
nlp = spacy.load("en_core_web_sm")

print(f"Loading Model 5: Face Detector (OpenCV)")
face_cascade_path = "haarcascade_frontalface_default.xml"
if not os.path.exists(face_cascade_path):
    print("Downloading face cascade...")
    url = "https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/haarcascade_frontalface_default.xml"
    r = requests.get(url, allow_redirects=True)
    open(face_cascade_path, 'wb').write(r.content)
face_cascade = cv2.CascadeClassifier(face_cascade_path)
print("All models loaded.")

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
    """(UPDATED) Uses the new SigLIP-Giant model"""
    inputs = processor(images=image, return_tensors="pt").to(DEVICE, DTYPE)
    image_features = model.get_image_features(**inputs["pixel_values"]) #
    vector = image_features.cpu().numpy().astype(np.float32)
    faiss.normalize_L2(vector)
    return vector

@torch.no_grad()
def get_or_create_person_group(person_embedding):
    """(UPDATED) Uses the SigLIP-Giant embeddings for faces"""
    global faiss_face_index
    if faiss_face_index.ntotal > 0:
        scores, faiss_ids = faiss_face_index.search(person_embedding, k=1)
        if scores > FACE_SIMILARITY_THRESHOLD:
            cursor.execute("SELECT person_group_id FROM person_groups WHERE faiss_face_id =?", (int(faiss_ids),))
            result = cursor.fetchone()
            if result: return result # Return just the ID

    # MISS: This is a new person
    new_faiss_id = faiss_face_index.ntotal
    faiss_face_index.add(person_embedding)
    new_person_group_id = f"person_{new_faiss_id}"
    default_name = f"Person {new_faiss_id + 1}" 
    cursor.execute("INSERT INTO person_groups (person_group_id, name, representative_faiss_id) VALUES (?,?,?)",
                   (new_person_group_id, default_name, int(new_faiss_id)))
    db_conn.commit()
    return new_person_group_id

@torch.no_grad()
def run_deep_processing_pipeline(image_path, pil_image):
    """
    (UPDATED) Runs the full SOTA analysis suite on a single unique image.
    """
    ai_results = {}
    
    # Convert PIL Image to OpenCV format for detectors
    cv_image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
    gray_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)

    # --- 6.1: Deep Captioning (UPGRADED Qwen3-VL) ---
    prompt_messages =}]
    inputs = caption_processor.apply_chat_template(prompt_messages, images=[pil_image], return_tensors="pt").to(DEVICE)
    generated_ids = caption_model.generate(inputs, max_new_tokens=100)
    caption = caption_processor.batch_decode(generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False).strip()
    ai_results["generated_caption"] = caption
    print(f"  > Deep Caption: '{caption}'")

    # --- 6.2: Tag Extraction (spaCy) ---
    tags_list =
    try:
        doc = nlp(caption)
        # Extract Nouns and Proper Nouns
        for token in doc:
            if token.pos_ in ("NOUN", "PROPN"):
                tags_list.append(token.lemma_.lower()) # Use lemma for root form
        tags_list = list(set(tags_list)) # De-duplicate
    except Exception as e:
        print(f"  > Tagger failed: {e}")
    ai_results["extracted_tags_json"] = json.dumps(tags_list)
    print(f"  > Extracted Tags: {tags_list}")

    # --- 6.3: Object Detection (RT-DETR) ---
    detect_inputs = detect_processor(images=pil_image, return_tensors="pt").to(DEVICE)
    outputs = detect_model(**detect_inputs)
    target_sizes = torch.tensor([pil_image.size[::-1]], device=DEVICE)
    detections = detect_processor.post_process_object_detection(outputs, threshold=0.9, target_sizes=target_sizes) #
    
    detected_objects_list =
    for score, label_id, box in zip(detections["scores"], detections["labels"], detections["boxes"]):
        detection_data = {
            "label": detect_model.config.id2label[label_id.item()], 
            "score": score.item(), 
            "box": box.tolist()
        }
        detected_objects_list.append(detection_data)
        
    ai_results["detected_objects_json"] = json.dumps(detected_objects_list)
    print(f"  > Detected {len(detected_objects_list)} objects.")

    # --- 6.4: Face Detection & Recognition (NEW Specialist) ---
    # Find faces using OpenCV
    faces = face_cascade.detectMultiScale(gray_image, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    print(f"  > (OpenCV) Found {len(faces)} faces.")
    
    for (x, y, w, h) in faces:
        try:
            # Crop the *exact* face from the PIL image
            face_crop = pil_image.crop((x, y, x+w, y+h))
            
            # Embed using the SOTA SigLIP-Giant model
            face_embedding = get_embedding(face_crop, embed_model, embed_processor)
            
            # Get or create the unique ID for this person
            person_group_id = get_or_create_person_group(face_embedding)
            
            # Store this appearance in the new table
            cursor.execute(
                "INSERT INTO face_appearances (image_path, person_group_id, box_json) VALUES (?,?,?)",
                (image_path, person_group_id, json.dumps([x,y,w,h]))
            )
            print(f"    > Linked face at [{x},{y}] to {person_group_id}")

        except Exception as e:
            print(f"  > Face embedding failed: {e}")
    
    db_conn.commit() # Commit face appearances
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
            pil_image = Image.open(image_path).convert("RGB")
            
            ai_results = run_deep_processing_pipeline(image_path, pil_image)
            
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

## 5. Script 3: `maintenance.py` (Semantic Merge)

This script is upgraded to use the new `google/siglip2-giant-opt-patch16-384` model, matching the other scripts.

```python
# File: run_maintenance.py
import torch
import faiss
import numpy as np
import sqlite3
from transformers import AutoProcessor, AutoModel

# --- 1. Load Required Components ---
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
    """(UPDATED) Uses the SigLIP-Giant model"""
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

            if neighbor_score > SEMANTIC_SIMILARITY_THRESHOLD: #
                merge_groups(primary_group_id, secondary_group_id)
                merged_groups.add(secondary_group_id)
                total_merges += 1

    print(f"--- Semantic Merge Complete. Total merges: {total_merges} ---")

if __name__ == "__main__":
    run_semantic_merge_task()
    db_conn.close()
```

## 6. Script 4: `search.py` (Application UI Backend)

This script is also upgraded to use the `google/siglip2-giant-opt-patch16-384` model, which is critical for fixing the similarity bug.

```python
# File: search.py
import torch
import faiss
import numpy as np
import sqlite3
import json
from transformers import AutoProcessor, AutoModel

# --- 1. Load Required Components ---
# (UPDATED) Use the same SOTA embedder as all other scripts
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
print("Search components loaded.")

# --- 2. Search Function ---
@torch.no_grad()
def get_text_embedding(text_list):
    """(UPDATED) Generates SigLIP-Giant embeddings for text queries."""
    inputs = embed_processor(text=text_list, return_tensors="pt", padding=True, truncation=True).to(DEVICE)
    text_features = embed_model.get_text_features(**inputs)
    vector = text_features.cpu().numpy().astype(np.float32)
    faiss.normalize_L2(vector)
    return vector

@torch.no_grad()
def search_photo_library(text_query, top_k=5):
    """
    (UPDATED)
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

    results =
    for i, faiss_id in enumerate(faiss_id_list):
        cursor.execute(
            """
            SELECT 
                i.image_path, i.group_id, 
                g.generated_caption, g.fast_caption, g.extracted_tags_json,
                g.taken_at_timestamp, g.gps_latitude, g.gps_longitude,
                g.processing_status
            FROM images i
            JOIN image_groups g ON i.group_id = g.group_id
            WHERE i.faiss_image_id =?
            """,
            (faiss_id,)
        )
        row = cursor.fetchone()
        if row:
            image_path, group_id, deep_caption, fast_caption, tags, timestamp, lat, lon, status = row
            
            # Use deep caption if available, otherwise fall back to fast caption
            final_caption = deep_caption if status == 'COMPLETED' else fast_caption
            
            result_data = {
                "rank": i + 1,
                "similarity": f"{scores_list[i]:.4f}",
                "path": image_path,
                "group_id": group_id,
                "timestamp": timestamp,
                "gps": f"({lat}, {lon})" if lat is not None else "None",
                "caption": final_caption,
                "tags": json.loads(tags) if tags else "",
                "status": status
            }
            results.append(result_data)
            
            # Print to console
            print(f"Rank {i+1} (Similarity: {result_data['similarity']})")
            print(f"  Path: {result_data['path']}")
            print(f"  Timestamp: {result_data['timestamp']}")
            print(f"  Caption: {result_data['caption']}")
            print(f"  Tags: {result_data['tags']}\n")
            
    return results

# --- Example Execution ---
if __name__ == "__main__":
    search_photo_library(text_query="a picture of a cat")
    search_photo_library(text_query="a burger on a plate")
    
    # Your "pizza in Italy" search
    # This will work if EXIF data (GPS) is processed and tags ("pizza") are extracted
    search_photo_library(text_query="a pizza in Italy") 
    
    db_conn.close()
```
