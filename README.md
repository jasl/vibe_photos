# AI Photo Management System

An intelligent photo management system powered by state-of-the-art AI models for automatic photo classification, duplicate detection, and semantic organization.

## Features

- **AI-Powered Classification**: Automatically generates descriptive captions for photos using BLIP2
- **Object Detection**: Identifies objects in photos with RT-DETR
- **Smart Duplicate Detection**: Uses SigLIP2 embeddings to find near-duplicate photos (burst shots, similar images)
- **Semantic Search**: Vector-based similarity search using FAISS
- **Web Interface**: Beautiful Flask-based UI for browsing and organizing photos
- **Efficient Processing**: Caches AI results for duplicate photos to save processing time

## Technology Stack

### AI Models (State-of-the-Art)

- **SigLIP2** (`google/siglip2-so400m-patch14-384`): Multi-modal embeddings for image similarity
- **RT-DETR** (`PekingU/rtdetr_r101vd`): Real-time object detection
- **BLIP2** (`Salesforce/blip2-opt-2.7b`): Generative image captioning

### Infrastructure

- **FAISS**: High-performance vector similarity search
- **SQLite**: Metadata storage
- **Flask**: Web interface
- **PyTorch**: Deep learning framework

## System Requirements

### Hardware

- **GPU**: NVIDIA GPU with 16GB+ VRAM (recommended)
  - Alternatively: Apple Silicon (MPS) or CPU (slower)
- **RAM**: 32GB+ recommended for large photo libraries
- **Storage**: ~500MB for models, ~130MB for FAISS index (30K photos)

### Software

- Python 3.10+
- CUDA toolkit (for NVIDIA GPU)
- `uv` package manager

## Installation

1. **Clone or navigate to the project directory**:
   ```bash
   cd /home/jasl/Workspace/ai_photos_management
   ```

2. **Install dependencies using `uv`**:
   ```bash
   uv sync
   ```

   This will install:
   - PyTorch with CUDA support
   - Transformers (Hugging Face)
   - FAISS-GPU
   - Flask
   - Other dependencies

3. **Verify installation**:
   ```bash
   uv run python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
   ```

## Usage

### Step 1: Index Your Photos

Process all photos in the configured directory (`/home/jasl/datasets/my_photos` by default):

```bash
uv run index_photos.py
```

**What happens during indexing:**

1. Scans the photos directory recursively for images
2. For each photo:
   - Creates a 1152-dimension embedding (fingerprint)
   - Checks for duplicates using FAISS similarity search
   - If duplicate (similarity > 0.98): Reuses cached AI results
   - If unique: Runs full AI pipeline (caption + object detection)
3. Saves results to SQLite database and FAISS index

**Expected Runtime:**

- **30,000 photos**: Several hours (depends on GPU)
- **Unique photos**: ~5-10 seconds per photo (full AI pipeline)
- **Duplicate photos**: <1 second per photo (cached results)

**Progress tracking:**

- Real-time progress bar with tqdm
- Saves every 100 images (safe to interrupt with Ctrl+C)
- Resume by re-running the script (skips already processed photos)
- Logs are stored in the `logs/` directory

### Step 2: Browse with Web UI

Launch the Flask web interface:

```bash
uv run web_app.py
```

Then open in your browser:
- **Gallery View**: http://localhost:5000/
- **Grouped View**: http://localhost:5000/groups

**Web UI Features:**

- **Gallery View**: Grid of all unique photo groups with captions
- **Grouped View**: Photos organized by similar captions
- **Detail View**: Click any photo to see:
  - Full resolution image
  - AI-generated caption
  - Detected objects with confidence scores
  - Similar/duplicate images in the group
  - File metadata

## Configuration

Edit `config.py` to customize:

```python
# Photo directory
PHOTOS_DIR = "/home/jasl/datasets/my_photos"

# Similarity threshold for duplicate detection (0.0-1.0)
SIMILARITY_THRESHOLD = 0.98  # Higher = more strict

# Object detection confidence threshold
DETECTION_THRESHOLD = 0.9

# Flask server settings
FLASK_HOST = "0.0.0.0"
FLASK_PORT = 5000
```

## Project Structure

```
ai_photos_management/
├── config.py              # Configuration constants
├── models.py              # AI model initialization
├── database.py            # SQLite and FAISS management
├── processing.py          # Image processing pipeline
├── index_photos.py        # Main indexing script
├── web_app.py            # Flask web interface
├── templates/            # HTML templates
│   ├── index.html        # Gallery view
│   ├── groups.html       # Grouped view
│   ├── objects.html      # Objects view
│   ├── object_detail.html # Object detail page
│   ├── group_detail.html # Detail view
│   └── search_results.html # Search results
├── static/
│   └── style.css         # Styling
├── data/                 # Generated data (gitignored)
│   ├── photo_library.index  # FAISS vector index
│   └── photo_metadata.db    # SQLite database
├── logs/                 # Application logs (gitignored)
├── LICENSE               # AGPL-3.0 license
└── README.md
```

## How It Works

### 1. Embedding Extraction (Every Photo)

- Converts each image into a 1152-dimension vector using SigLIP2
- Vectors are L2-normalized for cosine similarity comparison
- Fast operation (~0.5 seconds per image on GPU)

### 2. Duplicate Detection (Smart Caching)

- Searches FAISS index for similar images
- If similarity > 0.98: Marks as duplicate and reuses existing AI results
- If similarity ≤ 0.98: Treats as unique and runs full AI pipeline

### 3. AI Analysis (Unique Photos Only)

- **Caption Generation**: BLIP2 generates descriptive text (max 50 tokens)
- **Object Detection**: RT-DETR detects objects with bounding boxes
- Results are cached and reused for duplicate photos

### 4. Storage

- **SQLite**: Stores metadata, captions, and detected objects
  - `images` table: Maps each photo to its group
  - `image_groups` table: Stores AI results for each unique group
- **FAISS**: Stores normalized vectors for fast similarity search

## Database Schema

### `images` table

| Column      | Type    | Description                           |
|-------------|---------|---------------------------------------|
| image_path  | TEXT    | Full path to the image file (PRIMARY KEY) |
| group_id    | TEXT    | ID of the photo group                 |
| faiss_id    | INTEGER | Index in FAISS vector database        |

### `image_groups` table

| Column                  | Type | Description                          |
|-------------------------|------|--------------------------------------|
| group_id                | TEXT | Unique group ID (PRIMARY KEY)        |
| canonical_path          | TEXT | Path to the representative image     |
| generated_caption       | TEXT | AI-generated caption                 |
| detected_objects_json   | TEXT | JSON array of detected objects       |

## Performance Tips

1. **GPU Acceleration**: Ensure CUDA is properly installed for NVIDIA GPUs
2. **Batch Processing**: The system processes images one at a time to avoid OOM
3. **Duplicate Detection**: Saves significant time on photo libraries with many similar images
4. **Incremental Updates**: Re-running `index_photos.py` only processes new photos

## Troubleshooting

### Out of Memory (OOM) Errors

- Reduce batch size (already set to 1 for safety)
- Use smaller model variants (edit `config.py`)
- Close other applications using GPU memory

### Slow Processing

- Verify GPU is being used: Check console output during model loading
- Many unique photos = slower (expected, runs full AI pipeline)
- Many duplicates = faster (uses cached results)

### No Photos Showing in Web UI

- Ensure indexing completed successfully
- Check that `data/photo_metadata.db` exists
- Verify photos directory path in `config.py`

## License

This project is licensed under the **GNU Affero General Public License v3.0 (AGPL-3.0)**.

See the [LICENSE](LICENSE) file for full license text.

### Third-Party Models

This project uses the following open-source models:
- SigLIP2: Apache 2.0
- RT-DETR: Apache 2.0
- BLIP2: BSD 3-Clause

## Acknowledgments

- Google Research (SigLIP2)
- Peking University (RT-DETR)
- Salesforce Research (BLIP2)
- Facebook AI Research (FAISS)
