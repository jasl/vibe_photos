# AI Photo Management System

An intelligent photo management system powered by state-of-the-art AI models for automatic photo classification, duplicate detection, and semantic organization.

## Features (v3.3)

- **Cascade Architecture**: Two-tier processing for 5-10x faster ingestion
- **AI-Powered Captioning**: Deep, detailed captions using Qwen3-VL-8B-Thinking
- **Face Recognition**: Automatic face detection and grouping with OpenCV + SigLIP embeddings
- **Object Detection**: Identifies objects in photos with RT-DETR (56.2 AP)
- **Smart Duplicate Detection**: Dual deduplication (visual + semantic) using SigLIP2-Giant
- **Tag Extraction**: Automatic searchable tags extracted from captions using spaCy
- **EXIF Support**: Extracts timestamp and GPS coordinates
- **Semantic Merge**: Groups similar photos even with different visual embeddings
- **Web Interface**: Beautiful Flask-based UI for browsing and organizing photos
- **Efficient Processing**: Queues unique photos for deep processing, reuses AI results for duplicates

## Technology Stack

### AI Models (State-of-the-Art v3.3)

- **SigLIP2-Giant** (`google/siglip2-giant-opt-patch16-384`): Multi-modal embeddings (1152-dim)
- **RT-DETR** (`PekingU/rtdetr_r101vd`): Real-time object detection (56.2 AP)
- **Qwen3-VL** (`Qwen/Qwen3-VL-8B-Thinking`): Advanced vision-language model for captions
- **BLIP2** (`Salesforce/blip2-opt-2.7b`): Fast captioning for tier-1 processing
- **spaCy** (`en_core_web_sm`): Tag extraction from captions
- **OpenCV Haar Cascade**: Face detection for person recognition

### Infrastructure

- **FAISS**: High-performance vector similarity search
- **SQLite**: Metadata storage
- **Flask**: Web interface
- **PyTorch**: Deep learning framework

## System Requirements

### Hardware

- **GPU**: NVIDIA GPU with 24GB VRAM (recommended for deep processing)
  - Tier 1 (Fast Ingest): 12GB VRAM
  - Tier 2 (Deep Processing): 24GB VRAM
  - Alternatively: Apple Silicon (MPS) or CPU (slower)
- **RAM**: 32GB+ recommended for large photo libraries
- **Storage**: ~10GB for models, variable for FAISS indexes

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

### Quick Start (v3.3 Unified Pipeline)

The simplest way to process your photos is using the unified pipeline:

```bash
# Run the complete pipeline (ingest + deep processing + semantic merge)
uv run run_pipeline.py
```

This single command will:
1. **Fast Ingest**: Scan photos, extract EXIF, detect duplicates, queue unique photos
2. **Deep Processing**: Generate captions, detect objects, recognize faces
3. **Semantic Merge**: Consolidate similar groups

### Advanced Usage

Run individual stages:

```bash
# Only run fast ingest (visual + semantic deduplication)
uv run run_pipeline.py --ingest-only

# Only run deep processing on queued items
uv run run_pipeline.py --process-only

# Only run semantic merge
uv run run_pipeline.py --merge-only

# Run ingest + processing, skip merge
uv run run_pipeline.py --no-merge
```

### Manual Stage-by-Stage Processing

For more control, you can run each stage separately:

**Stage 1: Fast Ingest**
```bash
uv run index_photos.py
```

**Stage 2: Deep Processing**
```bash
uv run process_queue.py
```

**Stage 3: Semantic Merge (Optional)**
```bash
uv run run_maintenance.py
```

### Prerequisites

Before running the pipeline, download the spaCy language model:

```bash
python -m spacy download en_core_web_sm
```

**Progress tracking:**

- Real-time progress bar with tqdm
- Saves every 100 images (safe to interrupt with Ctrl+C)
- Resume by re-running the script (skips already processed photos)
- Logs are stored in the `logs/` directory

**Check Queue Status:**

After fast ingest, check how many photos are queued for deep processing:

```bash
uv run check_queue.py
```

**Deep Processing (Future):**

Photos are queued with fast captions and EXIF data. Deep processing with Qwen2-VL, face recognition, and tag extraction will be added in a future update. The current web UI works with fast captions.

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
