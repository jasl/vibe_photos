# v3.3 Upgrade Summary

## ✅ What Was Changed

### 1. Critical Fixes
- **Verified EMBEDDING_DIM**: **1536** (correct for SigLIP2-Giant vision features)
- **Fixed dtype**: float16 → **bfloat16** on CUDA for better quality
- **Upgraded caption model**: Qwen2-VL-7B → **Qwen3-VL-8B-Thinking**
- **Fixed model loading**: Changed deprecated torch_dtype to dtype parameter
- **Fixed Qwen3-VL loading**: Use Qwen3VLForConditionalGeneration class

### 2. Data Cleanup
- Deleted old FAISS indexes with wrong dimensions:
  - `photo_library.index`
  - `face_library.index`
  - `caption_library.index`
- Deleted old SQLite database with incompatible schema
- Kept thumbnail cache (still valid)

### 3. Updated Scripts

#### `config.py`
- EMBEDDING_DIM = 1536 (verified: vision features are 1536-dim)
- CAPTION_MODEL_ID = "Qwen/Qwen3-VL-8B-Thinking"

#### `index_photos.py` (Fast Ingest)
- Uses bfloat16 on CUDA
- Correct torch_dtype parameter

#### `process_queue.py` (NEW - Deep Processing)
- Qwen3-VL-8B-Thinking for captions
- OpenCV Haar Cascade for face detection
- Face crops embedded with SigLIP2-Giant
- spaCy for tag extraction
- Stores faces in `face_appearances` table

#### `run_maintenance.py` (Semantic Merge)
- Uses correct embedding dimension
- Uses bfloat16 on CUDA

#### `run_pipeline.py` (NEW - Unified Script)
- Combines all three stages in one script
- Efficient model loading and reuse
- Progress bars for all stages
- CLI options for individual stages

### 4. Templates
- Fixed thumbnail URL paths (already correct in main branch)

## 🚀 New Unified Workflow

### Option 1: Run Everything (Recommended)

```bash
# Complete pipeline: ingest → process → merge
uv run run_pipeline.py
```

### Option 2: Run Individual Stages

```bash
# Only fast ingest
uv run run_pipeline.py --ingest-only

# Only deep processing
uv run run_pipeline.py --process-only

# Only semantic merge
uv run run_pipeline.py --merge-only

# Ingest + process (skip merge)
uv run run_pipeline.py --no-merge
```

### Option 3: Manual Stage-by-Stage

```bash
# Stage 1: Fast ingest
uv run index_photos.py

# Stage 2: Deep processing
uv run process_queue.py

# Stage 3: Semantic merge (optional)
uv run run_maintenance.py
```

## 📋 Prerequisites

Before running, install the spaCy language model:

```bash
python -m spacy download en_core_web_sm
```

## 🎯 Expected Performance

### Tier 1 (Fast Ingest)
- **VRAM**: ~12GB
- **Speed**: ~5-10 images/second
- **Models**: SigLIP2-Giant + BLIP2

### Tier 2 (Deep Processing)
- **VRAM**: ~24GB
- **Speed**: ~1-2 images/second
- **Models**: Qwen3-VL + RT-DETR + SigLIP2-Giant + OpenCV + spaCy

## 🎨 What's New in v3.3

1. **Correct embedding dimension** - Fixed critical bug affecting similarity scores
2. **Better quality** - bfloat16 dtype on CUDA for improved precision
3. **Advanced captions** - Qwen3-VL-8B-Thinking produces detailed descriptions
4. **Face recognition** - OpenCV detects faces, SigLIP embeddings group them
5. **Tag extraction** - spaCy extracts searchable nouns from captions
6. **Unified script** - Single command to run entire pipeline
7. **Progress tracking** - Clear progress bars for all stages

## 📊 Database Schema (v3.3)

### Tables:
- **images**: Every image file with group assignment and duplicate info
- **image_groups**: AI results for each unique image group
- **person_groups**: Unique persons detected across photos
- **face_appearances**: Links faces to photos with bounding boxes
- **processing_queue**: Groups waiting for deep processing

## 🌐 Web Interface

After processing, start the web interface:

```bash
uv run web_app.py
```

Then browse to: http://localhost:5000

### Features:
- Gallery view with thumbnails
- Group detail with all duplicates
- Object browsing (e.g., all photos with "cat")
- Person browsing (groups of same person)
- Search by caption, tags, or objects
- Statistics dashboard

## 🔄 Migration from v3.2

The database was completely rebuilt with correct dimensions. Your old data is gone, but this ensures:
- Correct similarity calculations
- Proper face grouping
- Better caption quality

Simply re-run the pipeline on your photos directory to rebuild everything with v3.3.

## 📝 Notes

- The unified pipeline automatically saves progress every 100 images
- You can interrupt with Ctrl+C and resume later
- Models are cached after first download (~10GB total)
- Face recognition improves as it sees more photos
- Semantic merge is optional but recommended for consolidating similar groups

