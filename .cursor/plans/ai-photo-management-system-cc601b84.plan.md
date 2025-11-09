<!-- cc601b84-d704-42cc-b7f6-75d7b9df79d6 9db02c32-6e49-4d4b-97e8-b3c716737094 -->
# V3.2 Refinements and Consistency Improvements

## Overview

V3.2 focuses on **consistency** and **debugging improvements** rather than new features:

### Critical Fix: Embedding Model Consistency

**Problem**: Current system mixes embedding models:

- Ingest (tier-1): SigLIP-so400m (1B)
- Deep processing (tier-2): Would use SigLIP-so400m for faces
- Search/maintenance: Would use different embeddings

**Impact**: Embeddings from different models are incompatible, causing poor search results and incorrect face matching.

**Solution**: Use **SigLIP-Giant (2B)** everywhere for all embeddings:

- Visual deduplication
- Caption embeddings  
- Face embeddings
- Search queries
- Semantic merging

**VRAM**: ~8GB for SigLIP-Giant + ~7GB for BLIP2/Qwen = ~15GB (fits in 24GB)

### Key Improvements

1. **Embedding Consistency** (CRITICAL)

- Replace SigLIP-so400m with SigLIP-Giant in index_photos.py
- Use SigLIP-Giant in all processing scripts
- Ensures all embeddings are compatible

2. **Duplicate Debugging**

- Add `duplicate_of_path TEXT` to images table
- Add `duplicate_score REAL` to images table
- Store which photo this is a duplicate of
- Store similarity score for analysis

3. **Improved Face Tracking**

- New table: `face_appearances` (links faces to photos)
- Store bounding box per appearance
- Use OpenCV Haar Cascade for initial face detection (faster)
- Use SigLIP-Giant embeddings for face matching (better quality)
- Change `faiss_face_id` to `representative_faiss_id` in person_groups

4. **Better POS Tagging** (Optional)

- Replace BERT with Spacy (en_core_web_sm)
- Better noun extraction
- Simpler API

## Implementation Plan

### Phase 1: Critical Embedding Fix

**Update config.py**:

- Change EMBEDDING_MODEL_ID to "google/siglip2-giant-opt-patch16-384"
- This applies to ALL scripts using embeddings

**Impact**: BREAKING - old embeddings incompatible

- Must re-index all photos
- Old FAISS indexes invalid
- Worth it for search quality

### Phase 2: Database Schema Updates

**Update database.py**:

- Add columns to images table:
- `duplicate_of_path TEXT`
- `duplicate_score REAL`
- Add new table: `face_appearances`
- `appearance_id INTEGER PRIMARY KEY`
- `image_path TEXT`
- `person_group_id TEXT`
- `box_json TEXT`
- Update person_groups:
- Rename `faiss_face_id` → `representative_faiss_id`
- Add migration logic for v3 → v3.2

### Phase 3: Update Ingest Script

**Update index_photos.py**:

- Already uses correct model (if EMBEDDING_MODEL_ID updated in config)
- Store duplicate debugging info:
- When duplicate found, save `duplicate_of_path` and `duplicate_score`
- Helps debug false positives/negatives

### Phase 4: Improve Processing (Optional)

**Update processing.py** (if creating process_queue.py):

- Use OpenCV for initial face detection (faster)
- Use SigLIP-Giant for face embeddings
- Store face appearances in new table
- Better face tracking

**Alternative**: Keep current processing.py, it already works

### Phase 5: Spacy Integration (Optional)

**Update processing.py tag extraction**:

- Replace BERT POS tagger with Spacy
- Better noun extraction
- Requires: `pip install spacy` and `python -m spacy download en_core_web_sm`

### Phase 6: Testing

- Clear v3 data (embedding model changed)
- Re-run index_photos.py with SigLIP-Giant
- Verify search quality improved
- Check duplicate debugging info

## Recommended Minimal Implementation

Given token constraints and diminishing returns, I recommend:

**CRITICAL (Must Do)**:

1. Update EMBEDDING_MODEL_ID to SigLIP-Giant in config.py
2. Add duplicate debugging columns to images table
3. Re-index with consistent embeddings

**NICE TO HAVE (Can Defer)**:

- Face appearances table (complex)
- OpenCV face detection (optimization)
- Spacy POS tagging (marginal improvement)

## Breaking Changes

- **Embedding model change**: All existing FAISS indexes invalid
- **Database schema**: New columns (but backward compatible with migration)
- **Must re-index**: To get benefits of consistent embeddings

## My Recommendation

**Option A (Minimal, High Impact)**:

1. Change EMBEDDING_MODEL_ID → SigLIP-Giant
2. Add duplicate debugging columns
3. Re-index photos
4. Test search quality

**Option B (Complete v3.2)**:

- Implement all v3.2 features
- Takes significant time/tokens
- Marginal benefit over Option A

**Option C (Skip v3.2)**:

- Current v3 system works well
- V3.2 is refinement, not essential
- Save tokens for other work

Which approach do you prefer?

### To-dos

- [ ] Create project structure, requirements.txt with dependencies, and config.py with constants
- [ ] Implement models.py to load SigLIP2, RT-DETR, and BLIP2 with auto device detection
- [ ] Implement database.py for SQLite tables and FAISS index management
- [ ] Implement processing.py with embedding extraction, expensive AI pipeline, and caching logic
- [ ] Create index_photos.py script to recursively process all 30K+ photos with progress bars
- [ ] Create web_app.py with three routes: gallery, grouped view, and group detail
- [ ] Create HTML templates for gallery, groups, and detail views with modern responsive design
- [ ] Create static/style.css with responsive grid layout and modern UI styling
- [ ] Write comprehensive README.md with installation, usage, and system requirements
- [ ] Update config.py to use SigLIP-Giant for embedding consistency
- [ ] Add duplicate_of_path and duplicate_score columns to images table
- [ ] Update index_photos.py to store duplicate debugging information
- [ ] Re-index sample photos and verify embedding consistency