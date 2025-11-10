<!-- 1b2196e3-de6c-4e2f-8908-a713ab3d5067 8fabe737-2766-401a-8a7b-97beda9c98be -->
# Implement GEMINI_RESEARCH v3.3 Upgrade

## Critical Changes Summary

1. **Fix EMBEDDING_DIM**: 1536 → 1152 (verified correct for SigLIP2-Giant)
2. **Fix dtype**: float16 → bfloat16 on CUDA for better quality
3. **Upgrade captioning**: Qwen2-VL-7B → Qwen3-VL-8B-Thinking
4. **Improve face detection**: Use OpenCV Haar Cascade first, then embed face crops
5. **Ensure consistency**: Same embedding model (SigLIP2-Giant) everywhere
6. **Clear existing data**: Delete all FAISS indexes and rebuild from scratch

## Implementation Steps

### Phase 1: Configuration Updates

**Update `config.py`:**

- Fix `EMBEDDING_DIM = 1152` (currently wrong at 1536)
- Change `CAPTION_MODEL_ID = "Qwen/Qwen3-VL-8B-Thinking"` (upgrade from Qwen2-VL-7B)
- Add OpenCV face cascade configuration

### Phase 2: Data Cleanup

**Clear existing data to rebuild with correct dimensions:**

- Delete `data/photo_library.index` (wrong dimension)
- Delete `data/face_library.index` (wrong dimension)
- Delete `data/caption_library.index` (wrong dimension)
- Delete `data/photo_metadata.db` (incompatible with new dimensions)
- Keep `data/thumbnails/` directory (these are still valid)

### Phase 3: Core Script Updates

**Update `index_photos.py` (Fast Ingest Tier):**

- Fix `get_inference_device_and_dtype()` to return bfloat16 on CUDA (not float16)
- Ensure using `EMBEDDING_DIM = 1152` from config
- Update model loading to use `torch_dtype=DTYPE` instead of `dtype="auto"`
- Verify correct implementation of visual and semantic deduplication
- Already has duplicate debugging (duplicate_of_path, duplicate_score) ✓

**Create new `process_queue.py` (Deep Processing Tier):**

- Based on v3.3 specification in GEMINI_RESEARCH_v3_3.md
- Load heavyweight models: Qwen3-VL-8B-Thinking, RT-DETR, SigLIP2-Giant, spaCy, OpenCV
- Use bfloat16 dtype on CUDA
- Implement NEW face detection workflow:

1. Use OpenCV Haar Cascade to detect faces in grayscale image
2. Crop each detected face from PIL image
3. Embed face crop using SigLIP2-Giant
4. Match against face index or create new person group
5. Store in `face_appearances` table with box coordinates

- Use spaCy for tag extraction (extract NOUN and PROPN)
- Process items from `processing_queue` table
- Update `image_groups` with deep caption, tags, objects
- Save face index when complete

**Update `run_maintenance.py` (Semantic Merge):**

- Fix to use correct `EMBEDDING_DIM = 1152`
- Use bfloat16 dtype on CUDA
- Ensure consistent with other scripts

### Phase 4: Search and Web App

**Update search functionality in `web_app.py`:**

- Verify it loads correct `EMBEDDING_DIM = 1152`
- Use bfloat16 dtype for CUDA inference
- No major logic changes needed

**Apply template fixes from diff:**

- Fix thumbnail URL paths: `/thumbnail{{ ... }}` → `/thumbnail/{{ ... }}`
- Files: `index.html`, `person_detail.html`, `object_detail.html`, `groups.html`, `search_results.html`

### Phase 5: Dependencies

**Verify all dependencies in `pyproject.toml`:**

- opencv-python-headless ✓ (already present)
- spacy ✓ (already present)
- Add note to download spaCy model: `python -m spacy download en_core_web_sm`

### Phase 6: Testing

**Create minimal test script:**

- Test embedding dimension is correct (1152)
- Test bfloat16 dtype is used on CUDA
- Test face detection with OpenCV
- Test Qwen3-VL model loading

### To-dos

- [ ] Fix config.py: EMBEDDING_DIM=1152, Qwen3-VL-8B-Thinking, bfloat16 dtype
- [ ] Delete existing FAISS indexes and SQLite database to rebuild with correct dimensions
- [ ] Update index_photos.py: bfloat16 dtype, correct embedding usage
- [ ] Create process_queue.py with OpenCV face detection and Qwen3-VL captioning
- [ ] Update run_maintenance.py to use correct dimension and dtype
- [ ] Fix thumbnail URL paths in all template files
- [ ] Verify web_app.py uses correct embedding dimension and dtype