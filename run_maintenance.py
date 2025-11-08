#!/usr/bin/env python3
"""
Maintenance script for semantic merging (Solution 2).
Finds and merges image groups with similar captions but different visual embeddings.
This corrects for errors caused by lighting, angles, or other variations.
"""

import torch
import faiss
import numpy as np
import sqlite3
import sys

from config import (
    EMBEDDING_MODEL_ID,
    EMBEDDING_DIM,
    SEMANTIC_SIMILARITY_THRESHOLD,
    SQLITE_DB_PATH
)
from transformers import AutoProcessor, AutoModel


def get_inference_device():
    """Auto-detect device for inference."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


@torch.no_grad()
def get_text_embedding(text_list: list, processor, model, device) -> np.ndarray:
    """
    Generate SigLIP embeddings for a batch of text captions.
    
    Args:
        text_list: List of caption strings
        processor: SigLIP processor
        model: SigLIP model
        device: Torch device
        
    Returns:
        L2-normalized numpy array of embeddings
    """
    inputs = processor(
        text=text_list,
        return_tensors="pt",
        padding=True,
        truncation=True
    ).to(device)
    
    text_features = model.get_text_features(**inputs)
    
    vector = text_features.cpu().numpy().astype(np.float32)
    faiss.normalize_L2(vector)
    return vector


def merge_groups(cursor, conn, primary_group_id: str, secondary_group_id: str):
    """
    Consolidate two groups in the database.
    
    Args:
        cursor: SQLite cursor
        conn: SQLite connection
        primary_group_id: Group to keep
        secondary_group_id: Group to merge into primary
    """
    print(f"  > MERGING: {secondary_group_id} → {primary_group_id}")
    
    try:
        # Re-assign all images from secondary to primary group
        cursor.execute(
            "UPDATE images SET group_id = ? WHERE group_id = ?",
            (primary_group_id, secondary_group_id)
        )
        
        # Delete the redundant secondary group entry
        cursor.execute(
            "DELETE FROM image_groups WHERE group_id = ?",
            (secondary_group_id,)
        )
        
        conn.commit()
        
    except Exception as e:
        print(f"  > MERGE FAILED: {e}")
        conn.rollback()


def run_semantic_merge_task():
    """
    Main semantic merging workflow.
    Finds groups with similar captions and merges them.
    """
    print("\n" + "="*60)
    print("Semantic Merging Maintenance Task (Solution 2)")
    print("="*60)
    
    # Load database
    db_conn = sqlite3.connect(SQLITE_DB_PATH)
    cursor = db_conn.cursor()
    
    # Get all unique captions and their group IDs
    cursor.execute("SELECT group_id, generated_caption FROM image_groups")
    rows = cursor.fetchall()
    
    if len(rows) < 2:
        print("\nNot enough groups to merge. Exiting.")
        db_conn.close()
        return
        
    group_ids = [row[0] for row in rows]
    captions = [row[1] for row in rows]
    
    print(f"\nFound {len(captions)} unique image groups")
    print(f"Creating text embeddings for all captions...")
    
    # Load embedding model
    device = get_inference_device()
    print(f"Loading Embedding Model: {EMBEDDING_MODEL_ID}")
    
    embed_processor = AutoProcessor.from_pretrained(EMBEDDING_MODEL_ID)
    embed_model = AutoModel.from_pretrained(
        EMBEDDING_MODEL_ID,
        dtype="auto",
        device_map="auto"
    ).eval()
    
    # Create embeddings for all captions (batch processing)
    caption_embeddings = get_text_embedding(captions, embed_processor, embed_model, device)
    
    # Build temporary FAISS index for captions
    caption_index = faiss.IndexFlatIP(EMBEDDING_DIM)
    caption_index.add(caption_embeddings)
    
    print("Searching for semantic duplicates...")
    
    # Search for k nearest neighbors
    k = 5
    all_scores, all_indices = caption_index.search(caption_embeddings, k=k)
    
    merged_groups = set()
    total_merges = 0
    
    # Iterate and apply merge logic
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
                
            # Decision: Merge if semantically similar enough
            if neighbor_score > SEMANTIC_SIMILARITY_THRESHOLD:
                print(f"\nSimilarity: {neighbor_score:.4f}")
                print(f"  Primary: '{captions[i][:60]}...'")
                print(f"  Secondary: '{captions[neighbor_index][:60]}...'")
                
                merge_groups(cursor, db_conn, primary_group_id, secondary_group_id)
                merged_groups.add(secondary_group_id)
                total_merges += 1
                
    print("\n" + "="*60)
    print(f"Semantic Merge Complete!")
    print(f"Total merges performed: {total_merges}")
    print("="*60 + "\n")
    
    db_conn.close()


if __name__ == "__main__":
    try:
        run_semantic_merge_task()
    except KeyboardInterrupt:
        print("\n\nMaintenance interrupted by user.")
        sys.exit(0)
    except Exception as e:
        print(f"\n\nFATAL ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

