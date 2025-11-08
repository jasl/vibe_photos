#!/usr/bin/env python3
"""
Main script to index all photos in the configured directory.
Recursively scans for images and processes them with AI models.
"""

import os
import sys
from pathlib import Path
from tqdm import tqdm

from config import PHOTOS_DIR, SUPPORTED_EXTENSIONS
from processing import get_processor
from database import get_database


def find_all_images(root_dir: str) -> list:
    """
    Recursively find all image files in the given directory.
    
    Args:
        root_dir: Root directory to search
        
    Returns:
        List of full paths to image files
    """
    image_paths = []
    
    print(f"\nScanning for images in: {root_dir}")
    
    for root, dirs, files in os.walk(root_dir):
        for file in files:
            if Path(file).suffix in SUPPORTED_EXTENSIONS:
                full_path = os.path.join(root, file)
                image_paths.append(full_path)
                
    print(f"Found {len(image_paths)} images")
    return image_paths


def main():
    """
    Main indexing workflow.
    """
    print("\n" + "="*60)
    print("AI Photo Management System - Photo Indexing")
    print("="*60)
    
    # Check if photos directory exists
    if not os.path.exists(PHOTOS_DIR):
        print(f"\nERROR: Photos directory not found: {PHOTOS_DIR}")
        sys.exit(1)
        
    # Find all images
    image_paths = find_all_images(PHOTOS_DIR)
    
    if not image_paths:
        print("\nNo images found to process!")
        sys.exit(1)
        
    # Initialize processor (this loads models and databases)
    print("\nInitializing AI models and databases...")
    processor = get_processor()
    db = get_database()
    
    # Check which images are already processed
    print("\nChecking for already processed images...")
    db.cursor.execute("SELECT image_path FROM images")
    processed_paths = {row[0] for row in db.cursor.fetchall()}
    
    images_to_process = [p for p in image_paths if p not in processed_paths]
    
    print(f"\nTotal images: {len(image_paths)}")
    print(f"Already processed: {len(processed_paths)}")
    print(f"To process: {len(images_to_process)}")
    
    if not images_to_process:
        print("\nAll images have already been processed!")
        stats = db.get_stats()
        print(f"\nFinal Statistics:")
        print(f"  Total images: {stats['total_images']}")
        print(f"  Unique groups: {stats['unique_groups']}")
        print(f"  Duplicate images: {stats['duplicate_images']}")
        return
        
    # Process each image with progress bar
    print("\nProcessing images...")
    print("(This may take several hours for 30K+ images)\n")
    
    success_count = 0
    error_count = 0
    duplicate_count = 0
    unique_count = 0
    
    # Track stats before processing
    initial_groups = db.get_stats()['unique_groups']
    
    with tqdm(total=len(images_to_process), unit="img") as pbar:
        for image_path in images_to_process:
            pbar.set_description(f"Processing {Path(image_path).name}")
            
            # Track groups before processing this image
            groups_before = db.get_stats()['unique_groups']
            
            # Process the image
            success = processor.process_new_image(image_path)
            
            if success:
                success_count += 1
                
                # Check if a new group was created
                groups_after = db.get_stats()['unique_groups']
                if groups_after > groups_before:
                    unique_count += 1
                else:
                    duplicate_count += 1
            else:
                error_count += 1
                
            pbar.update(1)
            
            # Save progress every 100 images
            if success_count % 100 == 0:
                db.save_databases()
                
    # Final save
    print("\nSaving databases...")
    db.save_databases()
    
    # Print final statistics
    print("\n" + "="*60)
    print("Indexing Complete!")
    print("="*60)
    
    stats = db.get_stats()
    
    print(f"\nProcessing Summary:")
    print(f"  Successfully processed: {success_count}")
    print(f"  Errors/Skipped: {error_count}")
    print(f"  New unique images: {unique_count}")
    print(f"  Duplicates found: {duplicate_count}")
    
    print(f"\nFinal Database Statistics:")
    print(f"  Total images: {stats['total_images']}")
    print(f"  Unique groups: {stats['unique_groups']}")
    print(f"  Duplicate images: {stats['duplicate_images']}")
    print(f"  FAISS vectors: {stats['faiss_vectors']}")
    
    print(f"\nDatabase files saved:")
    print(f"  SQLite: {db.db_conn.execute('PRAGMA database_list').fetchone()[2]}")
    print(f"  FAISS: {os.path.abspath('data/photo_library.index')}")
    
    print("\n" + "="*60)
    print("You can now run the web UI with: uv run web_app.py")
    print("="*60 + "\n")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nIndexing interrupted by user.")
        print("Saving progress...")
        db = get_database()
        db.save_databases()
        print("Progress saved. You can resume by running this script again.")
        sys.exit(0)
    except Exception as e:
        print(f"\n\nFATAL ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
