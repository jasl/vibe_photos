#!/usr/bin/env python3
"""
Small test script to verify the system works before processing all photos.
"""

import os
from pathlib import Path
from config import PHOTOS_DIR, SUPPORTED_EXTENSIONS
from processing import get_processor
from database import get_database

def find_test_images(root_dir: str, max_images: int = 5) -> list:
    """Find a small set of test images."""
    image_paths = []
    
    for root, dirs, files in os.walk(root_dir):
        for file in files:
            if Path(file).suffix in SUPPORTED_EXTENSIONS:
                full_path = os.path.join(root, file)
                image_paths.append(full_path)
                
                if len(image_paths) >= max_images:
                    return image_paths
                    
    return image_paths

def main():
    print("\n" + "="*60)
    print("AI Photo Management System - Quick Test")
    print("="*60)
    
    # Find test images
    print(f"\nFinding test images in: {PHOTOS_DIR}")
    test_images = find_test_images(PHOTOS_DIR, max_images=5)
    
    if not test_images:
        print("ERROR: No test images found!")
        return
        
    print(f"Found {len(test_images)} test images")
    for img in test_images:
        print(f"  - {Path(img).name}")
    
    # Initialize system
    print("\nInitializing AI models and databases...")
    processor = get_processor()
    db = get_database()
    
    # Process test images
    print("\nProcessing test images...")
    for image_path in test_images:
        print(f"\nProcessing: {Path(image_path).name}")
        success = processor.process_new_image(image_path)
        if success:
            print("  ✓ Success")
        else:
            print("  ✗ Failed")
    
    # Save and show stats
    print("\nSaving databases...")
    db.save_databases()
    
    stats = db.get_stats()
    print("\n" + "="*60)
    print("Test Complete!")
    print("="*60)
    print(f"\nStatistics:")
    print(f"  Total images: {stats['total_images']}")
    print(f"  Unique groups: {stats['unique_groups']}")
    print(f"  Duplicate images: {stats['duplicate_images']}")
    print("\nTest successful! You can now run: uv run index_photos.py")
    print("="*60 + "\n")

if __name__ == "__main__":
    main()

