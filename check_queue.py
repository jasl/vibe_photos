#!/usr/bin/env python3
"""
Utility script to check processing queue status (v3).
"""

import sqlite3
from config import SQLITE_DB_PATH


def main():
    """Display processing queue status."""
    print("\n" + "="*60)
    print("Processing Queue Status (V3)")
    print("="*60)
    
    conn = sqlite3.connect(SQLITE_DB_PATH)
    cursor = conn.cursor()
    
    # Get overall stats
    cursor.execute("SELECT COUNT(*) FROM image_groups")
    total_groups = cursor.fetchone()[0]
    
    cursor.execute("SELECT COUNT(*) FROM processing_queue")
    queued = cursor.fetchone()[0]
    
    cursor.execute("SELECT COUNT(*) FROM image_groups WHERE processing_status = 'COMPLETED'")
    completed = cursor.fetchone()[0]
    
    cursor.execute("SELECT COUNT(*) FROM image_groups WHERE processing_status = 'FAILED'")
    failed = cursor.fetchone()[0]
    
    cursor.execute("SELECT COUNT(*) FROM image_groups WHERE processing_status = 'QUEUED'")
    pending = cursor.fetchone()[0]
    
    print(f"\nOverall Statistics:")
    print(f"  Total groups: {total_groups}")
    print(f"  In queue: {queued}")
    print(f"  Completed: {completed}")
    print(f"  Failed: {failed}")
    print(f"  Pending: {pending}")
    
    if queued > 0:
        print(f"\n📋 Queue Status:")
        print(f"  {queued} groups waiting for deep processing")
        print(f"\n  Run: uv run process_queue.py")
    else:
        print(f"\n✅ Queue is empty!")
        
    if failed > 0:
        print(f"\n⚠️  {failed} groups failed processing")
        print(f"  Consider re-queuing or investigating errors")
        
    print("="*60 + "\n")
    
    conn.close()


if __name__ == "__main__":
    main()


