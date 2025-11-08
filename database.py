"""
Database management for SQLite metadata and FAISS vector index.
"""

import sqlite3
import faiss
import os
from typing import Optional, Tuple

from config import FAISS_INDEX_PATH, SQLITE_DB_PATH, EMBEDDING_DIM


class DatabaseManager:
    """
    Manages both SQLite (metadata) and FAISS (vector search) databases.
    """
    
    def __init__(self):
        self.db_conn: Optional[sqlite3.Connection] = None
        self.cursor: Optional[sqlite3.Cursor] = None
        self.faiss_index: Optional[faiss.Index] = None
        
    def init_databases(self):
        """
        Initialize or load both SQLite and FAISS databases.
        """
        print("\n=== Initializing Databases ===")
        
        # Initialize SQLite
        self._init_sqlite()
        
        # Initialize FAISS
        self._init_faiss()
        
        print("=== Databases initialized ===\n")
        
    def _init_sqlite(self):
        """
        Initialize SQLite database with required tables.
        """
        self.db_conn = sqlite3.connect(SQLITE_DB_PATH)
        self.cursor = self.db_conn.cursor()
        
        # Table 1: Stores info for every single image file
        self.cursor.execute("""
        CREATE TABLE IF NOT EXISTS images (
            image_path TEXT PRIMARY KEY,
            group_id TEXT,
            faiss_id INTEGER
        )
        """)
        
        # Table 2: Stores the AI results for each unique group
        self.cursor.execute("""
        CREATE TABLE IF NOT EXISTS image_groups (
            group_id TEXT PRIMARY KEY,
            canonical_path TEXT,
            generated_caption TEXT,
            detected_objects_json TEXT
        )
        """)
        
        self.db_conn.commit()
        
        # Count existing entries
        self.cursor.execute("SELECT COUNT(*) FROM images")
        image_count = self.cursor.fetchone()[0]
        self.cursor.execute("SELECT COUNT(*) FROM image_groups")
        group_count = self.cursor.fetchone()[0]
        
        print(f"SQLite DB: {image_count} images, {group_count} groups")
        
    def _init_faiss(self):
        """
        Initialize or load FAISS vector index.
        """
        if os.path.exists(FAISS_INDEX_PATH):
            # Load existing index
            self.faiss_index = faiss.read_index(FAISS_INDEX_PATH)
            print(f"FAISS Index: Loaded {self.faiss_index.ntotal} vectors from disk")
        else:
            # Create new index (Inner Product - equivalent to cosine similarity for normalized vectors)
            self.faiss_index = faiss.IndexFlatIP(EMBEDDING_DIM)
            print(f"FAISS Index: Created new index (dim={EMBEDDING_DIM})")
            
    def save_databases(self):
        """
        Save both databases to disk.
        """
        # Commit SQLite changes
        if self.db_conn:
            self.db_conn.commit()
            print(f"✓ SQLite database saved to {SQLITE_DB_PATH}")
        
        # Save FAISS index
        if self.faiss_index:
            faiss.write_index(self.faiss_index, FAISS_INDEX_PATH)
            print(f"✓ FAISS index saved to {FAISS_INDEX_PATH}")
            
    def close(self):
        """
        Close database connections.
        """
        if self.db_conn:
            self.db_conn.close()
            
    def get_stats(self) -> dict:
        """
        Get database statistics.
        """
        stats = {}
        
        if self.cursor:
            self.cursor.execute("SELECT COUNT(*) FROM images")
            stats['total_images'] = self.cursor.fetchone()[0]
            
            self.cursor.execute("SELECT COUNT(*) FROM image_groups")
            stats['unique_groups'] = self.cursor.fetchone()[0]
            
            stats['duplicate_images'] = stats['total_images'] - stats['unique_groups']
            
        if self.faiss_index:
            stats['faiss_vectors'] = self.faiss_index.ntotal
            
        return stats


# Global instance
_db_instance = None


def get_database():
    """
    Get or create the global DatabaseManager instance.
    """
    global _db_instance
    if _db_instance is None:
        _db_instance = DatabaseManager()
        _db_instance.init_databases()
    return _db_instance
