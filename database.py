"""
Database management for SQLite metadata and FAISS vector indexes (v3).
Supports cascade architecture with fast ingest and deep processing queue.
"""

import sqlite3
import faiss
import os
from typing import Optional, Tuple

from config import (
    FAISS_IMAGE_INDEX_PATH,
    FAISS_FACE_INDEX_PATH,
    SQLITE_DB_PATH,
    EMBEDDING_DIM
)

# V3: Add caption index path
CAPTION_INDEX_PATH = os.path.join(os.path.dirname(FAISS_IMAGE_INDEX_PATH), "caption_library.index")


class DatabaseManager:
    """
    Manages both SQLite (metadata) and FAISS (vector search) databases (v3).
    Supports cascade architecture with processing queue.
    """
    
    def __init__(self):
        self.db_conn: Optional[sqlite3.Connection] = None
        self.cursor: Optional[sqlite3.Cursor] = None
        self.faiss_image_index: Optional[faiss.Index] = None
        self.faiss_face_index: Optional[faiss.Index] = None
        self.faiss_caption_index: Optional[faiss.Index] = None
        
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
            faiss_image_id INTEGER
        )
        """)
        
        # Table 2: Stores the AI results for each unique group (v3 - cascade)
        self.cursor.execute("""
        CREATE TABLE IF NOT EXISTS image_groups (
            group_id TEXT PRIMARY KEY,
            canonical_path TEXT,
            fast_caption TEXT,
            generated_caption TEXT,
            detected_objects_json TEXT,
            extracted_tags_json TEXT,
            processing_status TEXT DEFAULT 'QUEUED',
            taken_at_timestamp TEXT,
            gps_latitude REAL,
            gps_longitude REAL
        )
        """)
        
        # Table 3: Stores each unique person (v2 - Solution 1)
        self.cursor.execute("""
        CREATE TABLE IF NOT EXISTS person_groups (
            person_group_id TEXT PRIMARY KEY,
            name TEXT,
            faiss_face_id INTEGER
        )
        """)
        
        # Table 4: Queue for deep processing (v3)
        self.cursor.execute("""
        CREATE TABLE IF NOT EXISTS processing_queue (
            group_id TEXT PRIMARY KEY,
            FOREIGN KEY (group_id) REFERENCES image_groups (group_id)
        )
        """)
        
        self.db_conn.commit()
        
        # V3: Schema migration - Add new columns if they don't exist
        self._migrate_schema_to_v3()
        
        # Count existing entries
        self.cursor.execute("SELECT COUNT(*) FROM images")
        image_count = self.cursor.fetchone()[0]
        self.cursor.execute("SELECT COUNT(*) FROM image_groups")
        group_count = self.cursor.fetchone()[0]
        self.cursor.execute("SELECT COUNT(*) FROM person_groups")
        person_count = self.cursor.fetchone()[0]
        
        print(f"SQLite DB: {image_count} images, {group_count} groups, {person_count} persons")
        
    def _migrate_schema_to_v3(self):
        """
        Migrate v2 schema to v3 by adding new columns if they don't exist.
        This handles upgrades gracefully without losing data.
        """
        # Check which columns exist in image_groups
        self.cursor.execute("PRAGMA table_info(image_groups)")
        existing_columns = {row[1] for row in self.cursor.fetchall()}
        
        migrations_needed = []
        
        # Add fast_caption column if missing
        if 'fast_caption' not in existing_columns:
            migrations_needed.append("ALTER TABLE image_groups ADD COLUMN fast_caption TEXT")
            
        # Add processing_status column if missing
        if 'processing_status' not in existing_columns:
            migrations_needed.append("ALTER TABLE image_groups ADD COLUMN processing_status TEXT DEFAULT 'QUEUED'")
            
        # Add taken_at_timestamp column if missing
        if 'taken_at_timestamp' not in existing_columns:
            migrations_needed.append("ALTER TABLE image_groups ADD COLUMN taken_at_timestamp TEXT")
            
        # Add gps_latitude column if missing
        if 'gps_latitude' not in existing_columns:
            migrations_needed.append("ALTER TABLE image_groups ADD COLUMN gps_latitude REAL")
            
        # Add gps_longitude column if missing
        if 'gps_longitude' not in existing_columns:
            migrations_needed.append("ALTER TABLE image_groups ADD COLUMN gps_longitude REAL")
            
        # Execute migrations
        if migrations_needed:
            print(f"\nMigrating schema to v3 ({len(migrations_needed)} columns to add)...")
            for migration in migrations_needed:
                try:
                    self.cursor.execute(migration)
                    print(f"  ✓ {migration.split('ADD COLUMN')[1].split()[0]}")
                except Exception as e:
                    print(f"  ✗ Migration failed: {e}")
            self.db_conn.commit()
            print("Schema migration complete.\n")
        
    def _init_faiss(self):
        """
        Initialize or load FAISS vector indexes (v3: image + face + caption).
        """
        # Image index
        if os.path.exists(FAISS_IMAGE_INDEX_PATH):
            self.faiss_image_index = faiss.read_index(FAISS_IMAGE_INDEX_PATH)
            print(f"FAISS Image Index: Loaded {self.faiss_image_index.ntotal} vectors from disk")
        else:
            self.faiss_image_index = faiss.IndexFlatIP(EMBEDDING_DIM)
            print(f"FAISS Image Index: Created new index (dim={EMBEDDING_DIM})")
            
        # Face index (v2 - Solution 1)
        if os.path.exists(FAISS_FACE_INDEX_PATH):
            self.faiss_face_index = faiss.read_index(FAISS_FACE_INDEX_PATH)
            print(f"FAISS Face Index: Loaded {self.faiss_face_index.ntotal} vectors from disk")
        else:
            self.faiss_face_index = faiss.IndexFlatIP(EMBEDDING_DIM)
            print(f"FAISS Face Index: Created new index (dim={EMBEDDING_DIM})")
            
        # Caption index (v3 - semantic deduplication)
        if os.path.exists(CAPTION_INDEX_PATH):
            self.faiss_caption_index = faiss.read_index(CAPTION_INDEX_PATH)
            print(f"FAISS Caption Index: Loaded {self.faiss_caption_index.ntotal} vectors from disk")
        else:
            self.faiss_caption_index = faiss.IndexFlatIP(EMBEDDING_DIM)
            print(f"FAISS Caption Index: Created new index (dim={EMBEDDING_DIM})")
            
    def save_databases(self):
        """
        Save all databases to disk (v3).
        """
        # Commit SQLite changes
        if self.db_conn:
            self.db_conn.commit()
            print(f"✓ SQLite database saved to {SQLITE_DB_PATH}")
        
        # Save FAISS indexes
        if self.faiss_image_index:
            faiss.write_index(self.faiss_image_index, FAISS_IMAGE_INDEX_PATH)
            print(f"✓ FAISS image index saved to {FAISS_IMAGE_INDEX_PATH}")
            
        if self.faiss_face_index:
            faiss.write_index(self.faiss_face_index, FAISS_FACE_INDEX_PATH)
            print(f"✓ FAISS face index saved to {FAISS_FACE_INDEX_PATH}")
            
        if self.faiss_caption_index:
            faiss.write_index(self.faiss_caption_index, CAPTION_INDEX_PATH)
            print(f"✓ FAISS caption index saved to {CAPTION_INDEX_PATH}")
            
    def close(self):
        """
        Close database connections.
        """
        if self.db_conn:
            self.db_conn.close()
            
    def get_stats(self) -> dict:
        """
        Get database statistics (v3).
        """
        stats = {}
        
        if self.cursor:
            self.cursor.execute("SELECT COUNT(*) FROM images")
            stats['total_images'] = self.cursor.fetchone()[0]
            
            self.cursor.execute("SELECT COUNT(*) FROM image_groups")
            stats['unique_groups'] = self.cursor.fetchone()[0]
            
            self.cursor.execute("SELECT COUNT(*) FROM person_groups")
            stats['unique_persons'] = self.cursor.fetchone()[0]
            
            # V3: Processing queue stats
            self.cursor.execute("SELECT COUNT(*) FROM processing_queue")
            stats['queued_for_processing'] = self.cursor.fetchone()[0]
            
            self.cursor.execute("SELECT COUNT(*) FROM image_groups WHERE processing_status = 'COMPLETED'")
            stats['completed_processing'] = self.cursor.fetchone()[0]
            
            self.cursor.execute("SELECT COUNT(*) FROM image_groups WHERE processing_status = 'FAILED'")
            stats['failed_processing'] = self.cursor.fetchone()[0]
            
            stats['duplicate_images'] = stats['total_images'] - stats['unique_groups']
            
        if self.faiss_image_index:
            stats['faiss_image_vectors'] = self.faiss_image_index.ntotal
            
        if self.faiss_face_index:
            stats['faiss_face_vectors'] = self.faiss_face_index.ntotal
            
        if self.faiss_caption_index:
            stats['faiss_caption_vectors'] = self.faiss_caption_index.ntotal
            
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
