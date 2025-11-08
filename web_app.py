#!/usr/bin/env python3
"""
Flask web interface for browsing indexed photos.
Provides gallery view, grouped view, and detail view.
"""

import os
import json
from pathlib import Path
from flask import Flask, render_template, send_file, abort
from collections import defaultdict

from config import FLASK_HOST, FLASK_PORT, FLASK_DEBUG, PHOTOS_DIR, SQLITE_DB_PATH
import sqlite3

app = Flask(__name__)


def get_db_connection():
    """Get a new database connection for this thread."""
    conn = sqlite3.connect(SQLITE_DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def get_group_categories():
    """
    Group photos by similar captions (first 3-4 words).
    Returns a dictionary mapping category names to lists of groups.
    """
    conn = get_db_connection()
    cursor = conn.cursor()
    
    cursor.execute("""
        SELECT group_id, canonical_path, generated_caption, detected_objects_json
        FROM image_groups
        ORDER BY generated_caption
    """)
    
    groups = cursor.fetchall()
    conn.close()
    
    # Categorize by first few words of caption
    categories = defaultdict(list)
    
    for row in groups:
        group_id = row['group_id']
        canonical_path = row['canonical_path']
        caption = row['generated_caption']
        objects_json = row['detected_objects_json']
        
        # Extract first 3-4 words as category
        words = caption.split()[:3]
        category = " ".join(words).capitalize()
        
        # Count objects
        objects = json.loads(objects_json)
        
        categories[category].append({
            'group_id': group_id,
            'canonical_path': canonical_path,
            'caption': caption,
            'object_count': len(objects)
        })
    
    return dict(categories)


@app.route('/')
def gallery():
    """
    Gallery view - show all unique photo groups in a grid.
    """
    conn = get_db_connection()
    cursor = conn.cursor()
    
    # Get all unique groups
    cursor.execute("""
        SELECT group_id, canonical_path, generated_caption, detected_objects_json
        FROM image_groups
        ORDER BY group_id
    """)
    
    groups = []
    for row in cursor.fetchall():
        group_id = row['group_id']
        canonical_path = row['canonical_path']
        caption = row['generated_caption']
        objects_json = row['detected_objects_json']
        objects = json.loads(objects_json)
        
        # Get count of images in this group
        cursor.execute(
            "SELECT COUNT(*) FROM images WHERE group_id = ?",
            (group_id,)
        )
        image_count = cursor.fetchone()[0]
        
        groups.append({
            'group_id': group_id,
            'canonical_path': canonical_path,
            'caption': caption,
            'object_count': len(objects),
            'image_count': image_count
        })
    
    # Get stats
    cursor.execute("SELECT COUNT(*) FROM images")
    total_images = cursor.fetchone()[0]
    cursor.execute("SELECT COUNT(*) FROM image_groups")
    unique_groups = cursor.fetchone()[0]
    
    stats = {
        'total_images': total_images,
        'unique_groups': unique_groups,
        'duplicate_images': total_images - unique_groups
    }
    
    conn.close()
    
    return render_template(
        'index.html',
        groups=groups,
        stats=stats
    )


@app.route('/groups')
def groups_view():
    """
    Grouped/categorized view - show photos grouped by similar captions.
    """
    categories = get_group_categories()
    
    conn = get_db_connection()
    cursor = conn.cursor()
    
    cursor.execute("SELECT COUNT(*) FROM images")
    total_images = cursor.fetchone()[0]
    cursor.execute("SELECT COUNT(*) FROM image_groups")
    unique_groups = cursor.fetchone()[0]
    
    stats = {
        'total_images': total_images,
        'unique_groups': unique_groups,
        'duplicate_images': total_images - unique_groups
    }
    
    conn.close()
    
    return render_template(
        'groups.html',
        categories=categories,
        stats=stats
    )


@app.route('/group/<group_id>')
def group_detail(group_id):
    """
    Detail view for a specific photo group.
    Shows full image, caption, detected objects, and similar images.
    """
    conn = get_db_connection()
    cursor = conn.cursor()
    
    # Get group information
    cursor.execute(
        """
        SELECT canonical_path, generated_caption, detected_objects_json
        FROM image_groups
        WHERE group_id = ?
        """,
        (group_id,)
    )
    
    result = cursor.fetchone()
    if not result:
        conn.close()
        abort(404)
        
    canonical_path = result['canonical_path']
    caption = result['generated_caption']
    objects_json = result['detected_objects_json']
    objects = json.loads(objects_json)
    
    # Get all images in this group
    cursor.execute(
        "SELECT image_path FROM images WHERE group_id = ? ORDER BY image_path",
        (group_id,)
    )
    
    similar_images = [row['image_path'] for row in cursor.fetchall()]
    
    conn.close()
    
    # Get file info for canonical image
    file_size = os.path.getsize(canonical_path) if os.path.exists(canonical_path) else 0
    file_size_mb = file_size / (1024 * 1024)
    
    return render_template(
        'group_detail.html',
        group_id=group_id,
        canonical_path=canonical_path,
        caption=caption,
        objects=objects,
        similar_images=similar_images,
        file_size_mb=file_size_mb
    )


@app.route('/photos/<path:filepath>')
def serve_photo(filepath):
    """
    Serve a photo file from the photos directory.
    Security: Validate path to prevent directory traversal.
    """
    # Construct full path
    full_path = os.path.join(PHOTOS_DIR, filepath)
    
    # Security check: ensure the path is within PHOTOS_DIR
    real_path = os.path.realpath(full_path)
    real_photos_dir = os.path.realpath(PHOTOS_DIR)
    
    if not real_path.startswith(real_photos_dir):
        abort(403)  # Forbidden
        
    if not os.path.exists(real_path):
        abort(404)
        
    return send_file(real_path)


@app.route('/photo/<path:full_path>')
def serve_photo_absolute(full_path):
    """
    Serve a photo file using its absolute path.
    Security: Validate path to prevent unauthorized access.
    """
    # Prepend leading slash if not present
    if not full_path.startswith('/'):
        full_path = '/' + full_path
        
    # Security check: ensure the path is within PHOTOS_DIR
    real_path = os.path.realpath(full_path)
    real_photos_dir = os.path.realpath(PHOTOS_DIR)
    
    if not real_path.startswith(real_photos_dir):
        abort(403)  # Forbidden
        
    if not os.path.exists(real_path):
        abort(404)
        
    return send_file(real_path)


@app.template_filter('basename')
def basename_filter(path):
    """Template filter to get just the filename from a path."""
    return os.path.basename(path)


@app.template_filter('relative_to_photos')
def relative_to_photos_filter(path):
    """Template filter to get path relative to PHOTOS_DIR."""
    try:
        return os.path.relpath(path, PHOTOS_DIR)
    except ValueError:
        return path


if __name__ == '__main__':
    print("\n" + "="*60)
    print("AI Photo Management System - Web Interface")
    print("="*60)
    
    # Get database statistics
    conn = get_db_connection()
    cursor = conn.cursor()
    
    cursor.execute("SELECT COUNT(*) FROM images")
    total_images = cursor.fetchone()[0]
    cursor.execute("SELECT COUNT(*) FROM image_groups")
    unique_groups = cursor.fetchone()[0]
    
    print(f"\nDatabase Statistics:")
    print(f"  Total images: {total_images}")
    print(f"  Unique groups: {unique_groups}")
    print(f"  Duplicate images: {total_images - unique_groups}")
    
    conn.close()
    
    print(f"\nStarting Flask server...")
    print(f"  URL: http://{FLASK_HOST}:{FLASK_PORT}")
    print(f"  Gallery: http://{FLASK_HOST}:{FLASK_PORT}/")
    print(f"  Groups: http://{FLASK_HOST}:{FLASK_PORT}/groups")
    print("="*60 + "\n")
    
    app.run(host=FLASK_HOST, port=FLASK_PORT, debug=FLASK_DEBUG)
