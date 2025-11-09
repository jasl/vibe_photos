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
        SELECT group_id, canonical_path, generated_caption, fast_caption, detected_objects_json
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
        caption = row['generated_caption'] or row['fast_caption']  # V3: Fallback to fast caption
        objects_json = row['detected_objects_json']
        
        # Skip if no caption available
        if not caption:
            continue
        
        # Extract first 3-4 words as category
        words = caption.split()[:3]
        category = " ".join(words).capitalize()
        
        # Count objects
        objects = json.loads(objects_json) if objects_json else []
        
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
        SELECT group_id, canonical_path, generated_caption, fast_caption, detected_objects_json
        FROM image_groups
        ORDER BY group_id
    """)
    
    groups = []
    for row in cursor.fetchall():
        group_id = row['group_id']
        canonical_path = row['canonical_path']
        caption = row['generated_caption'] or row['fast_caption']  # Fallback to fast caption if deep not done
        objects_json = row['detected_objects_json']
        objects = json.loads(objects_json) if objects_json else []
        
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
        SELECT canonical_path, generated_caption, detected_objects_json, extracted_tags_json
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
    tags_json = result['extracted_tags_json']
    objects = json.loads(objects_json) if objects_json else []
    tags = json.loads(tags_json) if tags_json else []
    
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
        tags=tags,
        similar_images=similar_images,
        file_size_mb=file_size_mb
    )


def get_all_objects():
    """
    Extract all unique objects from the database with their photo counts.
    Returns a dictionary mapping object names to photo counts, sorted by frequency.
    """
    conn = get_db_connection()
    cursor = conn.cursor()
    
    # Get all detected objects JSON
    cursor.execute("SELECT detected_objects_json FROM image_groups")
    
    object_counts = defaultdict(int)
    
    for row in cursor.fetchall():
        objects_json = row['detected_objects_json']
        if not objects_json:
            continue
        objects = json.loads(objects_json)
        
        # Count each unique object
        seen_in_this_photo = set()
        for obj in objects:
            label = obj['label']
            if label not in seen_in_this_photo:
                object_counts[label] += 1
                seen_in_this_photo.add(label)
    
    conn.close()
    
    # Sort by frequency (most common first)
    sorted_objects = sorted(object_counts.items(), key=lambda x: x[1], reverse=True)
    
    return sorted_objects


@app.route('/objects')
def objects_index():
    """
    Objects index - show all detected objects with photo counts.
    """
    all_objects = get_all_objects()
    
    conn = get_db_connection()
    cursor = conn.cursor()
    
    # Get stats
    cursor.execute("SELECT COUNT(*) FROM images")
    total_images = cursor.fetchone()[0]
    cursor.execute("SELECT COUNT(*) FROM image_groups")
    unique_groups = cursor.fetchone()[0]
    
    stats = {
        'total_images': total_images,
        'unique_groups': unique_groups,
        'duplicate_images': total_images - unique_groups,
        'total_objects': len(all_objects)
    }
    
    conn.close()
    
    return render_template(
        'objects.html',
        objects=all_objects,
        stats=stats
    )


@app.route('/object/<object_name>')
def object_detail(object_name):
    """
    Object detail - show all photos containing a specific object.
    """
    from urllib.parse import unquote
    
    # URL decode the object name
    object_name = unquote(object_name)
    
    conn = get_db_connection()
    cursor = conn.cursor()
    
    # Get all groups containing this object
    cursor.execute("""
        SELECT group_id, canonical_path, generated_caption, detected_objects_json
        FROM image_groups
        ORDER BY group_id
    """)
    
    matching_groups = []
    
    for row in cursor.fetchall():
        group_id = row['group_id']
        canonical_path = row['canonical_path']
        caption = row['generated_caption']
        objects_json = row['detected_objects_json']
        if not objects_json:
            continue
        objects = json.loads(objects_json)
        
        # Check if this object is in the photo
        for obj in objects:
            if obj['label'] == object_name:
                # Get count of images in this group
                cursor.execute(
                    "SELECT COUNT(*) FROM images WHERE group_id = ?",
                    (group_id,)
                )
                image_count = cursor.fetchone()[0]
                
                matching_groups.append({
                    'group_id': group_id,
                    'canonical_path': canonical_path,
                    'caption': caption,
                    'confidence': obj['score'],
                    'image_count': image_count
                })
                break
    
    conn.close()
    
    if not matching_groups:
        abort(404)
    
    return render_template(
        'object_detail.html',
        object_name=object_name,
        groups=matching_groups,
        total_count=len(matching_groups)
    )


@app.route('/persons')
def persons_index():
    """
    Persons index - show all detected persons (v2 feature).
    """
    conn = get_db_connection()
    cursor = conn.cursor()
    
    # Get all persons
    cursor.execute("""
        SELECT person_group_id, name, faiss_face_id
        FROM person_groups
        ORDER BY person_group_id
    """)
    
    persons = []
    for row in cursor.fetchall():
        person_group_id = row['person_group_id']
        name = row['name']
        
        # Count photos containing this person
        cursor.execute("""
            SELECT COUNT(DISTINCT i.group_id)
            FROM images i
            JOIN image_groups g ON i.group_id = g.group_id
            WHERE g.detected_objects_json LIKE ?
        """, (f'%"person_group_id": "{person_group_id}"%',))
        
        photo_count = cursor.fetchone()[0]
        
        if photo_count > 0:
            persons.append({
                'person_group_id': person_group_id,
                'name': name,
                'photo_count': photo_count
            })
    
    # Get stats
    cursor.execute("SELECT COUNT(*) FROM images")
    total_images = cursor.fetchone()[0]
    cursor.execute("SELECT COUNT(*) FROM person_groups")
    total_persons = cursor.fetchone()[0]
    
    stats = {
        'total_images': total_images,
        'total_persons': total_persons
    }
    
    conn.close()
    
    return render_template(
        'persons.html',
        persons=persons,
        stats=stats
    )


@app.route('/person/<person_group_id>')
def person_detail(person_group_id):
    """
    Person detail - show all photos containing a specific person (v2 feature).
    """
    from urllib.parse import unquote
    
    person_group_id = unquote(person_group_id)
    
    conn = get_db_connection()
    cursor = conn.cursor()
    
    # Get person info
    cursor.execute("""
        SELECT name FROM person_groups WHERE person_group_id = ?
    """, (person_group_id,))
    
    result = cursor.fetchone()
    if not result:
        conn.close()
        abort(404)
    
    person_name = result['name']
    
    # Get all groups containing this person
    cursor.execute("""
        SELECT group_id, canonical_path, generated_caption, detected_objects_json
        FROM image_groups
        ORDER BY group_id
    """)
    
    matching_groups = []
    
    for row in cursor.fetchall():
        group_id = row['group_id']
        canonical_path = row['canonical_path']
        caption = row['generated_caption']
        objects_json = row['detected_objects_json']
        if not objects_json:
            continue
        objects = json.loads(objects_json)
        
        # Check if this person is in the photo
        for obj in objects:
            if obj.get('person_group_id') == person_group_id:
                # Get count of images in this group
                cursor.execute(
                    "SELECT COUNT(*) FROM images WHERE group_id = ?",
                    (group_id,)
                )
                image_count = cursor.fetchone()[0]
                
                matching_groups.append({
                    'group_id': group_id,
                    'canonical_path': canonical_path,
                    'caption': caption,
                    'image_count': image_count
                })
                break
    
    conn.close()
    
    return render_template(
        'person_detail.html',
        person_group_id=person_group_id,
        person_name=person_name,
        groups=matching_groups,
        total_count=len(matching_groups)
    )


@app.route('/search')
def search():
    """
    Search across photos by captions, objects, and file paths.
    Returns results grouped by type.
    """
    from flask import request
    
    query = request.args.get('q', '').strip()
    
    if not query:
        # Redirect to gallery if no query
        from flask import redirect, url_for
        return redirect(url_for('gallery'))
    
    conn = get_db_connection()
    cursor = conn.cursor()
    
    # Search in captions
    cursor.execute("""
        SELECT group_id, canonical_path, generated_caption, detected_objects_json
        FROM image_groups
        WHERE LOWER(generated_caption) LIKE LOWER(?)
        LIMIT 50
    """, (f'%{query}%',))
    
    photo_results = []
    for row in cursor.fetchall():
        group_id = row['group_id']
        canonical_path = row['canonical_path']
        caption = row['generated_caption'] or row['fast_caption']
        objects_json = row['detected_objects_json']
        objects = json.loads(objects_json) if objects_json else []
        
        # Get count of images in this group
        cursor.execute(
            "SELECT COUNT(*) FROM images WHERE group_id = ?",
            (group_id,)
        )
        image_count = cursor.fetchone()[0]
        
        photo_results.append({
            'group_id': group_id,
            'canonical_path': canonical_path,
            'caption': caption,
            'object_count': len(objects),
            'image_count': image_count,
            'match_reason': 'Caption'
        })
    
    # Search in detected objects
    cursor.execute("""
        SELECT group_id, canonical_path, generated_caption, detected_objects_json
        FROM image_groups
        WHERE LOWER(detected_objects_json) LIKE LOWER(?)
        LIMIT 50
    """, (f'%"{query}%',))
    
    # Track group IDs we've already added to avoid duplicates
    seen_groups = {p['group_id'] for p in photo_results}
    
    for row in cursor.fetchall():
        group_id = row['group_id']
        
        # Skip if already in results from caption search
        if group_id in seen_groups:
            continue
            
        canonical_path = row['canonical_path']
        caption = row['generated_caption'] or row['fast_caption']
        objects_json = row['detected_objects_json']
        if not objects_json:
            continue
        objects = json.loads(objects_json)
        
        # Verify the object actually contains the search term
        matching_objects = [obj for obj in objects if query.lower() in obj['label'].lower()]
        if not matching_objects:
            continue
        
        # Get count of images in this group
        cursor.execute(
            "SELECT COUNT(*) FROM images WHERE group_id = ?",
            (group_id,)
        )
        image_count = cursor.fetchone()[0]
        
        photo_results.append({
            'group_id': group_id,
            'canonical_path': canonical_path,
            'caption': caption,
            'object_count': len(objects),
            'image_count': image_count,
            'match_reason': f'Object: {matching_objects[0]["label"]}'
        })
        seen_groups.add(group_id)
    
    # Find matching object names for the objects section
    cursor.execute("""
        SELECT detected_objects_json
        FROM image_groups
    """)
    
    matching_objects = {}
    for row in cursor.fetchall():
        objects_json = row['detected_objects_json']
        if not objects_json:
            continue
        objects = json.loads(objects_json)
        
        for obj in objects:
            label = obj['label']
            if query.lower() in label.lower():
                if label not in matching_objects:
                    matching_objects[label] = 0
                matching_objects[label] += 1
    
    # Sort objects by count
    object_results = sorted(matching_objects.items(), key=lambda x: x[1], reverse=True)[:10]
    
    # Search in tags (v2 feature)
    cursor.execute("""
        SELECT group_id, canonical_path, generated_caption, detected_objects_json, extracted_tags_json
        FROM image_groups
        WHERE extracted_tags_json IS NOT NULL
    """)
    
    for row in cursor.fetchall():
        group_id = row['group_id']
        
        # Skip if already in results
        if group_id in seen_groups:
            continue
            
        tags_json = row['extracted_tags_json']
        if not tags_json:
            continue
            
        tags = json.loads(tags_json)
        
        # Check if any tag matches the query
        if any(query.lower() in tag.lower() for tag in tags):
            canonical_path = row['canonical_path']
            caption = row['generated_caption'] or row['fast_caption']
            objects_json = row['detected_objects_json']
            objects = json.loads(objects_json) if objects_json else []
            
            # Get count of images in this group
            cursor.execute(
                "SELECT COUNT(*) FROM images WHERE group_id = ?",
                (group_id,)
            )
            image_count = cursor.fetchone()[0]
            
            matching_tags = [t for t in tags if query.lower() in t.lower()]
            
            photo_results.append({
                'group_id': group_id,
                'canonical_path': canonical_path,
                'caption': caption,
                'object_count': len(objects),
                'image_count': image_count,
                'match_reason': f'Tag: {matching_tags[0]}'
            })
            seen_groups.add(group_id)
            
            if len(photo_results) >= 50:
                break
    
    conn.close()
    
    return render_template(
        'search_results.html',
        query=query,
        photo_results=photo_results,
        object_results=object_results,
        total_photos=len(photo_results),
        total_objects=len(object_results)
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
