"""
Export ChromaDB to JSON
========================

Simple script to export all ChromaDB data to JSON file.

Usage:
    python src/inspect_chromadb.py
"""

from pathlib import Path
import sys


sys.path.append(str(Path(__file__).parent.parent))

import chromadb
import json
from config import Config

# Connect to ChromaDB
client = chromadb.PersistentClient(path=Config.CHROMA_DB_PATH)

# Get all collections
collections = client.list_collections()

print("=" * 70)
print("EXPORTING CHROMADB TO JSON")
print("=" * 70)

export_data = {}
total_docs = 0

for collection_obj in collections:
    collection_name = collection_obj.name
    collection = client.get_collection(collection_name)
    
    count = collection.count()
    total_docs += count
    
    print(f"\nExporting: {collection_name} ({count} documents)")
    
    if count > 0:
        results = collection.get(include=['documents', 'metadatas'])
        
        export_data[collection_name] = {
            'count': count,
            'documents': [
                {
                    'id': doc_id,
                    'text': text,
                    'metadata': metadata
                }
                for doc_id, text, metadata in zip(
                    results['ids'],
                    results['documents'],
                    results['metadatas']
                )
            ]
        }

# Save to file in same directory as script
output_file = 'chromadb_export.json'

with open(output_file, 'w', encoding='utf-8') as f:
    json.dump(export_data, f, indent=2, ensure_ascii=False)

print("\n" + "=" * 70)
print(f"✅ EXPORTED {total_docs} DOCUMENTS")
print(f"📁 File: {output_file}")
print("=" * 70)