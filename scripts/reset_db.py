#!/usr/bin/env python3
"""Reset Helix database by deleting all documents."""

import sys
sys.path.insert(0, ".")

from src.storage import HelixChunkStore, HelixConfig

def main():
    config = HelixConfig(local=True, port=6969)
    store = HelixChunkStore(config)

    docs = store.list_documents()
    print(f"Found {len(docs)} documents to delete")

    if not docs:
        print("Database is already empty")
        return

    # Get unique doc_ids (some may be duplicated)
    unique_doc_ids = set(d.get("doc_id") for d in docs)
    print(f"Unique document IDs: {len(unique_doc_ids)}")

    for doc_id in unique_doc_ids:
        print(f"Deleting: {doc_id}")
        try:
            store.delete_document(doc_id)
        except Exception as e:
            print(f"  Error: {e}")

    # Verify
    remaining = store.list_documents()
    print(f"\nRemaining documents: {len(remaining)}")

if __name__ == "__main__":
    main()
