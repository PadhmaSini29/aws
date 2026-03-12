# reset_collections.py
"""
One-time reset for all PGVector collections.
USE ONLY FOR LOCAL DEV.
"""

from backend.vectorstore import delete_collection
from backend.config import settings

def reset_all():
    print("🔄 Resetting PGVector collections...\n")

    collections = [
        settings.collection_text,
        settings.collection_images,
        settings.collection_pdf_images,
        settings.collection_audio,
        settings.collection_video_text,
        settings.collection_video_frames,
    ]

    for c in collections:
        delete_collection(c)

    print("\n✅ Reset complete")

if __name__ == "__main__":
    reset_all()
