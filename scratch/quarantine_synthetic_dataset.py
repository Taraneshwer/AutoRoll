"""
Script to quarantine Phase 5 synthetic dataset to data/quarantine/synthetic_phase5/
and clean data/face_recognition/.
"""
import os
import shutil

def quarantine():
    src_root = "data/face_recognition"
    dest_root = "data/quarantine/synthetic_phase5"

    os.makedirs(dest_root, exist_ok=True)

    items_to_move = ["aligned", "raw", "splits", "splits_dev"]
    for item in items_to_move:
        src_path = os.path.join(src_root, item)
        dest_path = os.path.join(dest_root, item)
        if os.path.exists(src_path):
            print(f"Moving '{src_path}' -> '{dest_path}'...")
            if os.path.exists(dest_path):
                shutil.rmtree(dest_path)
            shutil.move(src_path, dest_path)

    # Move synthetic manifest if present
    old_manifest = os.path.join(src_root, "metadata", "dataset_manifest.json")
    if os.path.exists(old_manifest):
        os.makedirs(os.path.join(dest_root, "metadata"), exist_ok=True)
        shutil.move(old_manifest, os.path.join(dest_root, "metadata", "dataset_manifest.json"))

    # Ensure clean data/face_recognition root structure
    os.makedirs(os.path.join(src_root, "metadata"), exist_ok=True)
    os.makedirs(os.path.join(src_root, "raw"), exist_ok=True)
    os.makedirs(os.path.join(src_root, "aligned"), exist_ok=True)
    os.makedirs(os.path.join(src_root, "splits"), exist_ok=True)

    print("Quarantine completed successfully.")

if __name__ == "__main__":
    quarantine()
