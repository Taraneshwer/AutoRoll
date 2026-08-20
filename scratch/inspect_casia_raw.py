"""
Inspect extracted CASIA dataset contents.
"""
import os

def inspect():
    root = "data/raw_datasets/casia_webface_raw"
    print(f"Listing contents of '{root}':")
    for r, dirs, files in os.walk(root):
        print(f"  {r} -> {len(dirs)} subdirs, {len(files)} files. Sample files: {files[:5]}")

if __name__ == "__main__":
    inspect()
