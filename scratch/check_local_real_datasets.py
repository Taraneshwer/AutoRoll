"""
Inspect data subdirectories for real face dataset files.
"""
import os

def check_dirs():
    for root, dirs, files in os.walk("data"):
        if len(files) > 0 and not "aligned" in root and not "splits" in root and not "raw\\identity_" in root:
            print(f"{root} -> {len(files)} files, sample: {files[:5]}")

if __name__ == "__main__":
    check_dirs()
