"""
Download and extract CASIA.zip from Bill13579/casia-mirror HuggingFace repository.
"""
import urllib.request
import zipfile
import os
import shutil

def download_and_extract():
    url = "https://huggingface.co/datasets/Bill13579/casia-mirror/resolve/main/CASIA.zip"
    dest_zip = "data/tmp/CASIA.zip"
    extract_dir = "data/raw_datasets/casia_webface_raw"

    os.makedirs("data/tmp", exist_ok=True)
    os.makedirs(extract_dir, exist_ok=True)

    print(f"Downloading CASIA archive from {url}...")
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req) as resp, open(dest_zip, "wb") as f:
        shutil.copyfileobj(resp, f)

    print(f"Archive downloaded ({os.path.getsize(dest_zip)} bytes). Extracting to {extract_dir}...")
    with zipfile.ZipFile(dest_zip, 'r') as zip_ref:
        zip_ref.extractall(extract_dir)

    print("Extraction complete!")
    subdirs = [d for d in os.listdir(extract_dir) if os.path.isdir(os.path.join(extract_dir, d))]
    print(f"Extracted {len(subdirs)} identity directories from {extract_dir}.")

if __name__ == "__main__":
    download_and_extract()
