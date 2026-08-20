"""
Download CASIA-WebFace train.rec, train.idx, train.lst dataset from Hugging Face repository Pijush22049/casia-webface.
"""
import urllib.request
import os
import sys
import time

def download_file(url, dest_path):
    print(f"Downloading '{url}' -> '{dest_path}'...")
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"})
    t0 = time.time()
    with urllib.request.urlopen(req) as resp, open(dest_path, "wb") as f:
        total = int(resp.headers.get("Content-Length", 0))
        downloaded = 0
        chunk_size = 1024 * 1024
        while chunk := resp.read(chunk_size):
            f.write(chunk)
            downloaded += len(chunk)
            if total > 0:
                pct = (downloaded / total) * 100
                speed = downloaded / (time.time() - t0) / (1024 * 1024)
                sys.stdout.write(f"\rProgress: {downloaded / (1024*1024):.1f}/{total / (1024*1024):.1f} MB ({pct:.1f}%) | {speed:.2f} MB/s")
                sys.stdout.flush()
    print(f"\nDownload finished in {time.time() - t0:.1f}s.")

def main():
    base_url = "https://huggingface.co/datasets/Pijush22049/casia-webface/resolve/main/data"
    dest_dir = "data/tmp/casia_webface"
    os.makedirs(dest_dir, exist_ok=True)

    files = ["train.idx", "train.lst", "train.rec"]
    for fname in files:
        f_url = f"{base_url}/{fname}"
        f_dest = os.path.join(dest_dir, fname)
        if not os.path.exists(f_dest) or os.path.getsize(f_dest) == 0:
            download_file(f_url, f_dest)
        else:
            print(f"File '{f_dest}' already exists ({os.path.getsize(f_dest)} bytes). Skipping download.")

if __name__ == "__main__":
    main()
