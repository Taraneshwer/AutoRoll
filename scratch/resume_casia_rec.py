"""
Resume download of CASIA-WebFace train.rec using HTTP Range requests.
Continues from where the previous download stopped.
"""
import urllib.request
import os
import sys
import time

def resume_download(url, dest_path, expected_size_mb=2600):
    existing_size = os.path.getsize(dest_path) if os.path.exists(dest_path) else 0
    expected_bytes = expected_size_mb * 1024 * 1024
    
    if existing_size >= expected_bytes:
        print(f"File already complete: {existing_size / (1024*1024):.1f} MB")
        return True
    
    print(f"Resuming from byte {existing_size} ({existing_size / (1024*1024):.1f} MB)...")
    
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)",
        "Range": f"bytes={existing_size}-",
    }
    req = urllib.request.Request(url, headers=headers)
    
    t0 = time.time()
    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            status = resp.status
            content_range = resp.headers.get("Content-Range", "")
            content_length = int(resp.headers.get("Content-Length", 0))
            print(f"Response status: {status}")
            print(f"Content-Range: {content_range}")
            print(f"Content-Length: {content_length / (1024*1024):.1f} MB")
            
            if status not in (206, 200):
                print(f"ERROR: Server returned status {status} instead of 206 Partial Content")
                return False
            
            if status == 200 and existing_size > 0:
                print("WARNING: Server does not support resume (returned 200 instead of 206).")
                print("Will overwrite from start...")
                existing_size = 0
            
            mode = "ab" if existing_size > 0 else "wb"
            downloaded = existing_size
            total = existing_size + content_length if content_length > 0 else 0
            
            with open(dest_path, mode) as f:
                chunk_size = 1024 * 1024
                while chunk := resp.read(chunk_size):
                    f.write(chunk)
                    downloaded += len(chunk)
                    elapsed = time.time() - t0
                    speed = (downloaded - existing_size) / elapsed / (1024 * 1024) if elapsed > 0 else 0
                    if total > 0:
                        pct = (downloaded / total) * 100
                        sys.stdout.write(f"\rProgress: {downloaded/(1024*1024):.1f}/{total/(1024*1024):.1f} MB ({pct:.1f}%) | {speed:.2f} MB/s")
                    else:
                        sys.stdout.write(f"\rDownloaded: {downloaded/(1024*1024):.1f} MB | {speed:.2f} MB/s")
                    sys.stdout.flush()
            
            print(f"\nDownload finished. Total size: {os.path.getsize(dest_path) / (1024*1024):.1f} MB")
            return True
    except Exception as e:
        print(f"\nERROR: {e}")
        return False


def main():
    base_url = "https://huggingface.co/datasets/Pijush22049/casia-webface/resolve/main/data"
    dest_dir = "data/tmp/casia_webface"
    os.makedirs(dest_dir, exist_ok=True)
    
    rec_url = f"{base_url}/train.rec"
    rec_path = os.path.join(dest_dir, "train.rec")
    
    print(f"Target: {rec_url}")
    print(f"Destination: {rec_path}")
    print(f"Current size: {os.path.getsize(rec_path) / (1024*1024):.1f} MB" if os.path.exists(rec_path) else "File does not exist")
    print()
    
    success = resume_download(rec_url, rec_path)
    if success:
        final_size = os.path.getsize(rec_path) / (1024 * 1024)
        print(f"\nFinal file size: {final_size:.1f} MB")
    else:
        print("\nResume download failed.")
        sys.exit(1)


if __name__ == "__main__":
    main()
