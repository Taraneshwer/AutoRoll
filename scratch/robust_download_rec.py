"""
Robust multi-resume downloader for CASIA-WebFace train.rec.
Automatically retries/resumes on connection drops until the full file is downloaded.
"""
import urllib.request
import os
import sys
import time

URL = "https://huggingface.co/datasets/Pijush22049/casia-webface/resolve/main/data/train.rec"
DEST = "data/tmp/casia_webface/train.rec"
EXPECTED_MB = 2599
MAX_ATTEMPTS = 50
CHUNK_SIZE = 4 * 1024 * 1024  # 4 MB chunks


def get_file_size():
    return os.path.getsize(DEST) if os.path.exists(DEST) else 0


def download_chunk(start_byte):
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)",
        "Range": f"bytes={start_byte}-",
    }
    req = urllib.request.Request(URL, headers=headers)
    resp = urllib.request.urlopen(req, timeout=120)
    status = resp.status
    content_length = int(resp.headers.get("Content-Length", 0))
    total = start_byte + content_length
    return resp, status, content_length, total


def main():
    expected_bytes = EXPECTED_MB * 1024 * 1024
    os.makedirs(os.path.dirname(DEST), exist_ok=True)

    print("=" * 65)
    print("Robust CASIA train.rec Multi-Resume Downloader")
    print(f"Target: {EXPECTED_MB} MB")
    print("=" * 65)

    for attempt in range(1, MAX_ATTEMPTS + 1):
        current_size = get_file_size()
        if current_size >= expected_bytes:
            print(f"\n[COMPLETE] File is {current_size/(1024*1024):.1f} MB — download finished!")
            break

        remaining_mb = (expected_bytes - current_size) / (1024 * 1024)
        print(f"\nAttempt {attempt}: resuming from {current_size/(1024*1024):.1f} MB "
              f"({remaining_mb:.1f} MB remaining)...")

        try:
            resp, status, content_length, total = download_chunk(current_size)

            if status not in (200, 206):
                print(f"  Unexpected status {status}, waiting 10s...")
                time.sleep(10)
                continue

            if status == 200 and current_size > 0:
                print(f"  Server doesn't support resume (200), restarting from 0...")
                current_size = 0

            t0 = time.time()
            downloaded_this_session = 0

            with open(DEST, "ab" if current_size > 0 else "wb") as f:
                while True:
                    chunk = resp.read(CHUNK_SIZE)
                    if not chunk:
                        break
                    f.write(chunk)
                    downloaded_this_session += len(chunk)
                    total_now = current_size + downloaded_this_session
                    elapsed = time.time() - t0
                    speed = downloaded_this_session / elapsed / (1024 * 1024) if elapsed > 0 else 0
                    pct = (total_now / expected_bytes) * 100
                    sys.stdout.write(
                        f"\r  {total_now/(1024*1024):.1f}/{EXPECTED_MB} MB "
                        f"({pct:.1f}%) | {speed:.2f} MB/s"
                    )
                    sys.stdout.flush()

            session_mb = downloaded_this_session / (1024 * 1024)
            elapsed = time.time() - t0
            print(f"\n  Session: +{session_mb:.1f} MB in {elapsed:.0f}s")

        except Exception as e:
            print(f"\n  Connection error: {e}")
            print(f"  Waiting 15s before retry...")
            time.sleep(15)
            continue

    final_size = get_file_size()
    print(f"\nFinal file size: {final_size/(1024*1024):.1f} MB")
    if final_size >= expected_bytes:
        print("[SUCCESS] train.rec fully downloaded.")
        sys.exit(0)
    else:
        print(f"[INCOMPLETE] Only {final_size/(1024*1024):.1f} MB downloaded after {MAX_ATTEMPTS} attempts.")
        sys.exit(1)


if __name__ == "__main__":
    main()
