"""
Test direct file download access to CASIA-WebFace Hugging Face repos.
"""
import urllib.request
import ssl
import json

def test_file(url, desc=""):
    print(f"Testing [{desc}]: {url}")
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    ctx = ssl.create_default_context()
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE
    try:
        with urllib.request.urlopen(req, timeout=15, context=ctx) as resp:
            content_len = resp.headers.get('Content-Length', 'Unknown')
            content_type = resp.headers.get('Content-Type', '')
            print(f"  -> SUCCESS! Status: {resp.status}, Type: {content_type}, Length: {content_len}")
            return True
    except Exception as e:
        print(f"  -> FAILED: {e}")
        return False

def main():
    test_file("https://huggingface.co/datasets/Bill13579/casia-mirror/resolve/main/CASIA.zip", "Bill13579 CASIA.zip")
    test_file("https://huggingface.co/datasets/SaffalPoosh/casia_web_face/tree/main", "SaffalPoosh tree")
    test_file("https://huggingface.co/datasets/Pijush22049/casia-webface/tree/main", "Pijush22049 tree")

if __name__ == "__main__":
    main()
