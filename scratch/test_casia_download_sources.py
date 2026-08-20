"""
Search and verify candidate public download sources for CASIA-WebFace.
"""
import urllib.request
import json
import ssl

def check_url(url, name=""):
    print(f"Testing {name}: {url[:80]}...")
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    ctx = ssl.create_default_context()
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE
    try:
        with urllib.request.urlopen(req, timeout=10, context=ctx) as resp:
            content_type = resp.headers.get('Content-Type', '')
            content_len = resp.headers.get('Content-Length', 'Unknown')
            print(f"  -> SUCCESS! Status: {resp.status}, Type: {content_type}, Length: {content_len}")
            return True
    except Exception as e:
        print(f"  -> FAILED: {e}")
        return False

def main():
    print("Checking CASIA-WebFace source URLs...")
    candidate_urls = [
        ("Kaggle CASIA-WebFace direct mirror", "https://huggingface.co/datasets/pyskl/casia-webface/resolve/main/casia-webface.zip"),
        ("HuggingFace CASIA-WebFace subset", "https://huggingface.co/datasets/minimaxir/casia-webface/resolve/main/casia.zip"),
        ("GitHub Release CASIA-WebFace archive", "https://github.com/yakhyo/face-recognition/releases/download/v0.0.1/casia-webface.zip"),
        ("InsightFace Dataset Zoo CASIA mirror", "https://huggingface.co/datasets/deepinsight/casia-webface/resolve/main/casia-webface.zip"),
    ]

    for name, url in candidate_urls:
        check_url(url, name)

if __name__ == "__main__":
    main()
