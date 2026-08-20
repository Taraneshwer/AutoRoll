"""
Script to test multiple public CASIA-WebFace dataset mirror URLs.
"""
import urllib.request
import ssl
import json

def test_url(url, description=""):
    print(f"Testing [{description}]: {url}")
    req = urllib.request.Request(url, headers={
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    })
    ctx = ssl.create_default_context()
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE
    try:
        with urllib.request.urlopen(req, timeout=10, context=ctx) as resp:
            print(f"  [SUCCESS] Code: {resp.status}, Content-Length: {resp.headers.get('Content-Length')}, Content-Type: {resp.headers.get('Content-Type')}")
            return True
    except Exception as e:
        print(f"  [FAILED] {e}")
        return False

def main():
    urls = [
        ("Kaggle CASIA-WebFace raw download link", "https://www.kaggle.com/api/v1/datasets/download/saisirish/casia-webface"),
        ("GitHub Release CASIA 112x112 mirror", "https://github.com/mk-minchul/AdaFace/releases/download/v1.0/casia_webface.zip"),
        ("InsightFace GitHub Release CASIA", "https://github.com/deepinsight/insightface/releases/download/v0.0.0/casia.zip"),
        ("GitHub Release face.evoLVe CASIA", "https://github.com/ZhaoJ9014/face.evoLVe/releases/download/v1.0/casia-webface.zip"),
        ("HuggingFace direct LFW/CASIA raw zip", "https://huggingface.co/datasets/vumc/casia-webface/resolve/main/casia-webface.zip"),
        ("OpenDataLab CASIA-WebFace mirror", "https://opendatalab.com/CASIA-WebFace"),
    ]

    for desc, url in urls:
        test_url(url, desc)

if __name__ == "__main__":
    main()
