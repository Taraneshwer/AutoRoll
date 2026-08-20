"""
Inspect file listings inside HF dataset mirrors for CASIA-WebFace.
"""
import urllib.request
import json

def get_tree(repo_id, path=""):
    url = f"https://huggingface.co/api/datasets/{repo_id}/tree/main/{path}"
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = json.loads(resp.read().decode())
            print(f"Tree '{repo_id}' path '{path}' ({len(data)} items):")
            for item in data[:15]:
                print(f"  {item.get('type')}: {item.get('path')} ({item.get('size', 0)} bytes)")
            return data
    except Exception as e:
        print(f"Error getting tree for {repo_id}/{path}: {e}")
        return []

if __name__ == "__main__":
    get_tree("Bill13579/casia-mirror")
    get_tree("Bill13579/casia-mirror", "data")
    get_tree("SaffalPoosh/casia_web_face")
    get_tree("SaffalPoosh/casia_web_face", "data")
    get_tree("Pijush22049/casia-webface")
    get_tree("Pijush22049/casia-webface", "data")
