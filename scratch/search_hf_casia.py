"""
Search HuggingFace hub for CASIA-WebFace dataset entries.
"""
import urllib.request
import json

def search_hf_datasets(query):
    url = f"https://huggingface.co/api/datasets?search={query}&limit=20"
    print(f"Searching HuggingFace API for '{query}'...")
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = json.loads(resp.read().decode())
            print(f"Found {len(data)} datasets:")
            for item in data:
                print(f"  - {item['id']} (downloads: {item.get('downloads', 0)})")
                # List files in dataset
                tree_url = f"https://huggingface.co/api/datasets/{item['id']}/tree/main"
                try:
                    t_req = urllib.request.Request(tree_url, headers={"User-Agent": "Mozilla/5.0"})
                    with urllib.request.urlopen(t_req, timeout=5) as t_resp:
                        tree_data = json.loads(t_resp.read().decode())
                        files = [f['path'] for f in tree_data if isinstance(f, dict) and 'path' in f]
                        print(f"    Files: {files[:5]}")
                except Exception as e:
                    print(f"    Tree error: {e}")
    except Exception as e:
        print(f"Search failed: {e}")

if __name__ == "__main__":
    search_hf_datasets("casia")
    search_hf_datasets("webface")
