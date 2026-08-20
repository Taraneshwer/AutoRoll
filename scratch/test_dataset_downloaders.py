"""
Test python packages and datasets library for CASIA-WebFace download access.
"""
import urllib.request
import json
import ssl

def check_hf_hub_direct():
    print("Searching HuggingFace hub dataset repositories for public casia files...")
    # Query datasets API
    url = "https://huggingface.co/api/datasets?search=casia&limit=30"
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            items = json.loads(resp.read().decode())
            print(f"Found {len(items)} dataset candidates:")
            for it in items:
                repo_id = it['id']
                print(f"  Repo: {repo_id}")
                # Check tree for raw zip or tar
                tree_url = f"https://huggingface.co/api/datasets/{repo_id}/tree/main"
                try:
                    t_req = urllib.request.Request(tree_url, headers={"User-Agent": "Mozilla/5.0"})
                    with urllib.request.urlopen(t_req, timeout=5) as t_resp:
                        t_files = json.loads(t_resp.read().decode())
                        paths = [f.get('path', '') for f in t_files if isinstance(f, dict)]
                        print(f"    Paths: {paths[:8]}")
                except Exception as e:
                    print(f"    Error reading tree: {e}")
    except Exception as e:
        print(f"HF query failed: {e}")

if __name__ == "__main__":
    check_hf_hub_direct()
