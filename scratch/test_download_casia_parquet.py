"""
Test loading parquet file from SaffalPoosh/casia_web_face on Hugging Face.
"""
import urllib.request
import pandas as pd
import io

def test_parquet():
    url = "https://huggingface.co/datasets/SaffalPoosh/casia_web_face/resolve/main/data/train-00000-of-00020.parquet"
    print(f"Downloading parquet chunk from {url}...")
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=30) as resp:
        buffer = resp.read()
        print(f"Downloaded {len(buffer)} bytes!")
        df = pd.read_parquet(io.BytesIO(buffer))
        print("Parquet columns:", df.columns.tolist())
        print("Shape:", df.shape)
        print("Sample head:")
        print(df[['image', 'label']].head() if 'label' in df.columns else df.head())

if __name__ == "__main__":
    test_parquet()
