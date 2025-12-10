# pipeline_code/fetch_images.py
import os
import math
import json
import time
import requests
import pandas as pd
from io import BytesIO
from PIL import Image
from tqdm import tqdm

# ------------------ CONFIG ------------------
INPUT_XLSX = r"C:\Users\samru\OneDrive\Desktop\eco-ideathon\input\coords.xlsx"
OUT_IMG_DIR = r"C:\Users\samru\OneDrive\Desktop\eco-ideathon\output\images"
OUT_META_DIR = r"C:\Users\samru\OneDrive\Desktop\eco-ideathon\output\metadata"

MAPBOX_TOKEN = "pk.eyJ1Ijoic2FtcnVkZGhpLTA1MTEiLCJhIjoiY21pdzFtNTFzMDBpeDNnczV3ajdwNjQwdSJ9.4fjp6MtMnxK-tqRvs3nqog"
 # <-- paste your pk.xxxxx token here
IMG_SIZE_PX = 1024
ZOOM = 19          # 18–20 works; 19 is a good balance
SLEEP_BETWEEN = 0.3

# ------------------ UTILS ------------------
def ensure_dirs():
    os.makedirs(OUT_IMG_DIR, exist_ok=True)
    os.makedirs(OUT_META_DIR, exist_ok=True)

def sample_name(sample_id, lat, lon):
    return f"{sample_id}_{lat:.6f}_{lon:.6f}"

# ------------------ MAPBOX URL GENERATOR ------------------
def mapbox_satellite_url(lat, lon, zoom=ZOOM, size=IMG_SIZE_PX):
    return (
        f"https://api.mapbox.com/styles/v1/mapbox/satellite-v9/static/"
        f"{lon},{lat},{zoom}/{size}x{size}?access_token={MAPBOX_TOKEN}"
    )

# ------------------ FETCH FUNCTION ------------------
def fetch_mapbox_image(lat, lon):
    url = mapbox_satellite_url(lat, lon)
    resp = requests.get(url, timeout=30)

    if resp.status_code != 200:
        raise RuntimeError(f"Mapbox error {resp.status_code}: {resp.text[:200]}")

    img = Image.open(BytesIO(resp.content)).convert("RGB")

    meta = {
        "source": "Mapbox Satellite",
        "url": url,
        "status_code": resp.status_code
    }
    return img, meta

# ------------------ PROCESS ROW ------------------
def process_row(sample_id, lat, lon):
    try:
        img, meta = fetch_mapbox_image(lat, lon)
    except Exception as e:
        return {
            "sample_id": sample_id,
            "lat": lat,
            "lon": lon,
            "error": str(e)
        }

    base = sample_name(sample_id, lat, lon)
    img_path = os.path.join(OUT_IMG_DIR, base + ".png")
    meta_path = os.path.join(OUT_META_DIR, base + ".json")

    img.save(img_path)
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)

    return {
        "sample_id": sample_id,
        "lat": lat,
        "lon": lon,
        "image_path": img_path,
        "meta_path": meta_path,
        "success": True
    }

# ------------------ MAIN ------------------
def main():
    ensure_dirs()
    df = pd.read_excel(INPUT_XLSX)

    results = []

    for _, row in tqdm(df.iterrows(), total=len(df)):
        sid = row["sample_id"]
        lat = float(row["latitude"])
        lon = float(row["longitude"])

        result = process_row(sid, lat, lon)
        results.append(result)
        time.sleep(SLEEP_BETWEEN)

    with open(os.path.join(OUT_META_DIR, "fetch_summary.json"), "w") as f:
        json.dump(results, f, indent=2)

if __name__ == "__main__":
    main()
