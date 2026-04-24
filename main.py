from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import numpy as np
import cv2
import requests
import tempfile
import os
from PIL import Image
import io
import base64
import gc
import traceback
import time

app = FastAPI(title="HDR Merge Service")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

OUTPUT_WIDTH  = 2048
OUTPUT_HEIGHT = 1536

PHOTOMATIX_API = "https://api.hdrsoft.com"


# ---------------------------------------------------------------------------
# Photomatix API — HDR merge via cloud
# ---------------------------------------------------------------------------

def photomatix_merge(file_urls: List[str], api_key: str) -> np.ndarray:
    headers = {"x-pm-token": api_key}

    # Step 1: Create HDR engine
    print("Photomatix: creating HDR engine...")
    r = requests.post(
        f"{PHOTOMATIX_API}/hdrengines",
        headers=headers,
        params={
            "type": "multi",
            "alignment": "yes",
            "deghosting": "on",
            "noise-reduction": "underexposed",
            "lens-correction": "yes",
            "output-bit-depth": "8",
        },
        timeout=30,
    )
    if r.status_code not in (200, 201):
        raise Exception(f"Photomatix engine creation failed ({r.status_code}): {r.text[:300]}")

    data = r.json()
    engine_uri = data.get("data", {}).get("location") or data.get("location")
    if not engine_uri:
        raise Exception(f"No engine URI returned: {data}")
    print(f"Photomatix engine: {engine_uri}")

    # Step 2: Add images by URL
    for i, url in enumerate(file_urls):
        base = url.split("?")[0].rsplit("/", 1)[-1]
        ext = base.rsplit(".", 1)[-1].lower() if "." in base else "jpg"
        filename = f"image_{i+1}.{ext}"
        print(f"Photomatix: adding image {filename} from URL...")
        add_r = requests.post(
            f"{PHOTOMATIX_API}{engine_uri}/images/{filename}",
            headers=headers,
            data={"url": url},
            timeout=60,
        )
        if add_r.status_code not in (200, 201):
            raise Exception(f"Photomatix add image failed ({add_r.status_code}): {add_r.text[:300]}")

    # Step 3: Process with Real Estate preset
    print("Photomatix: processing with Real Estate preset...")
    process_r = requests.post(
        f"{PHOTOMATIX_API}{engine_uri}/processed/preset",
        headers=headers,
        params={
            "preset": "Real Estate",
            "output-format": "jpeg",
            "output-bit-depth": "8",
        },
        timeout=180,
    )
    if process_r.status_code not in (200, 201):
        raise Exception(f"Photomatix process failed ({process_r.status_code}): {process_r.text[:300]}")

    process_data = process_r.json()
    result_url = process_data.get("data", {}).get("location") or process_data.get("location")
    if not result_url:
        raise Exception(f"No result URL from Photomatix: {process_data}")

    print(f"Photomatix: downloading result from {result_url}...")
    for attempt in range(30):
        dl = requests.get(f"{PHOTOMATIX_API}{result_url}", headers=headers, timeout=60)
        if dl.status_code == 200:
            break
        elif dl.status_code == 202:
            print(f"Photomatix: still processing... attempt {attempt+1}")
            time.sleep(5)
        else:
            raise Exception(f"Photomatix download failed ({dl.status_code}): {dl.text[:300]}")
    else:
        raise Exception("Photomatix processing timed out after 150s")

    img_arr = np.frombuffer(dl.content, np.uint8)
    img = cv2.imdecode(img_arr, cv2.IMREAD_COLOR)
    if img is None:
        raise Exception("Could not decode Photomatix result image")

    print(f"Photomatix merge done: {img.shape[1]}x{img.shape[0]}")
    return img


# ---------------------------------------------------------------------------
# Fallback: local RAW loading + Mertens (used if no Photomatix key)
# ---------------------------------------------------------------------------

def download_file(url: str, suffix: str) -> str:
    with requests.get(url, timeout=60, stream=True, verify=False) as r:
        r.raise_for_status()
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix, dir="/tmp")
        for chunk in r.iter_content(chunk_size=1024 * 1024):
            tmp.write(chunk)
        tmp.close()
    return tmp.name


def decode_raw(path: str) -> np.ndarray:
    import rawpy
    with rawpy.imread(path) as raw:
        rgb = raw.postprocess(use_camera_wb=True, no_auto_bright=False, output_bps=8, half_size=True)
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    del rgb
    gc.collect()
    return cv2.resize(bgr, (OUTPUT_WIDTH, OUTPUT_HEIGHT), interpolation=cv2.INTER_LANCZOS4)


def load_image_bgr(path: str) -> np.ndarray:
    ext = os.path.splitext(path)[1].lower()
    raw_exts = {".cr3", ".cr2", ".nef", ".arw", ".dng", ".raf", ".rw2", ".orf", ".raw"}
    if ext in raw_exts:
        return decode_raw(path)
    img = cv2.imread(path)
    if img is None:
        raise ValueError(f"Could not read image: {path}")
    return cv2.resize(img, (OUTPUT_WIDTH, OUTPUT_HEIGHT), interpolation=cv2.INTER_LANCZOS4)


def local_merge(file_urls: List[str]) -> np.ndarray:
    """Download, decode RAW, Mertens merge. Fallback when no Photomatix key."""
    tmp_paths = []
    try:
        for url in file_urls:
            ext = url.split("?")[0].rsplit(".", 1)[-1]
            ext = f".{ext.lower()}" if ext else ".jpg"
            tmp_paths.append(download_file(url, ext))

        images = [load_image_bgr(p) for p in tmp_paths]
        gc.collect()

        if len(images) > 1:
            fused = cv2.createMergeMertens(
                contrast_weight=1.4, saturation_weight=0.9, exposure_weight=0.2
            ).process(images)
            merged = np.clip(fused * 255, 0, 255).astype(np.uint8)
            del fused
        else:
            merged = images[0]

        gc.collect()
        return merged
    finally:
        for p in tmp_paths:
            try:
                os.unlink(p)
            except Exception:
                pass


# ---------------------------------------------------------------------------
# Request model & endpoint
# ---------------------------------------------------------------------------

class MergeRequest(BaseModel):
    file_urls: List[str]
    bracket_name: str = "bracket"
    imagine_api_key: Optional[str] = None
    replace_sky: bool = False


@app.post("/merge")
async def merge_hdr(req: MergeRequest):
    if not req.file_urls:
        raise HTTPException(400, "No file URLs provided")
    if len(req.file_urls) not in (1, 3, 5):
        raise HTTPException(400, f"Expected 1, 3, or 5 files, got {len(req.file_urls)}")

    photomatix_key = os.environ.get("PHOTOMATIX_API_KEY")

    try:
        if photomatix_key:
            print("Using Photomatix API for HDR merge...")
            merged = photomatix_merge(req.file_urls, photomatix_key)
            merged = cv2.resize(merged, (OUTPUT_WIDTH, OUTPUT_HEIGHT), interpolation=cv2.INTER_LANCZOS4)
        else:
            print("No Photomatix key — falling back to local Mertens merge...")
            merged = local_merge(req.file_urls)

        # Encode directly — no post-processing
        pil = Image.fromarray(cv2.cvtColor(merged, cv2.COLOR_BGR2RGB))
        del merged
        gc.collect()

        buf = io.BytesIO()
        pil.save(buf, format="JPEG", quality=94, optimize=True)
        del pil
        gc.collect()

        buf.seek(0)
        jpg_b64 = base64.b64encode(buf.read()).decode("utf-8")
        del buf

        return {
            "success": True,
            "bracket_name": req.bracket_name,
            "width": OUTPUT_WIDTH,
            "height": OUTPUT_HEIGHT,
            "jpeg_base64": jpg_b64,
        }

    except Exception as e:
        tb = traceback.format_exc()
        print(f"ERROR in /merge: {tb}")
        raise HTTPException(500, detail=f"{str(e)}\n\nTraceback:\n{tb}")


@app.get("/health")
def health():
    return {"status": "ok"}
