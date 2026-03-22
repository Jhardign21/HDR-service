from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List
import rawpy
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

app = FastAPI(title="HDR Merge Service")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

OUTPUT_WIDTH = 2048
OUTPUT_HEIGHT = 1536


class MergeRequest(BaseModel):
    file_urls: List[str]
    bracket_name: str = "bracket"


def download_file(url: str, suffix: str) -> str:
    """Stream-download a file to /tmp, return path."""
    with requests.get(url, timeout=60, stream=True) as r:
        r.raise_for_status()
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix, dir="/tmp")
        for chunk in r.iter_content(chunk_size=1024 * 1024):
            tmp.write(chunk)
        tmp.close()
    return tmp.name


def decode_raw_to_small_rgb(path: str) -> np.ndarray:
    """Decode RAW directly to output size to avoid holding a huge array."""
    with rawpy.imread(path) as raw:
        rgb = raw.postprocess(
            use_camera_wb=True,
            no_auto_bright=False,
            output_bps=8,
            half_size=True,   # decode at half resolution first — 4x memory saving
        )
    # Immediately resize to output dims
    resized = cv2.resize(
        cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR),
        (OUTPUT_WIDTH, OUTPUT_HEIGHT),
        interpolation=cv2.INTER_LANCZOS4,
    )
    del rgb
    gc.collect()
    return resized  # BGR uint8 at output size


def load_image_bgr(path: str) -> np.ndarray:
    """Load RAW or JPEG into BGR uint8 at output size."""
    ext = os.path.splitext(path)[1].lower()
    raw_exts = {".cr3", ".cr2", ".nef", ".arw", ".dng", ".raf", ".rw2", ".orf", ".raw"}
    if ext in raw_exts:
        return decode_raw_to_small_rgb(path)
    else:
        img = cv2.imread(path)
        if img is None:
            raise ValueError(f"Could not read image: {path}")
        return cv2.resize(img, (OUTPUT_WIDTH, OUTPUT_HEIGHT), interpolation=cv2.INTER_LANCZOS4)


def align_images(images: List[np.ndarray]) -> List[np.ndarray]:
    """Align exposures using translation-only ECC (fast, low memory)."""
    if len(images) == 1:
        return images
    aligned = [images[0]]
    ref_gray = cv2.cvtColor(images[0], cv2.COLOR_BGR2GRAY)
    for img in images[1:]:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        warp = np.eye(2, 3, dtype=np.float32)
        try:
            _, warp = cv2.findTransformECC(
                ref_gray, gray, warp,
                cv2.MOTION_TRANSLATION,
                (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.001),
            )
            aligned_img = cv2.warpAffine(
                img, warp, (OUTPUT_WIDTH, OUTPUT_HEIGHT),
                flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP,
            )
            aligned.append(aligned_img)
        except Exception:
            aligned.append(img)
        del gray
    del ref_gray
    gc.collect()
    return aligned


def merge_mertens(images: List[np.ndarray]) -> np.ndarray:
    """Mertens exposure fusion."""
    merge = cv2.createMergeMertens(
        contrast_weight=1.0,
        saturation_weight=1.0,
        exposure_weight=0.0,
    )
    fused = merge.process(images)
    result = np.clip(fused * 255, 0, 255).astype(np.uint8)
    del fused
    gc.collect()
    return result


def apply_realestate_tone(img: np.ndarray) -> np.ndarray:
    """Subtle real-estate tone: lift shadows, slight warmth."""
    f = img.astype(np.float32) / 255.0
    f = np.power(f, 0.88)
    f = f * 0.92 + 0.04
    # Warm tint (BGR order)
    f[:, :, 2] = np.clip(f[:, :, 2] * 1.04, 0, 1)  # R
    f[:, :, 1] = np.clip(f[:, :, 1] * 1.02, 0, 1)  # G
    result = np.clip(f * 255, 0, 255).astype(np.uint8)
    del f
    gc.collect()
    return result


@app.post("/merge")
async def merge_hdr(req: MergeRequest):
    if not req.file_urls:
        raise HTTPException(400, "No file URLs provided")
    if len(req.file_urls) not in (1, 3, 5):
        raise HTTPException(400, f"Expected 1, 3, or 5 files, got {len(req.file_urls)}")

    tmp_paths = []
    try:
        # Download all files
        for url in req.file_urls:
            ext = url.split("?")[0].rsplit(".", 1)[-1]
            ext = f".{ext.lower()}" if ext else ".jpg"
            path = download_file(url, ext)
            tmp_paths.append(path)

        # Load images one at a time, immediately resize to output dims
        images = []
        for p in tmp_paths:
            img = load_image_bgr(p)
            images.append(img)
            gc.collect()

        # Align (skip for single shot)
        if len(images) > 1:
            images = align_images(images)

        # Merge
        if len(images) == 1:
            merged = images[0]
        else:
            merged = merge_mertens(images)
            del images
            gc.collect()

        # Tone
        toned = apply_realestate_tone(merged)
        del merged
        gc.collect()

        # Encode to JPEG in memory
        pil = Image.fromarray(cv2.cvtColor(toned, cv2.COLOR_BGR2RGB))
        del toned
        buf = io.BytesIO()
        pil.save(buf, format="JPEG", quality=92, optimize=True)
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
    finally:
        for p in tmp_paths:
            try:
                os.unlink(p)
            except Exception:
                pass
        gc.collect()


@app.get("/health")
def health():
    return {"status": "ok"}
def health():
    return {"status": "ok"}
