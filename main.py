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
    file_urls: List[str]   # 1, 3, or 5 file URLs
    bracket_name: str = "bracket"


def download_file(url: str, suffix: str) -> str:
    """Download a file from URL to a temp file, return path."""
    r = requests.get(url, timeout=60)
    r.raise_for_status()
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    tmp.write(r.content)
    tmp.close()
    return tmp.name


def decode_raw_to_rgb(path: str) -> np.ndarray:
    """Decode a RAW file to 8-bit RGB numpy array."""
    with rawpy.imread(path) as raw:
        rgb = raw.postprocess(
            use_camera_wb=True,
            no_auto_bright=False,
            output_bps=8,
        )
    return rgb  # HxWx3 uint8


def load_image(path: str) -> np.ndarray:
    """Load RAW or JPEG/PNG into uint8 RGB array."""
    ext = os.path.splitext(path)[1].lower()
    raw_exts = {".cr3", ".cr2", ".nef", ".arw", ".dng", ".raf", ".rw2", ".orf"}
    if ext in raw_exts:
        return decode_raw_to_rgb(path)
    else:
        img = cv2.imread(path)
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def align_images(images: List[np.ndarray]) -> List[np.ndarray]:
    """Align exposures using ECC alignment."""
    if len(images) == 1:
        return images
    aligned = [images[0]]
    ref_gray = cv2.cvtColor(images[0], cv2.COLOR_RGB2GRAY)
    for img in images[1:]:
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        warp = np.eye(2, 3, dtype=np.float32)
        try:
            _, warp = cv2.findTransformECC(
                ref_gray, gray, warp,
                cv2.MOTION_TRANSLATION,
                (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 50, 0.001)
            )
            aligned_img = cv2.warpAffine(
                img, warp, (img.shape[1], img.shape[0]),
                flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP
            )
            aligned.append(aligned_img)
        except:
            aligned.append(img)
    return aligned


def merge_hdr_mertens(images: List[np.ndarray]) -> np.ndarray:
    """Mertens exposure fusion — best for real estate (no ghost artifacts)."""
    bgr_images = [cv2.cvtColor(img, cv2.COLOR_RGB2BGR) for img in images]
    merge = cv2.createMergeMertens(contrast_weight=1.0, saturation_weight=1.0, exposure_weight=0.0)
    fused = merge.process(bgr_images)
    # Convert float32 [0,1] to uint8
    result = np.clip(fused * 255, 0, 255).astype(np.uint8)
    return cv2.cvtColor(result, cv2.COLOR_BGR2RGB)


def apply_realestate_tone(img: np.ndarray) -> np.ndarray:
    """Apply subtle real-estate tone: slight warmth, lift shadows, crisp whites."""
    # Convert to float
    f = img.astype(np.float32) / 255.0

    # Slight S-curve for contrast
    f = np.power(f, 0.88)

    # Lift blacks slightly (real estate look)
    f = f * 0.92 + 0.04

    # Warm tint: boost reds/greens slightly, keep blues
    f[:, :, 0] = np.clip(f[:, :, 0] * 1.04, 0, 1)  # R
    f[:, :, 1] = np.clip(f[:, :, 1] * 1.02, 0, 1)  # G

    return np.clip(f * 255, 0, 255).astype(np.uint8)


def resize_to_output(img: np.ndarray) -> np.ndarray:
    return cv2.resize(img, (OUTPUT_WIDTH, OUTPUT_HEIGHT), interpolation=cv2.INTER_LANCZOS4)


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

        # Load images
        images = [load_image(p) for p in tmp_paths]

        # Align (skip for single shot)
        if len(images) > 1:
            images = align_images(images)

        # Merge
        if len(images) == 1:
            merged = images[0]
        else:
            merged = merge_hdr_mertens(images)

        # Apply tone mapping
        toned = apply_realestate_tone(merged)

        # Resize to 2048x1536
        output = resize_to_output(toned)

        # Encode to JPEG
        pil = Image.fromarray(output)
        buf = io.BytesIO()
        pil.save(buf, format="JPEG", quality=95, optimize=True)
        buf.seek(0)

        jpg_b64 = base64.b64encode(buf.read()).decode("utf-8")

        return {
            "success": True,
            "bracket_name": req.bracket_name,
            "width": OUTPUT_WIDTH,
            "height": OUTPUT_HEIGHT,
            "jpeg_base64": jpg_b64,
        }

    finally:
        for p in tmp_paths:
            try:
                os.unlink(p)
            except:
                pass


@app.get("/health")
def health():
    return {"status": "ok"}
