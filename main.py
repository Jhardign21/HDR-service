from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
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


# ---------------------------------------------------------------------------
# Processing parameters
# ---------------------------------------------------------------------------

class ProcessingParams(BaseModel):
    exposure: float = 0.5        # -2 to +2  EV
    saturation: float = 1.25     # 0 to 2
    shadows: float = 0.18        # 0 to 0.5
    whites: float = 0.93         # 0.7 to 1
    blacks: float = 0.04         # 0 to 0.2
    temperature: float = 0.0     # -50 to +50
    window_pull: float = 0.7     # 0 to 1  (strength of window darkening)


# ---------------------------------------------------------------------------
# Window detection & pull
# ---------------------------------------------------------------------------

def detect_window_mask(img: np.ndarray, threshold: int = 210) -> np.ndarray:
    """
    Returns a float32 mask [0..1] where bright window regions are 1.0.
    Uses luminance in LAB space to find blown-out / very bright areas.
    """
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    L = lab[:, :, 0]  # 0-255

    # Threshold on luminance
    _, bright_mask = cv2.threshold(L, threshold, 255, cv2.THRESH_BINARY)

    # Morphological cleanup: close small gaps, remove tiny specks
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    bright_mask = cv2.morphologyEx(bright_mask, cv2.MORPH_CLOSE, kernel)
    bright_mask = cv2.morphologyEx(bright_mask, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))

    # Find connected components — keep only reasonably large blobs (windows are big)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(bright_mask, connectivity=8)
    min_area = (OUTPUT_WIDTH * OUTPUT_HEIGHT) * 0.001  # at least 0.1% of frame
    filtered = np.zeros_like(bright_mask)
    for i in range(1, num_labels):
        if stats[i, cv2.CC_STAT_AREA] >= min_area:
            filtered[labels == i] = 255

    # Feather the edges for smooth blending
    feathered = cv2.GaussianBlur(filtered.astype(np.float32), (0, 0), sigmaX=25)
    feathered = feathered / 255.0
    return feathered


def apply_window_pull(merged: np.ndarray, dark_img: np.ndarray,
                      mask: np.ndarray, strength: float = 0.7) -> np.ndarray:
    """
    Blend darkest exposure into window regions using the mask.
    strength=0 → no pull, strength=1 → full dark image in windows.
    """
    merged_f = merged.astype(np.float32)
    dark_f = dark_img.astype(np.float32)
    mask3 = np.stack([mask * strength] * 3, axis=2)
    result = merged_f * (1.0 - mask3) + dark_f * mask3
    return np.clip(result, 0, 255).astype(np.uint8)


# ---------------------------------------------------------------------------
# Tone pipeline
# ---------------------------------------------------------------------------

def apply_tone(img: np.ndarray, p: ProcessingParams) -> np.ndarray:
    f = img.astype(np.float32) / 255.0

    # 1. Gamma lift + exposure boost
    ev = 2.0 ** p.exposure
    f  = np.power(np.clip(f, 1e-6, 1.0), 0.72)
    f  = np.clip(f * ev, 0, 1)

    # 2. Black floor
    f = f * (1.0 - p.blacks) + p.blacks

    # 3. White ceiling
    f = np.clip(f, 0, p.whites) / p.whites

    # 4. S-curve contrast
    f = f * f * (3.0 - 2.0 * f)

    # 5. Shadow lift
    shadow_mask = np.clip(1.0 - f / 0.4, 0, 1)
    f = f + shadow_mask * p.shadows
    f = np.clip(f, 0, 1)

    # 6. Temperature shift
    if abs(p.temperature) > 0.5:
        shift = p.temperature / 400.0
        f[:, :, 2] = np.clip(f[:, :, 2] + shift, 0, 1)
        f[:, :, 0] = np.clip(f[:, :, 0] - shift * 0.5, 0, 1)

    # 7. Neutral white balance
    for c in range(3):
        p99 = float(np.percentile(f[:, :, c], 99))
        if p99 > 0.55:
            f[:, :, c] = np.clip(f[:, :, c] * (0.97 / p99), 0, 1)

    # 8. Saturation
    rgb_u8 = np.clip(f * 255, 0, 255).astype(np.uint8)
    hsv = cv2.cvtColor(rgb_u8, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[:, :, 1] = np.clip(hsv[:, :, 1] * p.saturation, 0, 255)
    result = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)

    # 9. Mild sharpening
    blurred = cv2.GaussianBlur(result, (0, 0), sigmaX=1.5)
    result = cv2.addWeighted(result, 1.4, blurred, -0.4, 0)

    del f, hsv, blurred
    gc.collect()
    return result


# ---------------------------------------------------------------------------
# RAW loading / alignment / Mertens
# ---------------------------------------------------------------------------

class MergeRequest(BaseModel):
    file_urls: List[str]
    bracket_name: str = "bracket"
    params: Optional[ProcessingParams] = None


def download_file(url: str, suffix: str) -> str:
    with requests.get(url, timeout=60, stream=True, verify=False) as r:
        r.raise_for_status()
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix, dir="/tmp")
        for chunk in r.iter_content(chunk_size=1024 * 1024):
            tmp.write(chunk)
        tmp.close()
    return tmp.name


def decode_raw(path: str) -> np.ndarray:
    with rawpy.imread(path) as raw:
        rgb = raw.postprocess(
            use_camera_wb=True,
            no_auto_bright=False,
            output_bps=8,
            half_size=True,
        )
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


def align_images(images: List[np.ndarray]) -> List[np.ndarray]:
    if len(images) == 1:
        return images
    aligned  = [images[0]]
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
            aligned.append(cv2.warpAffine(
                img, warp, (OUTPUT_WIDTH, OUTPUT_HEIGHT),
                flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP,
            ))
        except Exception:
            aligned.append(img)
        del gray
    del ref_gray
    gc.collect()
    return aligned


def merge_mertens(images: List[np.ndarray]) -> np.ndarray:
    fused = cv2.createMergeMertens(
        contrast_weight=1.0, saturation_weight=1.2, exposure_weight=0.2
    ).process(images)
    result = np.clip(fused * 255, 0, 255).astype(np.uint8)
    del fused
    gc.collect()
    return result


def get_darkest_image(images: List[np.ndarray]) -> np.ndarray:
    """Return the image with the lowest mean brightness (darkest = best window detail)."""
    return min(images, key=lambda img: float(np.mean(img)))


# ---------------------------------------------------------------------------
# Endpoint
# ---------------------------------------------------------------------------

@app.post("/merge")
async def merge_hdr(req: MergeRequest):
    if not req.file_urls:
        raise HTTPException(400, "No file URLs provided")
    if len(req.file_urls) not in (1, 3, 5):
        raise HTTPException(400, f"Expected 1, 3, or 5 files, got {len(req.file_urls)}")

    p = req.params or ProcessingParams()
    print(f"Processing params: {p}")

    tmp_paths = []
    try:
        for url in req.file_urls:
            ext = url.split("?")[0].rsplit(".", 1)[-1]
            ext = f".{ext.lower()}" if ext else ".jpg"
            tmp_paths.append(download_file(url, ext))

        images = [load_image_bgr(path) for path in tmp_paths]
        gc.collect()

        if len(images) > 1:
            images = align_images(images)

        # Keep a reference to the darkest image for window pull before merging
        dark_img = get_darkest_image(images)

        merged = images[0] if len(images) == 1 else merge_mertens(images)
        if len(images) > 1:
            del images
            gc.collect()

        # Window pull: detect blown-out window regions and blend in dark exposure
        if p.window_pull > 0:
            print("Detecting windows...")
            window_mask = detect_window_mask(merged, threshold=210)
            window_coverage = float(np.mean(window_mask))
            print(f"Window mask coverage: {window_coverage:.3f}")
            if window_coverage > 0.001:
                print(f"Applying window pull (strength={p.window_pull})...")
                merged = apply_window_pull(merged, dark_img, window_mask, strength=p.window_pull)

        del dark_img
        gc.collect()

        # Tone mapping
        toned = apply_tone(merged, p)
        del merged
        gc.collect()

        # Encode
        pil = Image.fromarray(cv2.cvtColor(toned, cv2.COLOR_BGR2RGB))
        del toned
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
    finally:
        for path in tmp_paths:
            try:
                os.unlink(path)
            except Exception:
                pass
        gc.collect()


@app.get("/health")
def health():
    return {"status": "ok"}
def health():
    return {"status": "ok"}
