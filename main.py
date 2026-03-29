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
    exposure: float = -0.15       # reduced — prevents floor blowout
    saturation: float = 1.1       # 0 to 2
    shadows: float = 0.08         # 0 to 0.5
    whites: float = 0.92          # pull whites down to retain floor texture
    blacks: float = 0.01          # 0 to 0.2
    temperature: float = -10.0    # negative = cooler/more neutral
    window_pull: float = 0.92     # stronger window pull


# ---------------------------------------------------------------------------
# Window detection & pull
# ---------------------------------------------------------------------------

def detect_window_mask(img: np.ndarray, threshold: int = 185) -> np.ndarray:
    """
    Detects blown-out / very bright regions (windows) via LAB luminance.
    Returns a float32 feathered mask [0..1].
    """
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    L = lab[:, :, 0]  # 0-255

    # Primary: very bright pixels
    _, bright = cv2.threshold(L, threshold, 255, cv2.THRESH_BINARY)

    # Also catch near-white regions — windows are bright AND low saturation
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    low_sat = (hsv[:, :, 1] < 30).astype(np.uint8) * 255
    bright_l = (L > 170).astype(np.uint8) * 255
    low_sat_bright = cv2.bitwise_and(low_sat, bright_l)

    combined = cv2.bitwise_or(bright, low_sat_bright)

    # Morphological cleanup
    close_k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (25, 25))
    open_k  = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    combined = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, close_k)
    combined = cv2.morphologyEx(combined, cv2.MORPH_OPEN, open_k)

    # Keep only large blobs (windows are at least 0.03% of frame)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(combined, connectivity=8)
    min_area = (OUTPUT_WIDTH * OUTPUT_HEIGHT) * 0.0003
    filtered = np.zeros_like(combined)
    for i in range(1, num_labels):
        if stats[i, cv2.CC_STAT_AREA] >= min_area:
            filtered[labels == i] = 255

    # CRITICAL: Zero out the bottom 45% of the image — floors reflect light
    # and get misidentified as windows. Windows only exist in the upper portion.
    height = filtered.shape[0]
    cutoff = int(height * 0.55)  # only keep mask above 55% height
    filtered[cutoff:, :] = 0

    # Feather edges
    feathered = cv2.GaussianBlur(filtered.astype(np.float32), (0, 0), sigmaX=18)
    feathered = feathered / 255.0

    del lab, hsv, bright, low_sat, bright_l, low_sat_bright, combined, filtered
    gc.collect()
    return feathered


def apply_window_pull(merged: np.ndarray, dark_img: np.ndarray,
                      mask: np.ndarray, strength: float = 0.92) -> np.ndarray:
    """
    Blend darkest exposure into blown window regions.
    The dark image is brightened just enough to reveal exterior detail clearly.
    """
    merged_f = merged.astype(np.float32)
    dark_f = dark_img.astype(np.float32)
    # Brighten dark image so exterior detail is clearly visible
    dark_brightened = np.clip(dark_f * 1.8, 0, 255)
    mask3 = np.stack([mask * strength] * 3, axis=2)
    result = merged_f * (1.0 - mask3) + dark_brightened * mask3
    return np.clip(result, 0, 255).astype(np.uint8)


# ---------------------------------------------------------------------------
# Tone pipeline — tuned to match professional real estate style:
# cool-neutral gray walls, crisp whites, balanced exposure
# ---------------------------------------------------------------------------

def apply_tone(img: np.ndarray, p: ProcessingParams) -> np.ndarray:
    f = img.astype(np.float32) / 255.0

    # 1. Gentle gamma correction — 0.88 avoids blowing bright floors
    f = np.power(np.clip(f, 1e-6, 1.0), 0.88)

    # 2. Exposure
    ev = 2.0 ** p.exposure
    f = np.clip(f * ev, 0, 1)

    # 3. Shadow lift (gentle)
    shadow_mask = np.clip(1.0 - f / 0.35, 0, 1)
    f = f + shadow_mask * p.shadows
    f = np.clip(f, 0, 1)

    # 4. Black floor
    f = f * (1.0 - p.blacks) + p.blacks

    # 5. White ceiling
    f = np.clip(f, 0, p.whites) / p.whites

    # 6. Mild S-curve for contrast
    f = f * f * (3.0 - 2.0 * f)

    # 7. Temperature shift — negative = cooler (remove warmth)
    if abs(p.temperature) > 0.5:
        shift = p.temperature / 500.0  # small shift
        # Cool: reduce red, boost blue slightly
        f[:, :, 2] = np.clip(f[:, :, 2] + shift, 0, 1)   # Blue (BGR)
        f[:, :, 1] = np.clip(f[:, :, 1] + shift * 0.2, 0, 1)  # Green slight
        f[:, :, 0] = np.clip(f[:, :, 0] - shift * 0.6, 0, 1)  # Red

    # 8. Gray-world white balance correction (neutral walls)
    # Only correct if image is actually warm-biased
    mean_b = float(np.mean(f[:, :, 0]))
    mean_g = float(np.mean(f[:, :, 1]))
    mean_r = float(np.mean(f[:, :, 2]))
    mean_all = (mean_b + mean_g + mean_r) / 3.0
    if mean_all > 0.01:
        f[:, :, 0] = np.clip(f[:, :, 0] * (mean_all / mean_b) * 0.97, 0, 1)
        f[:, :, 1] = np.clip(f[:, :, 1] * (mean_all / mean_g) * 0.99, 0, 1)
        f[:, :, 2] = np.clip(f[:, :, 2] * (mean_all / mean_r) * 1.0, 0, 1)

    # 9. Saturation
    rgb_u8 = np.clip(f * 255, 0, 255).astype(np.uint8)
    hsv = cv2.cvtColor(rgb_u8, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[:, :, 1] = np.clip(hsv[:, :, 1] * p.saturation, 0, 255)
    result = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)

    # 10. Gentle sharpening
    blurred = cv2.GaussianBlur(result, (0, 0), sigmaX=1.2)
    result = cv2.addWeighted(result, 1.35, blurred, -0.35, 0)

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
    # Use higher contrast weight to preserve detail, lower exposure weight to avoid halos
    fused = cv2.createMergeMertens(
        contrast_weight=1.2, saturation_weight=1.0, exposure_weight=0.1
    ).process(images)
    result = np.clip(fused * 255, 0, 255).astype(np.uint8)
    del fused
    gc.collect()
    return result


def get_darkest_image(images: List[np.ndarray]) -> np.ndarray:
    """Return the darkest image (underexposed bracket — best window detail)."""
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

        # Store dark image BEFORE alignment (preserves original underexposed detail)
        dark_img = get_darkest_image(images)

        if len(images) > 1:
            images = align_images(images)

        merged = images[0] if len(images) == 1 else merge_mertens(images)
        if len(images) > 1:
            del images
            gc.collect()

        # Window pull: detect blown windows, blend in darkest bracket
        if p.window_pull > 0:
            print("Detecting windows...")
            window_mask = detect_window_mask(merged, threshold=195)
            window_coverage = float(np.mean(window_mask))
            print(f"Window mask coverage: {window_coverage:.4f}")
            if window_coverage > 0.0005:
                print(f"Applying window pull (strength={p.window_pull})...")
                merged = apply_window_pull(merged, dark_img, window_mask, strength=p.window_pull)
            else:
                print("No significant window regions detected, skipping pull")

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
