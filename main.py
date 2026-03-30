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
    exposure: float = -0.20
    saturation: float = 1.05
    shadows: float = 0.12
    whites: float = 0.92
    blacks: float = 0.01
    temperature: float = -30.0   # push tan walls fully → cool neutral gray
    window_pull: float = 0.80    # gentler pull to avoid halo/ghost artifacts


# ---------------------------------------------------------------------------
# Window detection — real estate grade
# ---------------------------------------------------------------------------

def detect_window_mask(img: np.ndarray) -> np.ndarray:
    """
    Detect blown-out window/glass regions only.
    - Uses LAB luminance for primary detection
    - Excludes ceiling (top 10%) and floor (bottom 42%)
    - Requires large connected blobs (windows, not fixtures)
    - Heavy feathering for seamless blending
    """
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    L = lab[:, :, 0]  # 0-255

    # Primary: truly blown pixels (L > 205)
    _, blown = cv2.threshold(L, 205, 255, cv2.THRESH_BINARY)

    # Secondary: high-brightness + very low saturation (window glass / sky)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    low_sat = (hsv[:, :, 1] < 18).astype(np.uint8) * 255
    bright_l = (L > 192).astype(np.uint8) * 255
    window_glass = cv2.bitwise_and(low_sat, bright_l)

    combined = cv2.bitwise_or(blown, window_glass)

    # Morphological: close gaps inside window frame, remove small noise
    close_k = cv2.getStructuringElement(cv2.MORPH_RECT, (30, 30))
    open_k  = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    combined = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, close_k)
    combined = cv2.morphologyEx(combined, cv2.MORPH_OPEN, open_k)

    # Only keep large blobs — windows are big; ceiling fixtures are small
    # Minimum 0.25% of frame area
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(combined, connectivity=8)
    min_area = (OUTPUT_WIDTH * OUTPUT_HEIGHT) * 0.0025
    filtered = np.zeros_like(combined)
    for i in range(1, num_labels):
        if stats[i, cv2.CC_STAT_AREA] >= min_area:
            filtered[labels == i] = 255

    height = filtered.shape[0]
    # Exclude top 10% (ceiling lights/fixtures) and bottom 42% (floors reflect light)
    top_cut = int(height * 0.10)
    bot_cut = int(height * 0.58)
    filtered[:top_cut, :] = 0
    filtered[bot_cut:, :] = 0

    # Erode mask inward — must stay strictly inside glass, never touch curtains/frames
    erode_k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (25, 25))
    filtered = cv2.erode(filtered, erode_k, iterations=5)  # very aggressive: kills curtain bleed

    # Very tight feathering — minimal spread to prevent halo
    feathered = cv2.GaussianBlur(filtered.astype(np.float32), (0, 0), sigmaX=6)
    max_val = feathered.max()
    if max_val > 0:
        feathered = feathered / max_val

    del lab, hsv, blown, low_sat, bright_l, window_glass, combined, filtered
    gc.collect()
    return feathered


def apply_window_pull(merged: np.ndarray, dark_img: np.ndarray,
                      mask: np.ndarray, strength: float = 0.95) -> np.ndarray:
    """
    Blend darkest bracket into window regions, then add subtle sky-blue tint
    to bright exterior pixels (sky) visible through the glass.
    """
    merged_f = merged.astype(np.float32)
    dark_f = np.clip(dark_img.astype(np.float32) * 1.2, 0, 255)
    mask3 = np.stack([mask * strength] * 3, axis=2)
    result = (merged_f * (1.0 - mask3) + dark_f * mask3)
    result = np.clip(result, 0, 255).astype(np.uint8)

    # Sky blue tint: within the window mask, find very bright low-saturation pixels
    # (overcast/bright sky) and add a subtle cool-blue shift
    hsv = cv2.cvtColor(result, cv2.COLOR_BGR2HSV).astype(np.float32)
    brightness = hsv[:, :, 2] / 255.0   # 0-1
    saturation = hsv[:, :, 1] / 255.0   # 0-1
    # Sky pixels: bright (>0.55) and low saturation (<0.25) within the window region
    sky_pixel = (brightness > 0.55) & (saturation < 0.25)
    sky_strength = mask * sky_pixel.astype(np.float32) * 0.5  # max 50% of mask

    result_f = result.astype(np.float32)
    # BGR: add blue (+12), slight green (+3), reduce red (-8)
    result_f[:, :, 0] = np.clip(result_f[:, :, 0] + sky_strength * 12, 0, 255)  # B
    result_f[:, :, 1] = np.clip(result_f[:, :, 1] + sky_strength * 3,  0, 255)  # G
    result_f[:, :, 2] = np.clip(result_f[:, :, 2] - sky_strength * 8,  0, 255)  # R

    del hsv, brightness, saturation, sky_pixel, sky_strength
    return np.clip(result_f, 0, 255).astype(np.uint8)


# ---------------------------------------------------------------------------
# Tone pipeline
# ---------------------------------------------------------------------------

def apply_tone(img: np.ndarray, p: ProcessingParams) -> np.ndarray:
    f = img.astype(np.float32) / 255.0

    # 1. Gentle gamma lift (0.90 keeps brights from clipping)
    f = np.power(np.clip(f, 1e-6, 1.0), 0.90)

    # 2. Exposure
    ev = 2.0 ** p.exposure
    f = np.clip(f * ev, 0, 1)

    # 3. Shadow lift
    shadow_mask = np.clip(1.0 - f / 0.30, 0, 1)
    f = np.clip(f + shadow_mask * p.shadows, 0, 1)

    # 4. Black floor
    f = f * (1.0 - p.blacks) + p.blacks

    # 5. White ceiling
    f = np.clip(f, 0, p.whites) / p.whites

    # 6. Highlight rolloff — compress highlights above 65% to preserve floor grain
    hi = np.clip((f - 0.65) / 0.35, 0, 1)
    f = f - hi * (f - 0.65) * 0.60
    f = np.clip(f, 0, 1)

    # 7. Mild S-curve
    f = f * f * (3.0 - 2.0 * f)

    # 8. Temperature shift (cool) — midtone-only mask protects bright floor/ceiling
    # Luminance of each pixel: protect highlights (floor ~0.85+) from blue push
    lum = (f[:, :, 0] + f[:, :, 1] + f[:, :, 2]) / 3.0
    # Weight: full strength on midtones (0.3-0.7), fades to 0 on highlights (>0.82)
    mid_mask = np.clip((0.82 - lum) / 0.30, 0, 1)  # shape (H,W)
    mid_mask3 = np.stack([mid_mask] * 3, axis=2)

    if abs(p.temperature) > 0.5:
        shift = p.temperature / 500.0
        delta = np.zeros_like(f)
        delta[:, :, 2] += shift        # Blue channel
        delta[:, :, 1] += shift * 0.2  # Green (slight)
        delta[:, :, 0] -= shift * 0.6  # Red
        f = np.clip(f + delta * mid_mask3, 0, 1)

    # 9. Gray-world white balance — midtone-only to preserve warm floor tone
    mean_b = float(np.mean(f[:, :, 0]))
    mean_g = float(np.mean(f[:, :, 1]))
    mean_r = float(np.mean(f[:, :, 2]))
    mean_all = (mean_b + mean_g + mean_r) / 3.0
    if mean_all > 0.01:
        # Compute per-pixel WB correction blended with mid_mask
        wb_b = np.full_like(f[:, :, 0], (mean_all / mean_b) * 1.04)
        wb_g = np.full_like(f[:, :, 1], (mean_all / mean_g) * 0.99)
        wb_r = np.full_like(f[:, :, 2], (mean_all / mean_r) * 0.92)
        # Blend: midtones get full WB correction, highlights stay unchanged (scale=1.0)
        scale_b = 1.0 + (wb_b - 1.0) * mid_mask
        scale_g = 1.0 + (wb_g - 1.0) * mid_mask
        scale_r = 1.0 + (wb_r - 1.0) * mid_mask
        f[:, :, 0] = np.clip(f[:, :, 0] * scale_b, 0, 1)
        f[:, :, 1] = np.clip(f[:, :, 1] * scale_g, 0, 1)
        f[:, :, 2] = np.clip(f[:, :, 2] * scale_r, 0, 1)

    del lum, mid_mask, mid_mask3

    # 10. Saturation
    rgb_u8 = np.clip(f * 255, 0, 255).astype(np.uint8)
    hsv = cv2.cvtColor(rgb_u8, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[:, :, 1] = np.clip(hsv[:, :, 1] * p.saturation, 0, 255)
    result = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)

    # 11. Gentle sharpening
    blurred = cv2.GaussianBlur(result, (0, 0), sigmaX=1.2)
    result = cv2.addWeighted(result, 1.3, blurred, -0.3, 0)

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
    # Higher contrast weight catches detail, moderate exposure weight avoids halos
    fused = cv2.createMergeMertens(
        contrast_weight=1.4, saturation_weight=0.9, exposure_weight=0.2
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

        # Grab dark image BEFORE alignment — raw underexposed frame for window pull
        dark_img = get_darkest_image(images)

        if len(images) > 1:
            images = align_images(images)

        merged = images[0] if len(images) == 1 else merge_mertens(images)
        if len(images) > 1:
            del images
            gc.collect()

        # Detect windows on merged (blown windows clearly visible here)
        window_mask = None
        if p.window_pull > 0:
            print("Detecting windows...")
            window_mask = detect_window_mask(merged)
            window_coverage = float(np.mean(window_mask))
            print(f"Window mask coverage: {window_coverage:.4f}")
            if window_coverage <= 0.0003:
                print("No significant window regions detected")
                window_mask = None

        # Tone map the merged image
        toned = apply_tone(merged, p)
        del merged
        gc.collect()

        # Apply window pull AFTER tone mapping
        # Use raw dark_img (not tone-mapped) with slight lift — preserves natural exterior look
        if window_mask is not None and p.window_pull > 0:
            print(f"Applying window pull (strength={p.window_pull})...")
            toned = apply_window_pull(toned, dark_img, window_mask, strength=p.window_pull)

        del dark_img
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
