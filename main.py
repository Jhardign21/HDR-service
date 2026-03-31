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
    exposure: float = 0.18       # modest lift — prevents ceiling blow-out
    saturation: float = 1.0
    shadows: float = 0.28        # lift dark corners without bleaching
    whites: float = 0.92
    blacks: float = 0.02
    temperature: float = 0.0
    window_pull: float = 0.55


# ---------------------------------------------------------------------------
# Window detection — real estate grade
# ---------------------------------------------------------------------------

def detect_window_mask(img: np.ndarray) -> np.ndarray:
    """
    Detect the full window/glass opening — not just blown highlights.
    Strategy: find blown cores first, then dilate aggressively to cover
    the entire window frame area (including darker pane regions like fences).
    Then erode back inward to exclude curtains/frames before feathering.
    """
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    L = lab[:, :, 0]  # 0-255

    # Step 1: seed = truly blown pixels (L > 200)
    _, blown = cv2.threshold(L, 200, 255, cv2.THRESH_BINARY)

    # Also seed on high-brightness + low-sat (glass areas)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    low_sat = (hsv[:, :, 1] < 25).astype(np.uint8) * 255
    bright_l = (L > 185).astype(np.uint8) * 255
    window_glass = cv2.bitwise_and(low_sat, bright_l)
    seed = cv2.bitwise_or(blown, window_glass)

    # Step 2: close small gaps, remove tiny noise
    close_k = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 25))
    open_k  = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    seed = cv2.morphologyEx(seed, cv2.MORPH_CLOSE, close_k)
    seed = cv2.morphologyEx(seed, cv2.MORPH_OPEN, open_k)

    # Step 3: keep only large blobs (real windows, not fixtures)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(seed, connectivity=8)
    min_area = (OUTPUT_WIDTH * OUTPUT_HEIGHT) * 0.0025
    filtered = np.zeros_like(seed)
    for i in range(1, num_labels):
        if stats[i, cv2.CC_STAT_AREA] >= min_area:
            filtered[labels == i] = 255

    # Step 4: DILATE blobs to cover the entire window opening
    # This fills in darker pane areas (fence, lower sash) attached to the blown core
    dilate_k = cv2.getStructuringElement(cv2.MORPH_RECT, (60, 60))
    expanded = cv2.dilate(filtered, dilate_k, iterations=2)

    # Step 5: restrict to valid vertical band (exclude ceiling 10%, floor 55%)
    height = expanded.shape[0]
    top_cut = int(height * 0.10)
    bot_cut = int(height * 0.55)
    expanded[:top_cut, :] = 0
    expanded[bot_cut:, :] = 0

    # Step 6: erode inward aggressively to stay strictly inside glass, away from frames
    erode_k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (22, 22))
    shrunk = cv2.erode(expanded, erode_k, iterations=5)

    # Step 7: moderate feathering — smooth blend inside glass only
    feathered = cv2.GaussianBlur(shrunk.astype(np.float32), (0, 0), sigmaX=8)
    max_val = feathered.max()
    if max_val > 0:
        feathered = feathered / max_val

    del lab, hsv, blown, low_sat, bright_l, window_glass, seed, filtered, expanded, shrunk
    gc.collect()
    return feathered


def apply_window_pull(merged: np.ndarray, dark_img: np.ndarray,
                      mask: np.ndarray, strength: float = 0.80) -> np.ndarray:
    """
    Blend brightened dark bracket into window regions to reveal exterior.
    Multiply dark bracket by 2.0 so exterior scene (street, trees) is visible
    rather than near-black.
    """
    merged_f = merged.astype(np.float32)
    # Brighten the dark bracket significantly so exterior is naturally exposed
    dark_f = np.clip(dark_img.astype(np.float32) * 2.0, 0, 255)
    mask3 = np.stack([mask * strength] * 3, axis=2)
    result = np.clip(merged_f * (1.0 - mask3) + dark_f * mask3, 0, 255).astype(np.uint8)

    # Subtle sky-blue tint on very bright, low-sat window pixels
    hsv = cv2.cvtColor(result, cv2.COLOR_BGR2HSV).astype(np.float32)
    brightness = hsv[:, :, 2] / 255.0
    saturation = hsv[:, :, 1] / 255.0
    sky_pixel = (brightness > 0.55) & (saturation < 0.25)
    sky_strength = mask * sky_pixel.astype(np.float32) * 0.4

    result_f = result.astype(np.float32)
    result_f[:, :, 0] = np.clip(result_f[:, :, 0] + sky_strength * 10, 0, 255)
    result_f[:, :, 1] = np.clip(result_f[:, :, 1] + sky_strength * 3,  0, 255)
    result_f[:, :, 2] = np.clip(result_f[:, :, 2] - sky_strength * 6,  0, 255)

    del hsv, brightness, saturation, sky_pixel, sky_strength
    return np.clip(result_f, 0, 255).astype(np.uint8)


# ---------------------------------------------------------------------------
# Geometry correction — lens undistort + vertical keystone
# ---------------------------------------------------------------------------

def correct_geometry(img: np.ndarray) -> np.ndarray:
    """
    1. Barrel/pincushion undistort (typical wide-angle real-estate lens)
    2. Vertical perspective correction — straightens converging verticals
       by detecting near-vertical lines and warping them to be parallel.
    """
    h, w = img.shape[:2]

    # --- Step 1: Barrel distortion correction ---
    k1, k2, p1, p2 = -0.22, 0.07, 0.0, 0.0
    fx = fy = w * 1.05
    cx, cy = w / 2.0, h / 2.0
    K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float64)
    dist = np.array([k1, k2, p1, p2], dtype=np.float64)
    new_K, _ = cv2.getOptimalNewCameraMatrix(K, dist, (w, h), alpha=0.0)
    undistorted = cv2.undistort(img, K, dist, None, new_K)

    # --- Step 2: Vertical perspective correction (converging verticals) ---
    gray = cv2.cvtColor(undistorted, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 40, 120, apertureSize=3)
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=60,
                            minLineLength=h // 6, maxLineGap=15)

    # Collect near-vertical line segments and measure their horizontal drift
    # A perfect vertical has dx=0; converging lines have dx != 0
    drifts = []  # horizontal drift per unit height (positive = lean right at top)
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            dx = float(x2 - x1)
            dy = float(y2 - y1)
            if abs(dy) > abs(dx) * 2.5 and abs(dy) > h // 8:
                # drift: how much x changes per full image height
                drift = dx / dy  # pixel shift per pixel drop
                drifts.append(drift)

    if len(drifts) >= 4:
        median_drift = float(np.median(drifts))
        # Only correct if there is meaningful convergence
        if abs(median_drift) > 0.004:
            # Stronger clamp to allow real-estate lens corrections
            correction = np.clip(median_drift, -0.14, 0.14)
            # Amplify the shift — real-estate lenses need aggressive vertical fix
            shift = correction * h * 0.85
            src = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
            dst = np.float32([
                [shift,        0],
                [w - shift,    0],
                [w,            h],
                [0,            h],
            ])
            M = cv2.getPerspectiveTransform(src, dst)
            undistorted = cv2.warpPerspective(undistorted, M, (w, h),
                                              flags=cv2.INTER_LINEAR,
                                              borderMode=cv2.BORDER_REPLICATE)
            print(f"Vertical perspective corrected: drift={median_drift:.4f}, shift={shift:.1f}px")

    del gray, edges, lines
    gc.collect()
    return undistorted


# ---------------------------------------------------------------------------
# Tone pipeline
# ---------------------------------------------------------------------------

def apply_tone(img: np.ndarray, p: ProcessingParams) -> np.ndarray:
    f = img.astype(np.float32) / 255.0

    # 1. Gentle gamma lift (0.95 — softer, preserves highlight texture)
    f = np.power(np.clip(f, 1e-6, 1.0), 0.95)

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

    # 6. Highlight rolloff — compress highlights above 60% to protect ceiling texture
    hi = np.clip((f - 0.60) / 0.40, 0, 1)
    f = f - hi * (f - 0.60) * 0.70
    f = np.clip(f, 0, 1)

    # 7. Mild S-curve
    f = f * f * (3.0 - 2.0 * f)

    # 8. Optional temperature shift (user-controlled, off by default)
    lum = (f[:, :, 0] + f[:, :, 1] + f[:, :, 2]) / 3.0
    mid_mask = np.clip((0.75 - lum) / 0.25, 0, 1)
    mid_mask3 = np.stack([mid_mask] * 3, axis=2)

    if abs(p.temperature) > 0.5:
        shift = p.temperature / 500.0
        delta = np.zeros_like(f)
        delta[:, :, 2] += shift
        delta[:, :, 1] += shift * 0.2
        delta[:, :, 0] -= shift * 0.6
        f = np.clip(f + delta * mid_mask3, 0, 1)

    # 9. NO gray-world WB — removed: it fights warm-toned rooms (cream/peach walls,
    # honey wood floors) and produces blue casts. Camera WB is preserved from rawpy.

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

        # Geometry correction — undistort + straighten verticals
        print("Correcting geometry...")
        merged = correct_geometry(merged)

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
