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
    Lightroom Upright-style vertical perspective correction:
    1. Detect near-vertical lines via Hough
    2. Find vertical vanishing point via robust pairwise intersections
    3. Apply centered homography H that sends VP to infinity
       (correctly anchored at principal point cx,cy)
    No barrel correction — it creates edge artifacts on real-estate wide-angle shots.
    """
    h, w = img.shape[:2]
    cx, cy = w / 2.0, h / 2.0

    # --- Detect near-vertical lines ---
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray_eq = cv2.equalizeHist(gray)
    edges = cv2.Canny(gray_eq, 30, 100, apertureSize=3)
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=60,
                            minLineLength=h // 5, maxLineGap=15)

    vert_lines = []
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            dx, dy = float(x2 - x1), float(y2 - y1)
            if abs(dy) > abs(dx) * 2.5 and abs(dy) > h // 5:
                vert_lines.append((float(x1), float(y1), float(x2), float(y2)))

    print(f"Vertical lines: {len(vert_lines)}")

    if len(vert_lines) < 6:
        print("Not enough vertical lines — skipping VP correction")
        del gray, gray_eq, edges, lines
        gc.collect()
        return img

    # --- Find vertical VP via pairwise intersections ---
    import random
    sample = vert_lines if len(vert_lines) <= 40 else random.sample(vert_lines, 40)
    vp_candidates = []
    for i in range(len(sample)):
        for j in range(i + 1, len(sample)):
            x1, y1, x2, y2 = sample[i]
            x3, y3, x4, y4 = sample[j]
            denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
            if abs(denom) < 1e-6:
                continue
            t = ((x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)) / denom
            ix = x1 + t * (x2 - x1)
            iy = y1 + t * (y2 - y1)
            # Only intersections clearly outside the image (genuine tilt, not natural perspective)
            if (iy < -h * 0.5 or iy > h * 1.5) and (-w * 0.5 < ix < w * 1.5):
                vp_candidates.append((ix, iy))

    print(f"VP candidates: {len(vp_candidates)}")

    if len(vp_candidates) < 10:
        print("Not enough VP candidates — skipping correction")
        del gray, gray_eq, edges, lines
        gc.collect()
        return img

    vpy = float(np.median([v[1] for v in vp_candidates]))
    vpx = float(np.median([v[0] for v in vp_candidates]))
    print(f"VP: ({vpx:.0f}, {vpy:.0f}) image={w}x{h} center=({cx:.0f},{cy:.0f})")

    # VP must be genuinely outside (real tilt) not just natural in-frame convergence
    if -h * 0.4 <= vpy <= h * 1.4:
        print(f"VP y={vpy:.0f} too close to image — natural perspective, skipping")
        del gray, gray_eq, edges, lines
        gc.collect()
        return img

    # --- Centered Lightroom-style homography ---
    # Translate so principal point is at origin, apply keystone, translate back.
    # This matches how Lightroom/ACR compute Upright corrections.
    # H_centered = T_back @ H_vp @ T_to_origin
    # H_vp sends VP at (vpx-cx, vpy-cy) to infinity: row3 = [0, -1/(vpy-cy), 1]
    vpy_c = vpy - cy  # VP y in centered coords

    # Cap correction: allow stronger correction for exterior shots with clear tilt
    p = -1.0 / vpy_c
    max_p = 1.0 / (0.4 * h)
    p = float(np.clip(p, -max_p, max_p))

    # Full homography: T_back @ [[1,0,0],[0,1,0],[0,p,1]] @ T_to_origin
    H = np.array([
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0,   p, 1.0]
    ], dtype=np.float64)
    # Account for translation: T_back @ H @ T_to_origin
    T = np.array([[1,0,-cx],[0,1,-cy],[0,0,1]], dtype=np.float64)
    T_inv = np.array([[1,0,cx],[0,1,cy],[0,0,1]], dtype=np.float64)
    H_full = T_inv @ H @ T

    result = cv2.warpPerspective(
        img, H_full, (w, h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0)
    )
    print(f"VP correction applied: p={p:.6f} vpy_c={vpy_c:.0f}")

    # --- Analytical crop: project the 4 source corners through H_full ---
    # This finds the exact inscribed valid rectangle with no black corners.
    corners_src = np.array([[0, 0, 1], [w, 0, 1], [w, h, 1], [0, h, 1]], dtype=np.float64).T
    corners_dst = H_full @ corners_src
    corners_dst /= corners_dst[2, :]  # normalize homogeneous
    dst_x = corners_dst[0, :]
    dst_y = corners_dst[1, :]

    # Valid inscribed rectangle: tightest box where ALL corners are inside
    x0 = int(np.ceil(max(dst_x[0], dst_x[3])))   # left edge: max of left-side corners
    x1 = int(np.floor(min(dst_x[1], dst_x[2])))  # right edge: min of right-side corners
    y0 = int(np.ceil(max(dst_y[0], dst_y[1])))   # top edge: max of top corners
    y1 = int(np.floor(min(dst_y[2], dst_y[3])))  # bottom edge: min of bottom corners

    # Clamp to image bounds with small safety margin
    x0 = max(x0 + 2, 0)
    x1 = min(x1 - 2, w)
    y0 = max(y0 + 2, 0)
    y1 = min(y1 - 2, h)

    print(f"Analytical crop: x=[{x0}:{x1}] y=[{y0}:{y1}]")
    if x1 > x0 + 100 and y1 > y0 + 100:
        cropped = result[y0:y1, x0:x1]
        result = cv2.resize(cropped, (w, h), interpolation=cv2.INTER_LANCZOS4)
        del cropped

    del gray, gray_eq, edges, lines
    gc.collect()
    return result


# ---------------------------------------------------------------------------
# Per-image adaptive tone analysis
# ---------------------------------------------------------------------------

def analyze_image(img: np.ndarray) -> ProcessingParams:
    """
    Analyze the merged (pre-tone) image and return adaptive ProcessingParams
    tuned to this specific photo, targeting the professional real-estate look:
    bright whites, clean neutrals, lifted shadows, natural contrast.
    """
    f = img.astype(np.float32) / 255.0
    # Work in luminance
    lum = 0.2126 * f[:,:,2] + 0.7152 * f[:,:,1] + 0.0722 * f[:,:,0]  # RGB weights

    mean_lum   = float(np.mean(lum))
    p5         = float(np.percentile(lum, 5))    # deep shadows
    p25        = float(np.percentile(lum, 25))   # midtone-dark
    p75        = float(np.percentile(lum, 75))   # midtone-bright
    p95        = float(np.percentile(lum, 95))   # near-highlights
    p99        = float(np.percentile(lum, 99))   # highlight ceiling

    # --- Exposure: target bright airy real-estate look (mean ~0.62) ---
    target_mean = 0.62
    if mean_lum > 0.01:
        raw_ev = np.log2(target_mean / mean_lum)
    else:
        raw_ev = 1.0
    # Less dampening so we actually reach the target brightness
    exposure = float(np.clip(raw_ev * 0.78, -0.3, 0.85))

    # --- Shadow lift: raise p5 toward ~0.12 (bright clean shadows) ---
    shadow_gap = max(0.0, 0.12 - p5)
    shadows = float(np.clip(shadow_gap * 2.5, 0.14, 0.55))

    # --- Whites ceiling: keep high — real estate needs bright whites ---
    if p99 > 0.93:
        whites = float(np.clip(0.90 + (1.0 - p99) * 0.4, 0.86, 0.97))
    else:
        whites = 0.97

    # --- Blacks: minimal lift for clean look ---
    blacks = float(np.clip(p5 * 0.10, 0.003, 0.02))

    # Saturation: fixed neutral — real estate wants accurate colors
    saturation = 1.0

    # Window pull: more pull if image has strong blown regions
    blown_frac = float(np.mean(lum > 0.92))
    window_pull = float(np.clip(0.40 + blown_frac * 1.5, 0.40, 0.65))

    params = ProcessingParams(
        exposure=exposure,
        saturation=saturation,
        shadows=shadows,
        whites=whites,
        blacks=blacks,
        temperature=temperature,
        window_pull=window_pull,
    )
    print(f"Adaptive params: exp={exposure:.2f} shad={shadows:.2f} whites={whites:.2f} "
          f"blacks={blacks:.3f} temp={temperature:.0f} wp={window_pull:.2f} "
          f"[mean={mean_lum:.2f} p5={p5:.2f} p95={p95:.2f} p99={p99:.2f}]")
    return params


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

    # 6. Highlight rolloff — compress highlights above 75% to protect ceiling texture
    hi = np.clip((f - 0.75) / 0.25, 0, 1)
    f = f - hi * (f - 0.75) * 0.55
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

    # Use caller-supplied params if provided, otherwise analyse this image
    if req.params is not None:
        p = req.params
        print(f"Using supplied params: {p}")
    else:
        # Will be computed after merge, before tone mapping
        p = None
    print(f"Auto-analyse mode: {p is None}")

    tmp_paths = []
    try:
        for url in req.file_urls:
            ext = url.split("?")[0].rsplit(".", 1)[-1]
            ext = f".{ext.lower()}" if ext else ".jpg"
            tmp_paths.append(download_file(url, ext))

        images = [load_image_bgr(path) for path in tmp_paths]
        gc.collect()

        # Grab dark image BEFORE alignment — raw underexposed frame for window pull
        dark_img = get_darkest_image(images).copy()

        if len(images) > 1:
            images = align_images(images)

        merged = images[0] if len(images) == 1 else merge_mertens(images)
        if len(images) > 1:
            del images
            gc.collect()

        # Geometry correction — undistort + straighten verticals
        print("Correcting geometry on merged...")
        merged = correct_geometry(merged)
        # Apply same geometry correction to dark image so window pull aligns
        print("Correcting geometry on dark bracket...")
        dark_img = correct_geometry(dark_img)

        # Auto-analyse image if no params supplied
        if p is None:
            print("Analysing image for adaptive tone params...")
            p = analyze_image(merged)

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
