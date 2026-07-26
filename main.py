from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional, Tuple
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
import subprocess
import shutil

app = FastAPI(title="HDR Merge Service")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

OUTPUT_WIDTH  = 2048
OUTPUT_HEIGHT = 1536


# ---------------------------------------------------------------------------
# File helpers
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
        rgb = raw.postprocess(use_camera_wb=True, no_auto_bright=False, bright=1.0,
                              output_bps=8, half_size=False, median_filter_passes=0)
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    del rgb; gc.collect()
    return bgr


def load_image_bgr(path: str) -> np.ndarray:
    ext = os.path.splitext(path)[1].lower()
    raw_exts = {".cr3", ".cr2", ".nef", ".arw", ".dng", ".raf", ".rw2", ".orf", ".raw"}
    if ext in raw_exts:
        return decode_raw(path)
    img = cv2.imread(path)
    if img is None:
        raise ValueError(f"Could not read image: {path}")
    return img


# ---------------------------------------------------------------------------
# Exposure normalisation
# ---------------------------------------------------------------------------

def normalise_to_target(img: np.ndarray, target_mean: float, max_scale: float = 3.5) -> np.ndarray:
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    current = float(gray.mean()) + 1e-6
    scale = min(target_mean / current, max_scale)
    if scale <= 1.01:
        return img
    return np.clip(img.astype(np.float32) * scale, 0, 255).astype(np.uint8)


# ---------------------------------------------------------------------------
# LSD-assisted window mask (Autoenhance-style geometric + photometric blend)
#
# This mirrors how Autoenhance uses M-LSD line detection:
#   1. Use OpenCV's LineSegmentDetector (LSD) — the classic algorithm that
#      M-LSD was built to replace for mobile, but functionally equivalent for
#      our server-side use case. No model weights to download.
#   2. Find near-horizontal and near-vertical line segments — these are the
#      edges of window frames, door frames, and wall/ceiling junctions.
#   3. Find rectangular groups of these lines (window frame candidates).
#   4. Cross-reference with photometric brightness mask (overexposed = window).
#   5. Union of geometric candidates AND bright zones = final window mask.
#      Geometric-confirmed windows get hard masking; brightness-only gets softer.
# ---------------------------------------------------------------------------

def detect_line_segments(gray: np.ndarray) -> Optional[np.ndarray]:
    """
    Run OpenCV LSD on a grayscale image.
    Returns Nx4 array of [x1,y1,x2,y2] segments, or None if LSD unavailable.
    LSD is available in opencv-contrib; falls back gracefully if not present.
    """
    try:
        lsd = cv2.createLineSegmentDetector(cv2.LSD_REFINE_STD)
        lines, _, _, _ = lsd.detect(gray)
        if lines is None or len(lines) == 0:
            return None
        return lines.reshape(-1, 4)
    except Exception as e:
        print(f"  LSD not available ({e}), falling back to HoughLinesP")
        # Fallback: standard Hough line detection
        edges = cv2.Canny(gray, 50, 150, apertureSize=3)
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=80,
                                minLineLength=50, maxLineGap=10)
        if lines is None:
            return None
        return lines.reshape(-1, 4)


def classify_segments(segments: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Split line segments into near-horizontal and near-vertical groups.
    Returns (h_segments, v_segments) as float arrays.
    """
    segs = segments.astype(np.float32)
    dx = segs[:, 2] - segs[:, 0]
    dy = segs[:, 3] - segs[:, 1]
    angles = np.degrees(np.arctan2(np.abs(dy), np.abs(dx)))  # 0=horizontal, 90=vertical
    h_mask = angles < 20.0   # nearly horizontal (window top/bottom, horizon)
    v_mask = angles > 70.0   # nearly vertical (window sides, door frames)
    return segs[h_mask], segs[v_mask]


def segments_to_coverage_map(segments: np.ndarray, h: int, w: int,
                              thickness: int = 8) -> np.ndarray:
    """
    Render line segments into a float coverage map [0..1].
    Each segment adds coverage along its length.
    """
    canvas = np.zeros((h, w), dtype=np.float32)
    for seg in segments:
        x1, y1, x2, y2 = int(seg[0]), int(seg[1]), int(seg[2]), int(seg[3])
        cv2.line(canvas, (x1, y1), (x2, y2), 1.0, thickness)
    return canvas


def build_geometric_window_candidates(segments: np.ndarray, h: int, w: int,
                                       bright_mask: np.ndarray) -> np.ndarray:
    """
    Find rectangular window candidates using H/V line segment intersections.

    Strategy (mimicking Autoenhance M-LSD usage):
      - Group H segments into "rows" by Y-position (within 40px)
      - Group V segments into "cols" by X-position (within 40px)
      - For each (row_pair, col_pair) combination that forms a plausible rectangle:
          * Check that the enclosed area overlaps significantly with bright_mask
          * If overlap > 40%, mark the filled rectangle as a window candidate
    """
    geom_mask = np.zeros((h, w), dtype=np.float32)

    if segments is None or len(segments) < 4:
        return geom_mask

    h_segs, v_segs = classify_segments(segments)

    if len(h_segs) < 2 or len(v_segs) < 2:
        return geom_mask

    # Y-centers of horizontal segments
    h_ys = ((h_segs[:, 1] + h_segs[:, 3]) / 2.0)
    # X-centers of vertical segments
    v_xs = ((v_segs[:, 0] + v_segs[:, 2]) / 2.0)

    # Cluster H segments by Y (window top/bottom)
    h_ys_sorted = np.sort(h_ys)
    h_clusters = []
    current_cluster = [h_ys_sorted[0]]
    for y in h_ys_sorted[1:]:
        if y - current_cluster[-1] < 40:
            current_cluster.append(y)
        else:
            h_clusters.append(np.mean(current_cluster))
            current_cluster = [y]
    h_clusters.append(np.mean(current_cluster))

    # Cluster V segments by X (window left/right)
    v_xs_sorted = np.sort(v_xs)
    v_clusters = []
    current_cluster = [v_xs_sorted[0]]
    for x in v_xs_sorted[1:]:
        if x - current_cluster[-1] < 40:
            current_cluster.append(x)
        else:
            v_clusters.append(np.mean(current_cluster))
            current_cluster = [x]
    v_clusters.append(np.mean(current_cluster))

    min_rect_area = (h * w) * 0.004   # minimum 0.4% of image = real window
    max_rect_area = (h * w) * 0.55    # maximum 55% = entire wall of windows

    # Test all pairs of H-clusters and V-clusters for a valid window rectangle
    for i in range(len(h_clusters)):
        for j in range(i + 1, len(h_clusters)):
            y_top    = int(min(h_clusters[i], h_clusters[j]))
            y_bottom = int(max(h_clusters[i], h_clusters[j]))
            rect_h   = y_bottom - y_top
            if rect_h < 30:
                continue  # too thin vertically
            for p in range(len(v_clusters)):
                for q in range(p + 1, len(v_clusters)):
                    x_left  = int(min(v_clusters[p], v_clusters[q]))
                    x_right = int(max(v_clusters[p], v_clusters[q]))
                    rect_w  = x_right - x_left
                    if rect_w < 30:
                        continue  # too thin horizontally
                    area = rect_h * rect_w
                    if area < min_rect_area or area > max_rect_area:
                        continue
                    # Check brightness overlap — is this rectangle actually bright?
                    roi = bright_mask[y_top:y_bottom, x_left:x_right]
                    if roi.size == 0:
                        continue
                    bright_fraction = float(roi.mean())
                    if bright_fraction > 0.35:
                        # Geometric + photometric confirmation: this IS a window
                        cv2.rectangle(geom_mask, (x_left, y_top), (x_right, y_bottom), 1.0, -1)
                        print(f"    Window rect: ({x_left},{y_top})-({x_right},{y_bottom})"
                              f" bright={bright_fraction:.2f}")

    return geom_mask


def differential_window_mask(ambient_bgr: np.ndarray, darkest_bgr: np.ndarray) -> np.ndarray:
    """
    Detects windows by measuring the absolute pixel DROP between the bright ambient
    frame and the darkest bracket. Window panes drop 130+ levels; interior walls
    drop minimally — this mathematically separates them even when walls are white/glaring.
    """
    h, w = ambient_bgr.shape[:2]
    gray_ambient = cv2.cvtColor(ambient_bgr, cv2.COLOR_BGR2GRAY).astype(np.int16)
    gray_darkest = cv2.cvtColor(darkest_bgr, cv2.COLOR_BGR2GRAY).astype(np.int16)

    # How much did each pixel drop? Windows drop massively; walls drop uniformly.
    pixel_delta = np.clip(gray_ambient - gray_darkest, 0, 255).astype(np.uint8)

    # Pixels that dropped >120 levels are architectural openings
    _, window_zones = cv2.threshold(pixel_delta, 120, 255, cv2.THRESH_BINARY)

    # Connect panes across mullions — kernel scales with image size
    kernel_size = max(int(max(h, w) * 0.015), 3)
    if kernel_size % 2 == 0:
        kernel_size += 1
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
    closed_mask = cv2.morphologyEx(window_zones, cv2.MORPH_CLOSE, kernel)

    # Connected components — remove tiny TV/mirror reflections and full-frame blobs
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(closed_mask, connectivity=8)
    geom_mask = np.zeros((h, w), dtype=np.uint8)
    min_area = int((h * w) * 0.005)
    max_area = int((h * w) * 0.60)

    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        if min_area <= area <= max_area:
            geom_mask[labels == i] = 255
            print(f"    Delta window component {i}: area={area} px ({area/(h*w)*100:.1f}%)")

    # Feather edges for clean blending
    feathered = cv2.GaussianBlur(geom_mask.astype(np.float32), (21, 21), 0)
    return np.clip(feathered / 255.0, 0, 1)


def build_window_mask_from_raw(dark_raw: np.ndarray, bright_raw: np.ndarray, sigma: float = 12.0) -> np.ndarray:
    """
    Hybrid geometric + photometric window mask, inspired by Autoenhance's M-LSD approach.

    Phase A — Photometric (brightness threshold):
      Works for clearly blown-out windows on a dark bracket.
      Fails on overcast days, recessed windows, or curtained windows.

    Phase B — Geometric (LSD line segment detector):
      Finds window frame rectangles from structural lines in the image.
      Cross-references with Phase A brightness to confirm windows.
      Adds geometric precision: straight hard edges where the photometric
      mask would feather incorrectly (mullions, frame edges).

    Final mask = Phase A (soft) ∪ Phase B (hard, geometry-confirmed)
    """
    h, w = dark_raw.shape[:2]
    f    = dark_raw.astype(np.float32) / 255.0
    lum  = 0.299 * f[:, :, 2] + 0.587 * f[:, :, 1] + 0.114 * f[:, :, 0]

    # ── Phase A: Photometric mask ─────────────────────────────────────────────
    # Threshold 0.55 — aggressively catch all bright window zones on dark frame
    hard_thresh = (lum > 0.55).astype(np.uint8) * 255
    # Large morphological close to fill window frame gaps and sill spill
    kernel      = cv2.getStructuringElement(cv2.MORPH_RECT, (35, 35))
    hard_closed = cv2.morphologyEx(hard_thresh, cv2.MORPH_CLOSE, kernel)
    # Also dilate to extend mask beyond the raw bright zone (catches frame edges)
    dilate_k    = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (25, 25))
    hard_closed = cv2.dilate(hard_closed, dilate_k)
    contours, _ = cv2.findContours(hard_closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    photo_mask  = np.zeros((h, w), dtype=np.float32)
    min_window_area = (h * w) * 0.002

    for cnt in contours:
        if cv2.contourArea(cnt) >= min_window_area:
            cv2.drawContours(photo_mask, [cnt], -1, 1.0, thickness=cv2.FILLED)

    # Generous soft spill around bright zones to pull in frame/sill regions
    spill = np.clip((lum - 0.60) / (1.0 - 0.60 + 1e-6), 0, 1) ** 1.2
    photo_combined = np.clip(photo_mask + spill * 0.7, 0, 1)

    # ── Phase B: Differential delta mask ─────────────────────────────────────
    # Pixel drop between bright and dark frames isolates true window panes
    # even when walls are white/glaring (the old brightness threshold failure mode)
    print("  Running differential delta window detection...")
    delta_mask = differential_window_mask(bright_raw, dark_raw)
    combined = np.clip(photo_combined + delta_mask, 0, 1)

    # Final feather pass
    combined = cv2.GaussianBlur(combined.astype(np.float32), (0, 0), sigmaX=sigma, sigmaY=sigma)
    combined = np.clip(combined, 0, 1)

    print(f"  Window mask: {combined.mean()*100:.1f}% of image covered")
    return combined


# ---------------------------------------------------------------------------
# Select best window-detail bracket
# Autoenhance: highest gradient (edge density) inside the window zone
# ---------------------------------------------------------------------------

def select_best_window_frame(images_raw: List[np.ndarray], win_mask: np.ndarray) -> np.ndarray:
    best_idx   = 0
    best_score = -1.0
    mask_bool  = win_mask > 0.3

    for i, img in enumerate(images_raw):
        gray  = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)
        lap   = np.abs(cv2.Laplacian(gray, cv2.CV_32F))
        score = float(lap[mask_bool].mean()) if mask_bool.any() else 0.0
        print(f"  Window frame {i} detail score: {score:.2f}")
        if score > best_score:
            best_score = score
            best_idx   = i

    print(f"  Selected frame {best_idx} as best window source")
    return images_raw[best_idx]


# ---------------------------------------------------------------------------
# Ghost detection / deghosting
# ---------------------------------------------------------------------------

def deghost(images: List[np.ndarray], ref_idx: int) -> List[np.ndarray]:
    ref = cv2.cvtColor(images[ref_idx], cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
    diffs = []
    for i, img in enumerate(images):
        if i == ref_idx:
            diffs.append(np.zeros_like(ref)); continue
        g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
        diffs.append(np.abs(g - ref))
    max_diff = np.max(np.stack(diffs, axis=0), axis=0)
    ghost_prob = cv2.GaussianBlur(np.clip((max_diff - 0.12) / 0.20, 0, 1), (0, 0), 6, 6)
    clean_w = (1.0 - np.clip(ghost_prob, 0, 0.85))[:, :, np.newaxis]
    ref_f = images[ref_idx].astype(np.float32)
    result = []
    for i, img in enumerate(images):
        if i == ref_idx:
            result.append(img); continue
        result.append(np.clip(img.astype(np.float32) * clean_w + ref_f * (1 - clean_w), 0, 255).astype(np.uint8))
    return result


# ---------------------------------------------------------------------------
# Synthetic brackets (single-shot fallback)
# ---------------------------------------------------------------------------

def synthesize_brackets(img: np.ndarray) -> List[np.ndarray]:
    f = img.astype(np.float32) / 255.0
    dark   = np.clip(np.power(f, 1.8),  0, 1)
    bright = np.clip(np.power(f, 0.45), 0, 1)
    print("Synthesized virtual brackets.")
    return [(dark * 255).astype(np.uint8), img, (bright * 255).astype(np.uint8)]


# ---------------------------------------------------------------------------
# Tone-curve LUT
# ---------------------------------------------------------------------------

def apply_tone_curve(img_f: np.ndarray, curve_in: List[float], curve_out: List[float]) -> np.ndarray:
    lut    = np.interp(np.linspace(0, 1, 256), curve_in, curve_out).astype(np.float32)
    img_u8 = np.clip(img_f * 255, 0, 255).astype(np.uint8)
    return lut[img_u8].astype(np.float32)


# ---------------------------------------------------------------------------
# Large-radius unsharp mask on L-channel (3D pop, no grain)
# ---------------------------------------------------------------------------

def neutralize_yellow_cast(img_bgr: np.ndarray) -> np.ndarray:
    """
    Fixes yellow ceiling/wall cast using LAB b-channel pull on bright areas.
    Keeps floor/lamp warmth intact while pushing ceilings toward clean neutral white.
    """
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    l, a, b = lab[:, :, 0], lab[:, :, 1], lab[:, :, 2]

    # Bright areas (L > 200/255 ≈ 0.78) are ceilings and walls
    bright_w = np.clip((l - 180.0) / 40.0, 0, 1)

    # Pull b-channel toward 128 (neutral) in bright zones — removes yellow without touching floors
    b_neutral = b * (1.0 - bright_w * 0.55) + 128.0 * (bright_w * 0.55)
    # Also pull a-channel slightly toward neutral to kill green-yellow
    a_neutral = a * (1.0 - bright_w * 0.25) + 128.0 * (bright_w * 0.25)

    lab[:, :, 1] = np.clip(a_neutral, 0, 255)
    lab[:, :, 2] = np.clip(b_neutral, 0, 255)
    return cv2.cvtColor(np.clip(lab, 0, 255).astype(np.uint8), cv2.COLOR_LAB2BGR)


def correct_vertical_perspective(img: np.ndarray) -> np.ndarray:
    """
    Detects wide-angle vertical tilt via Hough lines and applies a
    small rotation to straighten leaning verticals. Safe — only fires
    when ≥5 near-vertical lines agree and tilt > 0.1°.
    """
    try:
        gray  = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150, apertureSize=3)
        lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=100,
                                minLineLength=150, maxLineGap=10)
        if lines is None or len(lines) == 0:
            return img
        angles = []
        for line in lines:
            x1, y1, x2, y2 = line[0]
            if abs(x1 - x2) < 15:   # near-vertical segment
                angle = np.arctan2(y2 - y1, x2 - x1) * 180.0 / np.pi
                angles.append(angle)
        if len(angles) < 5:
            return img
        median_angle = float(np.median(angles))
        tilt = median_angle - 90.0 if median_angle > 0 else median_angle + 90.0
        if abs(tilt) < 0.1:
            return img
        h, w = img.shape[:2]
        M = cv2.getRotationMatrix2D((w // 2, h // 2), tilt, 1.0)
        return cv2.warpAffine(img, M, (w, h),
                              flags=cv2.INTER_CUBIC,
                              borderMode=cv2.BORDER_REPLICATE)
    except Exception as e:
        print(f"  Perspective correction skipped: {e}")
        return img


def anchor_ambient_shadows(img: np.ndarray) -> np.ndarray:
    """
    Darkens the deepest shadow regions (L < 45) to reclaim structural
    contrast depth destroyed by Mertens milkiness. Applied BEFORE final
    sharpening so the shadow edges are also sharpened afterward.
    """
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    _, shadow_mask = cv2.threshold(l, 65, 255, cv2.THRESH_BINARY_INV)
    shadow_blur = cv2.GaussianBlur(shadow_mask, (15, 15), 0)
    l_f = l.astype(np.float32)
    factor = 1.0 - (shadow_blur.astype(np.float32) / 255.0) * 0.35
    l_anchored = np.clip(l_f * factor, 0, 255).astype(np.uint8)
    return cv2.cvtColor(cv2.merge((l_anchored, a, b)), cv2.COLOR_LAB2BGR)


def intelligent_white_balance(img: np.ndarray) -> np.ndarray:
    """
    Neutralizes yellow/brown cast on bright flat surfaces (ceiling, trim, upper walls)
    without draining warmth from floors and furniture.
    Pulls both a and b channels toward neutral 128 by 70% in L>160 zones.
    """
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    _, ceiling_zone = cv2.threshold(l, 140, 255, cv2.THRESH_BINARY)
    ceiling_blur = cv2.GaussianBlur(ceiling_zone, (21, 21), 0)
    mask_w = ceiling_blur.astype(np.float32) / 255.0
    a_f = a.astype(np.float32)
    b_f = b.astype(np.float32)
    l_f = l.astype(np.float32)
    pull = 0.85
    a_corrected = (a_f * (1.0 - mask_w * pull)) + (128.0 * (mask_w * pull))
    b_corrected = (b_f * (1.0 - mask_w * pull)) + (128.0 * (mask_w * pull))
    # Also lift L in ceiling zones to make them truly bright white
    l_corrected = l_f + mask_w * (240.0 - l_f) * 0.35
    a_corrected = np.clip(a_corrected, 0, 255).astype(np.uint8)
    b_corrected = np.clip(b_corrected, 0, 255).astype(np.uint8)
    l_corrected = np.clip(l_corrected, 0, 255).astype(np.uint8)
    return cv2.cvtColor(cv2.merge((l_corrected, a_corrected, b_corrected)), cv2.COLOR_LAB2BGR)


def local_contrast_enhance(img_bgr: np.ndarray, radius: float = 45.0, amount: float = 0.20) -> np.ndarray:
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    l   = lab[:, :, 0] / 255.0
    blurred = cv2.GaussianBlur(l, (0, 0), sigmaX=radius, sigmaY=radius)
    lab[:, :, 0] = np.clip(l + amount * (l - blurred), 0, 1) * 255.0
    return cv2.cvtColor(np.clip(lab, 0, 255).astype(np.uint8), cv2.COLOR_LAB2BGR)


# ---------------------------------------------------------------------------
# Enfuse-based exposure fusion (industry-standard, cleaner than Mertens)
# ---------------------------------------------------------------------------

def enfuse_available() -> bool:
    return shutil.which("enfuse") is not None


def enfuse_merge(images: List[np.ndarray], output_width: int, output_height: int) -> Optional[np.ndarray]:
    """
    Runs Enfuse on the given BGR uint8 frames and returns the fused result.
    Returns None if enfuse is unavailable or fails (caller falls back to Mertens).
    """
    if not enfuse_available():
        print("  Enfuse binary not found — falling back to Mertens")
        return None

    tmp_dir = tempfile.mkdtemp(prefix="enfuse_", dir="/tmp")
    in_paths = []
    out_path = os.path.join(tmp_dir, "fused.tif")
    try:
        for i, img in enumerate(images):
            p = os.path.join(tmp_dir, f"frame_{i:02d}.tif")
            cv2.imwrite(p, img)
            in_paths.append(p)

        cmd = [
            "enfuse",
            "--depth=8",
            "--exposure-weight=1.0",
            "--saturation-weight=0.2",
            "--contrast-weight=0.0",
            "--exposure-optimum=0.5",
            "--exposure-width=0.2",
            "--no-ciecam",
            "--output=" + out_path,
        ] + in_paths

        print(f"  Running enfuse on {len(in_paths)} frames...")
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
        if proc.returncode != 0:
            print(f"  Enfuse failed (rc={proc.returncode}): {proc.stderr[:500]}")
            return None

        result = cv2.imread(out_path, cv2.IMREAD_COLOR)
        if result is None:
            print("  Enfuse output unreadable")
            return None
        result = cv2.resize(result, (output_width, output_height), interpolation=cv2.INTER_AREA)
        print(f"  Enfuse OK. Output mean: {cv2.cvtColor(result, cv2.COLOR_BGR2GRAY).mean():.1f}")
        return result
    except Exception as e:
        print(f"  Enfuse error: {e}")
        return None
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


# ---------------------------------------------------------------------------
# CORE MERGE PIPELINE
# ---------------------------------------------------------------------------

def bracket_merge(file_urls: List[str]) -> np.ndarray:
    """
    Simple Enfuse-based HDR merge.

    Lets Enfuse handle the exposure fusion directly on the decoded RAW frames.
    Enfuse uses the real exposure differences between frames to decide which
    frame to weight where — so we do NOT pre-normalise exposures (that would
    destroy the very information Enfuse relies on).

    Falls back to OpenCV Mertens only if the Enfuse binary is unavailable.
    """
    tmp_paths = []
    try:
        # ── Download & decode ─────────────────────────────────────────────────
        print(f"Downloading {len(file_urls)} frames...")
        for url in file_urls:
            ext = url.split("?")[0].rsplit(".", 1)[-1]
            ext = f".{ext.lower()}" if ext else ".jpg"
            tmp_paths.append(download_file(url, ext))

        raw_images = []
        for p in tmp_paths:
            img = load_image_bgr(p)
            img = cv2.resize(img, (OUTPUT_WIDTH, OUTPUT_HEIGHT), interpolation=cv2.INTER_AREA)
            raw_images.append(img)
        print(f"Loaded {len(raw_images)} frames at {OUTPUT_WIDTH}×{OUTPUT_HEIGHT}")

        single_shot = len(raw_images) == 1
        if single_shot:
            raw_images = synthesize_brackets(raw_images[0])

        # ── Light denoise (preserves edges, cleans RAW noise) ─────────────────
        print("Denoising (bilateral)...")
        raw_images = [cv2.bilateralFilter(img, d=5, sigmaColor=45, sigmaSpace=45)
                      for img in raw_images]
        gc.collect()

        # ── Enfuse — let it do the exposure fusion ────────────────────────────
        merged = enfuse_merge(raw_images, OUTPUT_WIDTH, OUTPUT_HEIGHT)

        if merged is None:
            # Fallback: Mertens (only if Enfuse binary missing/failed)
            print("Mertens fallback fusion...")
            fused = cv2.createMergeMertens(
                contrast_weight=1.5,
                saturation_weight=1.2,
                exposure_weight=0.0,
            ).process(raw_images)
            merged = np.clip(fused * 255, 0, 255).astype(np.uint8)
            del fused; gc.collect()

        mean_out = cv2.cvtColor(merged, cv2.COLOR_BGR2GRAY).mean()
        print(f"Merge complete. Output mean brightness: {mean_out:.1f}/255")
        return merged

    finally:
        for p in tmp_paths:
            try: os.unlink(p)
            except Exception: pass


# ---------------------------------------------------------------------------
# POST-PROCESSING FINISH
# ---------------------------------------------------------------------------

def apply_autohdr_finish(img_bgr: np.ndarray) -> np.ndarray:
    """
    Clean finish pass — clarity (local contrast) + light sharpen + dither.
    NO CLAHE: histogram equalization creates visible tile banding on smooth
    ceilings/walls. Clarity (Gaussian-blur unsharp) is continuous and clean.
    Dithering breaks residual 8-bit gradient banding.
    """
    # ── 1. Clarity: large-radius unsharp on L-channel (midtone pop) ───────────
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    l = lab[:, :, 0] / 255.0
    blurred = cv2.GaussianBlur(l, (0, 0), sigmaX=25, sigmaY=25)
    l = np.clip(l + 0.25 * (l - blurred), 0, 1)
    lab[:, :, 0] = l * 255.0
    img_f = cv2.cvtColor(np.clip(lab, 0, 255).astype(np.uint8), cv2.COLOR_LAB2BGR).astype(np.float32)

    # ── 2. Light sharpen (unsharp mask, small radius — edge pop) ──────────────
    blur_s = cv2.GaussianBlur(img_f, (0, 0), sigmaX=1.5, sigmaY=1.5)
    img_f = np.clip(img_f + 0.3 * (img_f - blur_s), 0, 255)

    # ── 3. Dither: add ±1 noise to break gradient banding on ceilings/walls ──
    noise = np.random.randint(-1, 2, img_f.shape[:2], dtype=np.int16)
    img_f = np.clip(img_f + noise[:, :, None], 0, 255)

    result = img_f.astype(np.uint8)
    mean_final = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY).mean()
    print(f"Finish applied. Final mean: {mean_final:.1f}/255")
    return result


# ---------------------------------------------------------------------------
# Endpoint
# ---------------------------------------------------------------------------

class MergeRequest(BaseModel):
    file_urls: List[str]
    bracket_name: str = "bracket"
    imagine_api_key: Optional[str] = None
    photomatix_api_key: Optional[str] = None
    replace_sky: bool = False


@app.post("/merge")
async def merge_hdr(req: MergeRequest):
    if not req.file_urls:
        raise HTTPException(400, "No file URLs provided")
    if len(req.file_urls) not in (1, 3, 5):
        raise HTTPException(400, f"Expected 1, 3, or 5 files, got {len(req.file_urls)}")
    try:
        print(f"Starting HDR merge for '{req.bracket_name}' ({len(req.file_urls)} frames)...")
        merged = bracket_merge(req.file_urls)
        merged = apply_autohdr_finish(merged)

        pil = Image.fromarray(cv2.cvtColor(merged, cv2.COLOR_BGR2RGB))
        del merged; gc.collect()
        buf = io.BytesIO()
        pil.save(buf, format="JPEG", quality=94, optimize=True)
        del pil; gc.collect()
        buf.seek(0)
        jpg_b64 = base64.b64encode(buf.read()).decode("utf-8")
        del buf

        return {"success": True, "bracket_name": req.bracket_name,
                "width": OUTPUT_WIDTH, "height": OUTPUT_HEIGHT, "jpeg_base64": jpg_b64}

    except Exception as e:
        tb = traceback.format_exc()
        print(f"ERROR in /merge: {tb}")
        raise HTTPException(500, detail=f"{str(e)}\n\nTraceback:\n{tb}")


# ---------------------------------------------------------------------------
# SINGLE-PHOTO ENHANCEMENT (in-camera HDR → finished photo)
# ---------------------------------------------------------------------------

def detect_window_mask_single(img_bgr: np.ndarray) -> np.ndarray:
    """
    Detect blown window panes from a single photo.
    In-camera HDR exposes the interior well but typically blows out windows.
    Thresholding for near-clipped (L > 235) isolates window panes specifically —
    ceilings/walls in a good in-camera HDR won't be clipped.
    """
    h, w = img_bgr.shape[:2]
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    l = lab[:, :, 0].astype(np.float32)

    # Blown core (no recoverable detail) → sky replacement target
    blown = (l > 235).astype(np.uint8) * 255
    # Close gaps across mullions / frame bars
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 25))
    closed = cv2.morphologyEx(blown, cv2.MORPH_CLOSE, kernel)
    # Dilate slightly to cover the pane edge
    dilate_k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    closed = cv2.dilate(closed, dilate_k)

    # Drop tiny reflections / keep real window panes
    num, labels, stats, _ = cv2.connectedComponentsWithStats(closed, 8)
    mask = np.zeros((h, w), dtype=np.uint8)
    min_area = int((h * w) * 0.003)
    for i in range(1, num):
        if stats[i, cv2.CC_STAT_AREA] >= min_area:
            mask[labels == i] = 255

    feathered = cv2.GaussianBlur(mask.astype(np.float32), (21, 21), 0)
    return np.clip(feathered / 255.0, 0, 1)


def generate_sky(h: int, w: int) -> np.ndarray:
    """
    Procedural natural blue sky: vertical gradient + soft clouds.
    BGR float32 [0..255].
    """
    # Gradient: deeper blue at top → hazy light near horizon
    top   = np.array([170, 130, 70], dtype=np.float32)    # BGR deep blue
    horz  = np.array([225, 205, 185], dtype=np.float32)   # BGR warm haze
    grad  = np.zeros((h, w, 3), dtype=np.float32)
    ys    = np.linspace(0, 1, h)[:, None] ** 0.7
    grad[:] = top * (1 - ys[:, :, None]) + horz * ys[:, :, None]

    # Soft procedural clouds from blurred noise
    noise = np.random.rand(h, w).astype(np.float32)
    cloud = cv2.GaussianBlur(noise, (0, 0), sigmaX=w / 6)
    cloud = (cloud - cloud.min()) / (cloud.max() - cloud.min() + 1e-6)
    cloud = np.clip((cloud - 0.58) / 0.25, 0, 1)[:, :, None]
    white = np.array([245, 245, 245], dtype=np.float32)
    grad  = grad * (1 - cloud * 0.45) + white * (cloud * 0.45)
    return np.clip(grad, 0, 255)


def smart_resize(img: np.ndarray, max_dim: int = 3000) -> np.ndarray:
    """
    Resize so the longest side is at most max_dim, preserving aspect ratio.
    3000px keeps HD quality without the memory cost of full-resolution RAW.
    """
    h, w = img.shape[:2]
    longest = max(h, w)
    if longest <= max_dim:
        return img
    scale = max_dim / longest
    new_w, new_h = int(w * scale), int(h * scale)
    return cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)


# ---------------------------------------------------------------------------
# Lightroom-style adjustment primitives (masked, feathered, float32)
# ---------------------------------------------------------------------------

def enhance_single(file_url: str, replace_sky: bool = False) -> np.ndarray:
    """
    Lightroom-style real estate enhancement — SINGLE float32 LAB pass.

    All L-channel (exposure, highlights, shadows, whites/blacks, clarity) and
    a/b-channel (white balance, saturation) adjustments happen in ONE LAB
    round-trip to avoid 8-bit quantization banding on smooth ceilings/walls.
    Previous pipeline had 6 separate LAB conversions — each one quantized the
    ceiling gradient to 8-bit, creating visible horizontal banding lines.

    Saturation is masked to EXCLUDE bright zones (ceilings) so the white balance
    neutralization of green window-light spill isn't undone by re-amplification.
    """
    ext = file_url.split("?")[0].rsplit(".", 1)[-1]
    ext = f".{ext.lower()}" if ext else ".jpg"
    tmp = download_file(file_url, ext)
    try:
        img = load_image_bgr(tmp)
        img = smart_resize(img, max_dim=3000)
        h, w = img.shape[:2]
        print(f"Loaded single photo at {w}×{h}")

        # 1. Light denoise (preserve edges, don't soften)
        img = cv2.bilateralFilter(img, d=4, sigmaColor=30, sigmaSpace=30)

        # 2. SINGLE float32 LAB pass — ALL tone & color adjustments here
        #    One BGR→LAB conversion, one LAB→BGR conversion = no banding
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB).astype(np.float32)
        l = lab[:, :, 0].copy()
        a = lab[:, :, 1].copy()
        b = lab[:, :, 2].copy()

        # 2a. Exposure lift (gamma on L — high-key brightness)
        l = np.clip(np.power(l / 255.0, 0.48) * 255.0, 0, 255)

        # 2b. Lower highlights (masked to bright zones, feathered)
        hl_mask = np.clip((l - 215.0) / 40.0, 0, 1) ** 1.5
        hl_mask = cv2.GaussianBlur(hl_mask, (0, 0), sigmaX=20, sigmaY=20)
        l = np.clip(l - hl_mask * 45.0, 0, 255)

        # 2c. Lift shadows (masked to dark zones, feathered)
        sh_mask = np.clip((80.0 - l) / 80.0, 0, 1) ** 1.2
        sh_mask = cv2.GaussianBlur(sh_mask, (0, 0), sigmaX=20, sigmaY=20)
        l = np.clip(l + sh_mask * 65.0, 0, 255)

        # 2d. Whites & blacks (clean white ceilings, no crushed blacks)
        white_mask = np.clip((l - 200.0) / 55.0, 0, 1)
        black_mask = np.clip((40.0 - l) / 40.0, 0, 1)
        l = np.clip(l + white_mask * 18.0 + black_mask * 15.0, 0, 255)

        # 2e. White balance — neutralize green/yellow in bright zones (ceilings)
        #     85% pull toward neutral — kills window-light green spill on walls
        bright_w = np.clip((l - 160.0) / 40.0, 0, 1)
        bright_w = cv2.GaussianBlur(bright_w, (0, 0), sigmaX=15, sigmaY=15)
        wb_pull = 0.85
        a = a * (1.0 - bright_w * wb_pull) + 128.0 * (bright_w * wb_pull)
        b = b * (1.0 - bright_w * wb_pull) + 128.0 * (bright_w * wb_pull)

        # 2f. Saturation — boost ONLY in mid/dark zones (plants, furniture)
        #     Bright zones excluded so WB neutralization on ceilings holds
        sat_mask = 1.0 - bright_w
        sat_boost = 0.30
        a = np.clip(a + (a - 128.0) * sat_boost * sat_mask, 0, 255)
        b = np.clip(b + (b - 128.0) * sat_boost * sat_mask, 0, 255)

        # 2g. Clarity — large-radius unsharp on L (midtone pop, no CLAHE)
        l_norm = l / 255.0
        l_blur = cv2.GaussianBlur(l_norm, (0, 0), sigmaX=25, sigmaY=25)
        l_norm = np.clip(l_norm + 0.25 * (l_norm - l_blur), 0, 1)
        l = l_norm * 255.0

        # Convert back to BGR — single quantization, no banding
        lab[:, :, 0] = l
        lab[:, :, 1] = a
        lab[:, :, 2] = b
        img = cv2.cvtColor(np.clip(lab, 0, 255).astype(np.uint8), cv2.COLOR_LAB2BGR)
        print(f"LAB pass done. Mean: {cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).mean():.1f}")

        # 3. Sky replacement (optional, off by default)
        if replace_sky:
            win_mask = detect_window_mask_single(img)
            if win_mask.max() > 0.01:
                sky = generate_sky(h, w)
                mask3 = win_mask[:, :, None]
                img = np.clip(img.astype(np.float32) * (1 - mask3) + sky * mask3, 0, 255).astype(np.uint8)
                print("Sky replaced in window zones")

        # 4. Light sharpen (BGR float32 — no LAB round-trip needed)
        img_f = img.astype(np.float32)
        blur_s = cv2.GaussianBlur(img_f, (0, 0), sigmaX=1.5, sigmaY=1.5)
        img_f = np.clip(img_f + 0.3 * (img_f - blur_s), 0, 255)

        # 5. Dither — ±1 noise breaks residual 8-bit gradient banding
        noise = np.random.randint(-1, 2, img_f.shape[:2], dtype=np.int16)
        img_f = np.clip(img_f + noise[:, :, None], 0, 255)

        result = img_f.astype(np.uint8)
        print(f"Enhancement complete. Final: {result.shape[1]}×{result.shape[0]}")
        return result
    finally:
        try: os.unlink(tmp)
        except Exception: pass


class EnhanceRequest(BaseModel):
    file_url: str
    bracket_name: str = "enhance"
    replace_sky: bool = False


@app.post("/enhance")
async def enhance_hdr(req: EnhanceRequest):
    if not req.file_url:
        raise HTTPException(400, "No file URL provided")
    try:
        print(f"Starting single-photo enhancement for '{req.bracket_name}'...")
        result = enhance_single(req.file_url, req.replace_sky)
        out_h, out_w = result.shape[:2]

        pil = Image.fromarray(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))
        del result; gc.collect()
        buf = io.BytesIO()
        pil.save(buf, format="JPEG", quality=96, optimize=True)
        del pil; gc.collect()
        buf.seek(0)
        jpg_b64 = base64.b64encode(buf.read()).decode("utf-8")
        del buf

        return {"success": True, "bracket_name": req.bracket_name,
                "width": out_w, "height": out_h, "jpeg_base64": jpg_b64}

    except Exception as e:
        tb = traceback.format_exc()
        print(f"ERROR in /enhance: {tb}")
        raise HTTPException(500, detail=f"{str(e)}\n\nTraceback:\n{tb}")


@app.get("/health")
def health():
    return {"status": "ok"}
