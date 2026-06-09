from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional, Tuple
import numpy as np
import cv2
import requests
import tempfile
import os
from PIL import Image, ImageFilter
import io
import base64
import gc
import traceback

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


def build_geometric_window_candidates_v2(ambient_bgr: np.ndarray, darkest_bgr: np.ndarray) -> np.ndarray:
    """
    Replaces the nested-loop LSD cluster search.
    Uses luminance-differential thresholding + connected components to isolate
    true architectural window openings without combinatorial blind spots.
    """
    h, w = ambient_bgr.shape[:2]
    gray_ambient = cv2.cvtColor(ambient_bgr, cv2.COLOR_BGR2GRAY)
    gray_darkest = cv2.cvtColor(darkest_bgr, cv2.COLOR_BGR2GRAY)

    # Blown zones in the bright/ambient frame = window candidates
    _, bright_seed = cv2.threshold(gray_ambient, 220, 255, cv2.THRESH_BINARY)

    # Dark frame: walls are pitch-black, windows are midtone — exclude dead-black pixels
    _, dark_wall_mask = cv2.threshold(gray_darkest, 15, 255, cv2.THRESH_BINARY_INV)

    # Intersect: true windows must be blown in bright AND not pitch-black in dark
    validated_seeds = cv2.bitwise_and(bright_seed, cv2.bitwise_not(dark_wall_mask))

    # Morphological close to fuse window panes across mullions/frames — scales with image size
    kernel_size = max(int(max(h, w) * 0.02), 3)
    if kernel_size % 2 == 0:
        kernel_size += 1
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
    closed_mask = cv2.morphologyEx(validated_seeds, cv2.MORPH_CLOSE, kernel)

    # Connected components: filter out tiny reflections (TV, mirrors) and full-image blobs
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(closed_mask, connectivity=8)
    geom_mask = np.zeros((h, w), dtype=np.uint8)
    min_area = int((h * w) * 0.005)   # 0.5% min — real window
    max_area = int((h * w) * 0.60)    # 60% max — not the whole image

    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        if min_area <= area <= max_area:
            geom_mask[labels == i] = 255
            print(f"    Window component {i}: area={area} px ({area/(h*w)*100:.1f}%)")

    return geom_mask.astype(np.float32) / 255.0


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

    # ── Phase B: Geometric mask (connected components) ───────────────────────
    print("  Running connected-components window detection...")
    geom_mask = build_geometric_window_candidates_v2(bright_raw, dark_raw)
    geom_mask_blurred = cv2.GaussianBlur(geom_mask, (0, 0), 6, 6)
    combined = np.clip(photo_combined + geom_mask_blurred, 0, 1)

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

def local_contrast_enhance(img_bgr: np.ndarray, radius: float = 45.0, amount: float = 0.20) -> np.ndarray:
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    l   = lab[:, :, 0] / 255.0
    blurred = cv2.GaussianBlur(l, (0, 0), sigmaX=radius, sigmaY=radius)
    lab[:, :, 0] = np.clip(l + amount * (l - blurred), 0, 1) * 255.0
    return cv2.cvtColor(np.clip(lab, 0, 255).astype(np.uint8), cv2.COLOR_LAB2BGR)


# ---------------------------------------------------------------------------
# CORE MERGE PIPELINE
# ---------------------------------------------------------------------------

def bracket_merge(file_urls: List[str]) -> np.ndarray:
    """
    Professional HDR merge — flambient/Autoenhance-inspired:

    Phase 1 — RAW frames (pre-normalisation):
      a. Download & decode
      b. Resize
      c. Build HYBRID window mask (LSD geometric + photometric brightness)
         from the RAW dark frame — BEFORE any normalisation
      d. Select best window-detail frame (by edge score inside window zone)

    Phase 2 — Exposure-normalised frames (for interior):
      e. Normalise bright frames to target mean ~115 (open interior)
      f. Normalise dark frame to a LOWER target ~75 (preserve window detail)
      g. AlignMTB
      h. Mertens fusion of normalised frames (ambient interior base)

    Phase 3 — Composite:
      i. Start from Mertens base (good interior exposure)
      j. Blend brightest normalised frame into remaining dark interior zones
      k. Composite window zones from best raw window frame (real exterior detail)
      l. Feathered blend using the pre-normalisation hybrid window mask
    """
    tmp_paths = []
    try:
        # ── Phase 1a-b: Download, decode, resize ─────────────────────────────
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

        # Sort raw frames dark→bright by mean luminance
        means_raw = [cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).mean() for img in raw_images]
        order_raw = list(np.argsort(means_raw))
        raw_images = [raw_images[i] for i in order_raw]
        means_raw  = [means_raw[i]  for i in order_raw]
        print(f"Raw frame means dark→bright: {[round(m, 1) for m in means_raw]}")

        single_shot = len(raw_images) == 1
        if single_shot:
            raw_images = synthesize_brackets(raw_images[0])
            means_raw  = [cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).mean() for img in raw_images]

        raw_dark_frame   = raw_images[0]
        raw_bright_frame = raw_images[-1]

        # ── Phase 1c: Build HYBRID window mask from RAW dark frame ────────────
        # CRITICAL: must be done before normalisation — after normalisation the
        # exposure differential is destroyed and window detection fails.
        # LSD geometric layer catches windows that aren't fully blown out
        # (overcast days, curtained windows, windows at angle) where the
        # pure brightness threshold misses them.
        print("Building hybrid connected-components + photometric window mask...")
        win_mask  = build_window_mask_from_raw(raw_dark_frame, raw_bright_frame, sigma=18.0)
        win_mask3 = win_mask[:, :, np.newaxis]
        interior3 = 1.0 - win_mask3

        # ── Phase 1d: Select best window-detail frame ─────────────────────────
        # For window pull we ALWAYS want the darkest raw frame — it has the most
        # exterior detail. Edge-score selection can pick a mid/bright frame when
        # window mullions are sharp there, which is the wrong source for exterior.
        print("Using darkest raw frame as window source (best exterior detail)...")
        best_window_frame = raw_images[0]  # darkest = most exterior detail
        print(f"  Dark frame mean: {means_raw[0]:.1f}")

        # ── Phase 2: Denoise raw frames ───────────────────────────────────────
        # Bilateral filter: smooths flat walls, preserves window frame/mullion edges
        print("Denoising (bilateral)...")
        raw_images = [cv2.bilateralFilter(img, d=9, sigmaColor=75, sigmaSpace=75)
                      for img in raw_images]
        best_window_frame = cv2.bilateralFilter(best_window_frame, d=9,
                            sigmaColor=75, sigmaSpace=75)
        gc.collect()

        # ── Phase 2e-f: Exposure normalisation ────────────────────────────────
        print("Normalising exposures...")
        norm_images = []
        for i, img in enumerate(raw_images):
            if i == 0:      # darkest: lower target preserves window contrast
                target = 75.0
            elif i == len(raw_images) - 1:  # brightest: open the interior
                target = 120.0
            else:           # mid frames
                target = 105.0
            norm_images.append(normalise_to_target(img, target))

        norm_bright_frame = norm_images[-1]
        norm_means = [cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).mean() for img in norm_images]
        print(f"Normalised means: {[round(m, 1) for m in norm_means]}")
        gc.collect()

        # ── Phase 2g: AlignMTB ─────────────────────────────────────────────────
        if len(norm_images) > 1:
            print("Aligning...")
            align = cv2.createAlignMTB(max_bits=6, exclude_range=4, cut=True)
            align.process(norm_images, norm_images)
            gc.collect()

        # ── Phase 2h: Mertens fusion (interior ambient base) ──────────────────
        if not single_shot and len(norm_images) > 1:
            norm_images = deghost(norm_images, ref_idx=len(norm_images) // 2)
            gc.collect()

        print("Mertens fusion (interior base)...")
        fused = cv2.createMergeMertens(
            contrast_weight=1.0,
            saturation_weight=0.8,
            exposure_weight=0.0,
        ).process(norm_images)
        mertens_base = np.clip(fused * 255, 0, 255).astype(np.uint8)
        del fused; gc.collect()

        # Post-fusion: very mild CLAHE — just enough to lift dark corners without halos
        print("Applying CLAHE to Mertens base...")
        lab = cv2.cvtColor(mertens_base, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=0.4, tileGridSize=(32, 32))
        l = clahe.apply(l)
        mertens_base = cv2.cvtColor(cv2.merge((l, a, b)), cv2.COLOR_LAB2BGR)

        # ── Phase 3i-j: Start from Mertens, boost dark interior zones ─────────
        print("Interior composite...")
        mertens_f = mertens_base.astype(np.float32) / 255.0
        bright_f  = norm_bright_frame.astype(np.float32) / 255.0
        lum_m     = 0.299 * mertens_f[:, :, 2] + 0.587 * mertens_f[:, :, 1] + 0.114 * mertens_f[:, :, 0]

        dark_interior_mask = np.clip(1.0 - lum_m / 0.45, 0, 1) ** 1.5
        dark_interior_mask = dark_interior_mask * interior3[:, :, 0]
        dark_interior_mask = cv2.GaussianBlur(dark_interior_mask.astype(np.float32), (0, 0), 10, 10)
        dark_interior_mask = np.clip(dark_interior_mask, 0, 0.70)[:, :, np.newaxis]

        interior_f = mertens_f + dark_interior_mask * (bright_f - mertens_f)
        interior_f = np.clip(interior_f, 0, 1)

        # ── Phase 3k: Window pull — mask from interior result, source from dark frame ─
        print("Window composite...")
        # Lift dark frame enough to show bright exterior (trees, sky, street visible)
        # Target ~80 gives a naturally lit exterior without blowing it out
        win_frame_normed = normalise_to_target(best_window_frame, target_mean=80.0)
        win_frame_f      = win_frame_normed.astype(np.float32) / 255.0

        # PRIMARY MASK: detect blown windows on the INTERIOR composite result
        # (mirrors the reference algorithm: threshold 240/255 = 0.94 on bright image)
        lum_interior = (0.299 * interior_f[:, :, 2] +
                        0.587 * interior_f[:, :, 1] +
                        0.114 * interior_f[:, :, 0])

        # Threshold at 0.85 — catch more blown window zones to reveal exterior detail
        blown_hard = (lum_interior > 0.85).astype(np.uint8) * 255
        # Small close to fill mullion gaps only — don't bleed onto curtains/couch
        close_k    = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
        blown_hard = cv2.morphologyEx(blown_hard, cv2.MORPH_CLOSE, close_k)
        # More aggressive dilation — pulls exterior detail further into frames
        dilate_k   = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
        blown_hard = cv2.dilate(blown_hard, dilate_k)
        # Smooth edges
        blown_mask = cv2.GaussianBlur(blown_hard.astype(np.float32) / 255.0,
                                      (21, 21), 0)
        blown_mask = np.clip(blown_mask, 0, 1)[:, :, np.newaxis]

        # Intersection: both blown_mask AND win_mask must agree — prevents
        # white painted frames from being darkened by the window pull
        combined_mask = blown_mask * np.clip(win_mask3 * 4, 0, 1)
        combined_mask = np.clip(combined_mask + win_mask3 * 0.15, 0, 1)

        # Blend: blown/window zones → exterior dark frame; room → interior
        composited_f = interior_f * (1.0 - combined_mask) + win_frame_f * combined_mask
        composited_f = np.clip(composited_f, 0, 1)

        print(f"  Window pull: {blown_mask.mean()*100:.1f}% of image replaced")

        composited = np.clip(composited_f * 255, 0, 255).astype(np.uint8)
        del mertens_f, bright_f, win_frame_f, composited_f, win_mask3, mertens_base
        gc.collect()

        mean_out = cv2.cvtColor(composited, cv2.COLOR_BGR2GRAY).mean()
        print(f"Merge complete. Output mean brightness: {mean_out:.1f}/255")
        return composited

    finally:
        for p in tmp_paths:
            try: os.unlink(p)
            except Exception: pass


# ---------------------------------------------------------------------------
# POST-PROCESSING FINISH
# ---------------------------------------------------------------------------

def apply_autohdr_finish(img_bgr: np.ndarray) -> np.ndarray:
    """
    Finish pass — tuned to match target: bright neutral beige tones, strong
    shadow fill on dark sides, controlled window highlights, no yellow-green cast.

    Parameters calibrated by AI diff analysis against target reference:
      gamma=0.42, highlight_start=0.72, fill_cutoff=0.38, fill_strength=0.32
      r_mult=1.04, g_mult=1.01, b_mult=0.93, vibrance=18, sharpen=0.55/1.2r

    Steps:
      1. Gamma lift (aggressive — interior was too dark overall)
      2. Highlight rolloff (starts at 0.72 — earlier than before to control windows)
      3. Large-radius USM on L-channel: 3D depth pop, no grain
      4. Wall/ceiling zone: desaturate + cast removal (removes yellow-green cast)
      5. Shadow fill — stronger cutoff (0.38) and fill (0.32) for dark corners
      6. Colour grade: R+4%, G+1%, B-7% — eliminates yellow-green cast globally
      7. Vibrance (18 units)
      8. Sharpening (0.55 amount, 1.2px radius)
    """
    img = img_bgr.astype(np.float32) / 255.0

    # ── Window protect mask ───────────────────────────────────────────────────
    lum_raw = 0.299 * img[:, :, 2] + 0.587 * img[:, :, 1] + 0.114 * img[:, :, 0]
    b_chan, g_chan, r_chan = img[:, :, 0], img[:, :, 1], img[:, :, 2]
    # Lower threshold to 0.58 — protect windows more aggressively from gamma lift
    lum_win  = np.clip((lum_raw - 0.58) / (1.0 - 0.58 + 1e-6), 0, 1) ** 1.2
    blue_dom = np.clip((b_chan - np.maximum(r_chan, g_chan) + 0.05) / 0.12, 0, 1)
    blue_dom = blue_dom * (lum_raw > 0.45).astype(np.float32)
    win_raw      = np.clip(lum_win * 0.8 + blue_dom * 0.2, 0, 1)
    win_protect  = cv2.GaussianBlur(win_raw.astype(np.float32), (0, 0), 12, 12)
    win_protect  = np.clip(win_protect, 0, 1)
    win_protect3 = win_protect[:, :, np.newaxis]
    interior3    = 1.0 - win_protect3

    # ── 1. Gamma lift — aggressive to match bright target ────────────────────
    # gamma=0.42 → strong lift, pushing mid-tones from muddy dark to bright beige
    gamma = 0.42
    img_interior = np.clip(np.power(np.clip(img, 0, 1), gamma), 0, 1)
    img = img_interior * interior3 + img * win_protect3
    img = np.clip(img, 0, 1)

    # ── 2. Highlight rolloff (interior only, protect windows) ────────────────
    # Rolls off highlights starting at 0.72 to cap bright walls at 0.88
    # This prevents the white ceiling/walls from blowing while interior is bright
    highlight_start  = 0.72
    highlight_cap    = 0.96
    highlight_str    = 0.65
    lum_h = 0.299 * img[:, :, 2] + 0.587 * img[:, :, 1] + 0.114 * img[:, :, 0]
    hl_mask = np.clip((lum_h - highlight_start) / (1.0 - highlight_start + 1e-6), 0, 1) ** 1.5
    hl_mask = hl_mask * interior3[:, :, 0]
    hl_mask3 = hl_mask[:, :, np.newaxis]
    # Blend toward highlight_cap in blown zones
    target_hl = img * (highlight_cap / (lum_h[:, :, np.newaxis] + 1e-6))
    target_hl = np.clip(target_hl, 0, 1)
    img = img * (1.0 - hl_mask3 * highlight_str) + target_hl * (hl_mask3 * highlight_str)
    img = np.clip(img, 0, 1)

    # ── 3. Large-radius USM on L-channel ─────────────────────────────────────
    img_u8 = (img * 255).astype(np.uint8)
    img_u8 = local_contrast_enhance(img_u8, radius=45.0, amount=0.20)
    img    = img_u8.astype(np.float32) / 255.0

    # ── 4. Wall/ceiling zone — desaturate + cast removal ─────────────────────
    lum2 = 0.299 * img[:, :, 2] + 0.587 * img[:, :, 1] + 0.114 * img[:, :, 0]
    wall_mask = np.clip((lum2 - 0.30) / (0.95 - 0.30), 0, 1)
    wall_mask = wall_mask * (1.0 - win_protect)
    wall_mask = cv2.GaussianBlur(wall_mask.astype(np.float32), (0, 0), 8, 8)
    wall_mask = np.clip(wall_mask, 0, 1)
    wall_mask3 = wall_mask[:, :, np.newaxis]

    # Stronger desaturation on walls to remove yellow-green cast
    img_u8_tmp = (img * 255).astype(np.uint8)
    hsv_tmp = cv2.cvtColor(img_u8_tmp, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv_tmp[:, :, 1] = hsv_tmp[:, :, 1] * (1.0 - wall_mask * 0.55)  # was 0.45
    img = cv2.cvtColor(np.clip(hsv_tmp, 0, 255).astype(np.uint8), cv2.COLOR_HSV2BGR).astype(np.float32) / 255.0

    # Grey-world cast removal (±10% cap — wider to fix yellow-green cast)
    wall_sum = wall_mask.sum() + 1e-6
    mean_r   = (img[:, :, 2] * wall_mask).sum() / wall_sum
    mean_g   = (img[:, :, 1] * wall_mask).sum() / wall_sum
    mean_b   = (img[:, :, 0] * wall_mask).sum() / wall_sum
    mean_all = (mean_r + mean_g + mean_b) / 3.0 + 1e-6
    for ch, mean_ch in [(2, mean_r), (1, mean_g), (0, mean_b)]:
        cor = np.clip(mean_all / (mean_ch + 1e-6), 0.90, 1.10)  # was 0.92/1.08
        img[:, :, ch] = np.clip(img[:, :, ch] * (1.0 + (cor - 1.0) * wall_mask * 0.35), 0, 1)  # was 0.25

    # Brightness push on walls
    img = img + wall_mask3 * 0.08 * (1.0 - img)
    img = np.clip(img, 0, 1)

    # ── 5. Shadow fill — stronger to lift dark left-side corners ─────────────
    # fill_cutoff=0.38 (was 0.28), fill_strength=0.32 (was 0.40 but capped lower)
    lum3       = 0.299 * img[:, :, 2] + 0.587 * img[:, :, 1] + 0.114 * img[:, :, 0]
    fill_mask  = np.clip(1.0 - lum3 / 0.38, 0, 1) ** 1.5   # wider cutoff catches more shadow
    fill_mask3 = fill_mask[:, :, np.newaxis]
    img = img + fill_mask3 * 0.32 * (1.0 - img) * interior3  # controlled strength
    img = np.clip(img, 0, 1)

    # ── 6. Colour grade — R+4%, G+1%, B-7% to remove yellow-green, add warmth ─
    img[:, :, 2] = np.clip(img[:, :, 2] * 1.04, 0, 1)  # R up
    img[:, :, 1] = np.clip(img[:, :, 1] * 1.01, 0, 1)  # G neutral
    img[:, :, 0] = np.clip(img[:, :, 0] * 0.93, 0, 1)  # B down — kills yellow-green cast

    # ── 7. Vibrance (18 units — moderate, interior surfaces only) ─────────────
    img_u8 = (img * 255).astype(np.uint8)
    hsv    = cv2.cvtColor(img_u8, cv2.COLOR_BGR2HSV).astype(np.float32)
    sat_n  = hsv[:, :, 1] / 255.0
    vib_zone  = np.clip(1.0 - wall_mask, 0, 1)
    vib_boost = 18.0 * (1.0 - sat_n) ** 1.5 * vib_zone  # was 12
    hsv[:, :, 1] = np.clip(hsv[:, :, 1] + vib_boost, 0, 255)
    img = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR).astype(np.float32) / 255.0

    # ── 8. Sharpening (0.55 amount, 1.2px radius) ────────────────────────────
    img_u8    = (img * 255).astype(np.uint8)
    pil_img   = Image.fromarray(cv2.cvtColor(img_u8, cv2.COLOR_BGR2RGB))
    blurred   = pil_img.filter(ImageFilter.GaussianBlur(radius=1.2))  # was 0.8
    sharpened = np.clip(
        np.array(pil_img).astype(np.float32) + 0.55 * (   # was 0.45
            np.array(pil_img).astype(np.float32) - np.array(blurred).astype(np.float32)
        ), 0, 255
    ).astype(np.uint8)
    result_bgr = cv2.cvtColor(sharpened, cv2.COLOR_RGB2BGR)

    mean_final = cv2.cvtColor(result_bgr, cv2.COLOR_BGR2GRAY).mean()
    print(f"Finish applied. Final mean: {mean_final:.1f}/255")
    return result_bgr


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


@app.get("/health")
def health():
    return {"status": "ok"}    scale = min(target_mean / current, max_scale)
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


def build_geometric_window_candidates_v2(ambient_bgr: np.ndarray, darkest_bgr: np.ndarray) -> np.ndarray:
    """
    Replaces the nested-loop LSD cluster search.
    Uses luminance-differential thresholding + connected components to isolate
    true architectural window openings without combinatorial blind spots.
    """
    h, w = ambient_bgr.shape[:2]
    gray_ambient = cv2.cvtColor(ambient_bgr, cv2.COLOR_BGR2GRAY)
    gray_darkest = cv2.cvtColor(darkest_bgr, cv2.COLOR_BGR2GRAY)

    # Blown zones in the bright/ambient frame = window candidates
    _, bright_seed = cv2.threshold(gray_ambient, 220, 255, cv2.THRESH_BINARY)

    # Dark frame: walls are pitch-black, windows are midtone — exclude dead-black pixels
    _, dark_wall_mask = cv2.threshold(gray_darkest, 15, 255, cv2.THRESH_BINARY_INV)

    # Intersect: true windows must be blown in bright AND not pitch-black in dark
    validated_seeds = cv2.bitwise_and(bright_seed, cv2.bitwise_not(dark_wall_mask))

    # Morphological close to fuse window panes across mullions/frames — scales with image size
    kernel_size = max(int(max(h, w) * 0.02), 3)
    if kernel_size % 2 == 0:
        kernel_size += 1
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
    closed_mask = cv2.morphologyEx(validated_seeds, cv2.MORPH_CLOSE, kernel)

    # Connected components: filter out tiny reflections (TV, mirrors) and full-image blobs
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(closed_mask, connectivity=8)
    geom_mask = np.zeros((h, w), dtype=np.uint8)
    min_area = int((h * w) * 0.005)   # 0.5% min — real window
    max_area = int((h * w) * 0.60)    # 60% max — not the whole image

    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        if min_area <= area <= max_area:
            geom_mask[labels == i] = 255
            print(f"    Window component {i}: area={area} px ({area/(h*w)*100:.1f}%)")

    return geom_mask.astype(np.float32) / 255.0


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

    # ── Phase B: Geometric mask (connected components) ───────────────────────
    print("  Running connected-components window detection...")
    geom_mask = build_geometric_window_candidates_v2(bright_raw, dark_raw)
    geom_mask_blurred = cv2.GaussianBlur(geom_mask, (0, 0), 6, 6)
    combined = np.clip(photo_combined + geom_mask_blurred, 0, 1)

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

def local_contrast_enhance(img_bgr: np.ndarray, radius: float = 45.0, amount: float = 0.20) -> np.ndarray:
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    l   = lab[:, :, 0] / 255.0
    blurred = cv2.GaussianBlur(l, (0, 0), sigmaX=radius, sigmaY=radius)
    lab[:, :, 0] = np.clip(l + amount * (l - blurred), 0, 1) * 255.0
    return cv2.cvtColor(np.clip(lab, 0, 255).astype(np.uint8), cv2.COLOR_LAB2BGR)


# ---------------------------------------------------------------------------
# CORE MERGE PIPELINE
# ---------------------------------------------------------------------------

def bracket_merge(file_urls: List[str]) -> np.ndarray:
    """
    Professional HDR merge — flambient/Autoenhance-inspired:

    Phase 1 — RAW frames (pre-normalisation):
      a. Download & decode
      b. Resize
      c. Build HYBRID window mask (LSD geometric + photometric brightness)
         from the RAW dark frame — BEFORE any normalisation
      d. Select best window-detail frame (by edge score inside window zone)

    Phase 2 — Exposure-normalised frames (for interior):
      e. Normalise bright frames to target mean ~115 (open interior)
      f. Normalise dark frame to a LOWER target ~75 (preserve window detail)
      g. AlignMTB
      h. Mertens fusion of normalised frames (ambient interior base)

    Phase 3 — Composite:
      i. Start from Mertens base (good interior exposure)
      j. Blend brightest normalised frame into remaining dark interior zones
      k. Composite window zones from best raw window frame (real exterior detail)
      l. Feathered blend using the pre-normalisation hybrid window mask
    """
    tmp_paths = []
    try:
        # ── Phase 1a-b: Download, decode, resize ─────────────────────────────
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

        # Sort raw frames dark→bright by mean luminance
        means_raw = [cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).mean() for img in raw_images]
        order_raw = list(np.argsort(means_raw))
        raw_images = [raw_images[i] for i in order_raw]
        means_raw  = [means_raw[i]  for i in order_raw]
        print(f"Raw frame means dark→bright: {[round(m, 1) for m in means_raw]}")

        single_shot = len(raw_images) == 1
        if single_shot:
            raw_images = synthesize_brackets(raw_images[0])
            means_raw  = [cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).mean() for img in raw_images]

        raw_dark_frame   = raw_images[0]
        raw_bright_frame = raw_images[-1]

        # ── Phase 1c: Build HYBRID window mask from RAW dark frame ────────────
        # CRITICAL: must be done before normalisation — after normalisation the
        # exposure differential is destroyed and window detection fails.
        # LSD geometric layer catches windows that aren't fully blown out
        # (overcast days, curtained windows, windows at angle) where the
        # pure brightness threshold misses them.
        print("Building hybrid connected-components + photometric window mask...")
        win_mask  = build_window_mask_from_raw(raw_dark_frame, raw_bright_frame, sigma=18.0)
        win_mask3 = win_mask[:, :, np.newaxis]
        interior3 = 1.0 - win_mask3

        # ── Phase 1d: Select best window-detail frame ─────────────────────────
        # For window pull we ALWAYS want the darkest raw frame — it has the most
        # exterior detail. Edge-score selection can pick a mid/bright frame when
        # window mullions are sharp there, which is the wrong source for exterior.
        print("Using darkest raw frame as window source (best exterior detail)...")
        best_window_frame = raw_images[0]  # darkest = most exterior detail
        print(f"  Dark frame mean: {means_raw[0]:.1f}")

        # ── Phase 2: Denoise raw frames ───────────────────────────────────────
        # Bilateral filter: smooths flat walls, preserves window frame/mullion edges
        print("Denoising (bilateral)...")
        raw_images = [cv2.bilateralFilter(img, d=9, sigmaColor=75, sigmaSpace=75)
                      for img in raw_images]
        best_window_frame = cv2.bilateralFilter(best_window_frame, d=9,
                            sigmaColor=75, sigmaSpace=75)
        gc.collect()

        # ── Phase 2e-f: Exposure normalisation ────────────────────────────────
        print("Normalising exposures...")
        norm_images = []
        for i, img in enumerate(raw_images):
            if i == 0:      # darkest: lower target preserves window contrast
                target = 75.0
            elif i == len(raw_images) - 1:  # brightest: open the interior
                target = 120.0
            else:           # mid frames
                target = 105.0
            norm_images.append(normalise_to_target(img, target))

        norm_bright_frame = norm_images[-1]
        norm_means = [cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).mean() for img in norm_images]
        print(f"Normalised means: {[round(m, 1) for m in norm_means]}")
        gc.collect()

        # ── Phase 2g: AlignMTB ─────────────────────────────────────────────────
        if len(norm_images) > 1:
            print("Aligning...")
            align = cv2.createAlignMTB(max_bits=6, exclude_range=4, cut=True)
            align.process(norm_images, norm_images)
            gc.collect()

        # ── Phase 2h: Mertens fusion (interior ambient base) ──────────────────
        if not single_shot and len(norm_images) > 1:
            norm_images = deghost(norm_images, ref_idx=len(norm_images) // 2)
            gc.collect()

        print("Mertens fusion (interior base)...")
        fused = cv2.createMergeMertens(
            contrast_weight=1.0,
            saturation_weight=0.8,
            exposure_weight=0.0,
        ).process(norm_images)
        mertens_base = np.clip(fused * 255, 0, 255).astype(np.uint8)
        del fused; gc.collect()

        # Post-fusion: very mild CLAHE — just enough to lift dark corners without halos
        print("Applying CLAHE to Mertens base...")
        lab = cv2.cvtColor(mertens_base, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=0.8, tileGridSize=(16, 16))
        l = clahe.apply(l)
        mertens_base = cv2.cvtColor(cv2.merge((l, a, b)), cv2.COLOR_LAB2BGR)

        # ── Phase 3i-j: Start from Mertens, boost dark interior zones ─────────
        print("Interior composite...")
        mertens_f = mertens_base.astype(np.float32) / 255.0
        bright_f  = norm_bright_frame.astype(np.float32) / 255.0
        lum_m     = 0.299 * mertens_f[:, :, 2] + 0.587 * mertens_f[:, :, 1] + 0.114 * mertens_f[:, :, 0]

        dark_interior_mask = np.clip(1.0 - lum_m / 0.45, 0, 1) ** 1.5
        dark_interior_mask = dark_interior_mask * interior3[:, :, 0]
        dark_interior_mask = cv2.GaussianBlur(dark_interior_mask.astype(np.float32), (0, 0), 10, 10)
        dark_interior_mask = np.clip(dark_interior_mask, 0, 0.70)[:, :, np.newaxis]

        interior_f = mertens_f + dark_interior_mask * (bright_f - mertens_f)
        interior_f = np.clip(interior_f, 0, 1)

        # ── Phase 3k: Window pull — mask from interior result, source from dark frame ─
        print("Window composite...")
        # Lift dark frame enough to show bright exterior (trees, sky, street visible)
        # Target ~80 gives a naturally lit exterior without blowing it out
        win_frame_normed = normalise_to_target(best_window_frame, target_mean=80.0)
        win_frame_f      = win_frame_normed.astype(np.float32) / 255.0

        # PRIMARY MASK: detect blown windows on the INTERIOR composite result
        # (mirrors the reference algorithm: threshold 240/255 = 0.94 on bright image)
        lum_interior = (0.299 * interior_f[:, :, 2] +
                        0.587 * interior_f[:, :, 1] +
                        0.114 * interior_f[:, :, 0])

        # Threshold at 0.85 — catch more blown window zones to reveal exterior detail
        blown_hard = (lum_interior > 0.85).astype(np.uint8) * 255
        # Small close to fill mullion gaps only — don't bleed onto curtains/couch
        close_k    = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
        blown_hard = cv2.morphologyEx(blown_hard, cv2.MORPH_CLOSE, close_k)
        # More aggressive dilation — pulls exterior detail further into frames
        dilate_k   = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
        blown_hard = cv2.dilate(blown_hard, dilate_k)
        # Smooth edges
        blown_mask = cv2.GaussianBlur(blown_hard.astype(np.float32) / 255.0,
                                      (21, 21), 0)
        blown_mask = np.clip(blown_mask, 0, 1)[:, :, np.newaxis]

        # SECONDARY MASK: only use pre-computed dark-frame mask at low weight
        # — prevents it from darkening curtains/frames outside the true glass zone
        combined_mask = np.clip(blown_mask + win_mask3 * 0.25, 0, 1)

        # Blend: blown/window zones → exterior dark frame; room → interior
        composited_f = interior_f * (1.0 - combined_mask) + win_frame_f * combined_mask
        composited_f = np.clip(composited_f, 0, 1)

        print(f"  Window pull: {blown_mask.mean()*100:.1f}% of image replaced")

        composited = np.clip(composited_f * 255, 0, 255).astype(np.uint8)
        del mertens_f, bright_f, win_frame_f, composited_f, win_mask3, mertens_base
        gc.collect()

        mean_out = cv2.cvtColor(composited, cv2.COLOR_BGR2GRAY).mean()
        print(f"Merge complete. Output mean brightness: {mean_out:.1f}/255")
        return composited

    finally:
        for p in tmp_paths:
            try: os.unlink(p)
            except Exception: pass


# ---------------------------------------------------------------------------
# POST-PROCESSING FINISH
# ---------------------------------------------------------------------------

def apply_autohdr_finish(img_bgr: np.ndarray) -> np.ndarray:
    """
    Finish pass — tuned to match target: bright neutral beige tones, strong
    shadow fill on dark sides, controlled window highlights, no yellow-green cast.

    Parameters calibrated by AI diff analysis against target reference:
      gamma=0.42, highlight_start=0.72, fill_cutoff=0.38, fill_strength=0.32
      r_mult=1.04, g_mult=1.01, b_mult=0.93, vibrance=18, sharpen=0.55/1.2r

    Steps:
      1. Gamma lift (aggressive — interior was too dark overall)
      2. Highlight rolloff (starts at 0.72 — earlier than before to control windows)
      3. Large-radius USM on L-channel: 3D depth pop, no grain
      4. Wall/ceiling zone: desaturate + cast removal (removes yellow-green cast)
      5. Shadow fill — stronger cutoff (0.38) and fill (0.32) for dark corners
      6. Colour grade: R+4%, G+1%, B-7% — eliminates yellow-green cast globally
      7. Vibrance (18 units)
      8. Sharpening (0.55 amount, 1.2px radius)
    """
    img = img_bgr.astype(np.float32) / 255.0

    # ── Window protect mask ───────────────────────────────────────────────────
    lum_raw = 0.299 * img[:, :, 2] + 0.587 * img[:, :, 1] + 0.114 * img[:, :, 0]
    b_chan, g_chan, r_chan = img[:, :, 0], img[:, :, 1], img[:, :, 2]
    # Lower threshold to 0.58 — protect windows more aggressively from gamma lift
    lum_win  = np.clip((lum_raw - 0.58) / (1.0 - 0.58 + 1e-6), 0, 1) ** 1.2
    blue_dom = np.clip((b_chan - np.maximum(r_chan, g_chan) + 0.05) / 0.12, 0, 1)
    blue_dom = blue_dom * (lum_raw > 0.45).astype(np.float32)
    win_raw      = np.clip(lum_win * 0.8 + blue_dom * 0.2, 0, 1)
    win_protect  = cv2.GaussianBlur(win_raw.astype(np.float32), (0, 0), 12, 12)
    win_protect  = np.clip(win_protect, 0, 1)
    win_protect3 = win_protect[:, :, np.newaxis]
    interior3    = 1.0 - win_protect3

    # ── 1. Gamma lift — aggressive to match bright target ────────────────────
    # gamma=0.42 → strong lift, pushing mid-tones from muddy dark to bright beige
    gamma = 0.42
    img_interior = np.clip(np.power(np.clip(img, 0, 1), gamma), 0, 1)
    img = img_interior * interior3 + img * win_protect3
    img = np.clip(img, 0, 1)

    # ── 2. Highlight rolloff (interior only, protect windows) ────────────────
    # Rolls off highlights starting at 0.72 to cap bright walls at 0.88
    # This prevents the white ceiling/walls from blowing while interior is bright
    highlight_start  = 0.72
    highlight_cap    = 0.88
    highlight_str    = 0.65
    lum_h = 0.299 * img[:, :, 2] + 0.587 * img[:, :, 1] + 0.114 * img[:, :, 0]
    hl_mask = np.clip((lum_h - highlight_start) / (1.0 - highlight_start + 1e-6), 0, 1) ** 1.5
    hl_mask = hl_mask * interior3[:, :, 0]
    hl_mask3 = hl_mask[:, :, np.newaxis]
    # Blend toward highlight_cap in blown zones
    target_hl = img * (highlight_cap / (lum_h[:, :, np.newaxis] + 1e-6))
    target_hl = np.clip(target_hl, 0, 1)
    img = img * (1.0 - hl_mask3 * highlight_str) + target_hl * (hl_mask3 * highlight_str)
    img = np.clip(img, 0, 1)

    # ── 3. Large-radius USM on L-channel ─────────────────────────────────────
    img_u8 = (img * 255).astype(np.uint8)
    img_u8 = local_contrast_enhance(img_u8, radius=45.0, amount=0.20)
    img    = img_u8.astype(np.float32) / 255.0

    # ── 4. Wall/ceiling zone — desaturate + cast removal ─────────────────────
    lum2 = 0.299 * img[:, :, 2] + 0.587 * img[:, :, 1] + 0.114 * img[:, :, 0]
    wall_mask = np.clip((lum2 - 0.30) / (0.95 - 0.30), 0, 1)
    wall_mask = wall_mask * (1.0 - win_protect)
    wall_mask = cv2.GaussianBlur(wall_mask.astype(np.float32), (0, 0), 8, 8)
    wall_mask = np.clip(wall_mask, 0, 1)
    wall_mask3 = wall_mask[:, :, np.newaxis]

    # Stronger desaturation on walls to remove yellow-green cast
    img_u8_tmp = (img * 255).astype(np.uint8)
    hsv_tmp = cv2.cvtColor(img_u8_tmp, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv_tmp[:, :, 1] = hsv_tmp[:, :, 1] * (1.0 - wall_mask * 0.55)  # was 0.45
    img = cv2.cvtColor(np.clip(hsv_tmp, 0, 255).astype(np.uint8), cv2.COLOR_HSV2BGR).astype(np.float32) / 255.0

    # Grey-world cast removal (±10% cap — wider to fix yellow-green cast)
    wall_sum = wall_mask.sum() + 1e-6
    mean_r   = (img[:, :, 2] * wall_mask).sum() / wall_sum
    mean_g   = (img[:, :, 1] * wall_mask).sum() / wall_sum
    mean_b   = (img[:, :, 0] * wall_mask).sum() / wall_sum
    mean_all = (mean_r + mean_g + mean_b) / 3.0 + 1e-6
    for ch, mean_ch in [(2, mean_r), (1, mean_g), (0, mean_b)]:
        cor = np.clip(mean_all / (mean_ch + 1e-6), 0.90, 1.10)  # was 0.92/1.08
        img[:, :, ch] = np.clip(img[:, :, ch] * (1.0 + (cor - 1.0) * wall_mask * 0.35), 0, 1)  # was 0.25

    # Brightness push on walls
    img = img + wall_mask3 * 0.08 * (1.0 - img)
    img = np.clip(img, 0, 1)

    # ── 5. Shadow fill — stronger to lift dark left-side corners ─────────────
    # fill_cutoff=0.38 (was 0.28), fill_strength=0.32 (was 0.40 but capped lower)
    lum3       = 0.299 * img[:, :, 2] + 0.587 * img[:, :, 1] + 0.114 * img[:, :, 0]
    fill_mask  = np.clip(1.0 - lum3 / 0.38, 0, 1) ** 1.5   # wider cutoff catches more shadow
    fill_mask3 = fill_mask[:, :, np.newaxis]
    img = img + fill_mask3 * 0.32 * (1.0 - img) * interior3  # controlled strength
    img = np.clip(img, 0, 1)

    # ── 6. Colour grade — R+4%, G+1%, B-7% to remove yellow-green, add warmth ─
    img[:, :, 2] = np.clip(img[:, :, 2] * 1.04, 0, 1)  # R up
    img[:, :, 1] = np.clip(img[:, :, 1] * 1.01, 0, 1)  # G neutral
    img[:, :, 0] = np.clip(img[:, :, 0] * 0.93, 0, 1)  # B down — kills yellow-green cast

    # ── 7. Vibrance (18 units — moderate, interior surfaces only) ─────────────
    img_u8 = (img * 255).astype(np.uint8)
    hsv    = cv2.cvtColor(img_u8, cv2.COLOR_BGR2HSV).astype(np.float32)
    sat_n  = hsv[:, :, 1] / 255.0
    vib_zone  = np.clip(1.0 - wall_mask, 0, 1)
    vib_boost = 18.0 * (1.0 - sat_n) ** 1.5 * vib_zone  # was 12
    hsv[:, :, 1] = np.clip(hsv[:, :, 1] + vib_boost, 0, 255)
    img = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR).astype(np.float32) / 255.0

    # ── 8. Sharpening (0.55 amount, 1.2px radius) ────────────────────────────
    img_u8    = (img * 255).astype(np.uint8)
    pil_img   = Image.fromarray(cv2.cvtColor(img_u8, cv2.COLOR_BGR2RGB))
    blurred   = pil_img.filter(ImageFilter.GaussianBlur(radius=1.2))  # was 0.8
    sharpened = np.clip(
        np.array(pil_img).astype(np.float32) + 0.55 * (   # was 0.45
            np.array(pil_img).astype(np.float32) - np.array(blurred).astype(np.float32)
        ), 0, 255
    ).astype(np.uint8)
    result_bgr = cv2.cvtColor(sharpened, cv2.COLOR_RGB2BGR)

    mean_final = cv2.cvtColor(result_bgr, cv2.COLOR_BGR2GRAY).mean()
    print(f"Finish applied. Final mean: {mean_final:.1f}/255")
    return result_bgr


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


@app.get("/health")
def health():
    return {"status": "ok"}
