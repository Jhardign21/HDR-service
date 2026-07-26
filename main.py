"""
HDR Merge Service — Multi-stage real estate photo processing pipeline.

Stage A: File ingestion, EXIF grouping, SIFT alignment
Stage B: Enfuse exposure fusion + window pull (dark bracket window detail)
Stage C: Vertical correction (Hough transform perspective warp)
Stage D: White point protection + finish (clarity, sharpen, dither)
"""
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import numpy as np
import cv2
import requests
import tempfile
import os
from PIL import Image, ExifTags
import io
import base64
import gc
import traceback
import subprocess
import shutil
import datetime

app = FastAPI(title="HDR Merge Service")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

MAX_DIM = 3000


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


def smart_resize(img: np.ndarray, max_dim: int = MAX_DIM) -> np.ndarray:
    h, w = img.shape[:2]
    longest = max(h, w)
    if longest <= max_dim:
        return img
    scale = max_dim / longest
    new_w, new_h = int(w * scale), int(h * scale)
    return cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)


def synthesize_brackets(img: np.ndarray) -> List[np.ndarray]:
    f = img.astype(np.float32) / 255.0
    dark = np.clip(np.power(f, 1.8), 0, 1)
    bright = np.clip(np.power(f, 0.45), 0, 1)
    print("Synthesized virtual brackets.")
    return [(dark * 255).astype(np.uint8), img, (bright * 255).astype(np.uint8)]


# ---------------------------------------------------------------------------
# Enfuse-based exposure fusion
# ---------------------------------------------------------------------------

def enfuse_available() -> bool:
    return shutil.which("enfuse") is not None


def enfuse_merge(images: List[np.ndarray]) -> Optional[np.ndarray]:
    if not enfuse_available():
        print("  Enfuse binary not found — falling back to Mertens")
        return None

    h, w = images[0].shape[:2]
    tmp_dir = tempfile.mkdtemp(prefix="enfuse_", dir="/tmp")
    in_paths = []
    out_path = os.path.join(tmp_dir, "fused.tif")
    try:
        for i, img in enumerate(images):
            p = os.path.join(tmp_dir, f"frame_{i:02d}.tif")
            cv2.imwrite(p, img)
            in_paths.append(p)

        cmd = [
            "enfuse", "--depth=8",
            "--exposure-weight=1.0", "--saturation-weight=0.2", "--contrast-weight=0.0",
            "--exposure-optimum=0.5", "--exposure-width=0.2", "--no-ciecam",
            "--output=" + out_path,
        ] + in_paths

        print(f"  Running enfuse on {len(in_paths)} frames...")
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
        if proc.returncode != 0:
            print(f"  Enfuse failed (rc={proc.returncode}): {proc.stderr[:500]}")
            return None

        result = cv2.imread(out_path, cv2.IMREAD_COLOR)
        if result is None:
            return None
        result = cv2.resize(result, (w, h), interpolation=cv2.INTER_AREA)
        print(f"  Enfuse OK. Output mean: {cv2.cvtColor(result, cv2.COLOR_BGR2GRAY).mean():.1f}")
        return result
    except Exception as e:
        print(f"  Enfuse error: {e}")
        return None
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


# ---------------------------------------------------------------------------
# STAGE A: Bracket alignment (SIFT) + EXIF grouping
# ---------------------------------------------------------------------------

def align_brackets(images, ref_idx=1):
    """Align all bracket frames to the reference using SIFT + RANSAC."""
    if len(images) <= 1:
        return images
    ref_idx = max(0, min(ref_idx, len(images) - 1))
    ref = images[ref_idx]
    ref_gray = cv2.cvtColor(ref, cv2.COLOR_BGR2GRAY)
    sift = cv2.SIFT_create()
    kp_ref, desc_ref = sift.detectAndCompute(ref_gray, None)
    if desc_ref is None or len(kp_ref) < 10:
        print("  Alignment: Not enough reference keypoints, skipping")
        return images

    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)

    aligned = []
    for i, img in enumerate(images):
        if i == ref_idx:
            aligned.append(img)
            continue
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        kp, desc = sift.detectAndCompute(gray, None)
        if desc is None or len(kp) < 10:
            aligned.append(img)
            continue
        matches = flann.knnMatch(desc, desc_ref, k=2)
        good = [m for m, n in matches if m.distance < 0.7 * n.distance]
        if len(good) < 10:
            aligned.append(img)
            continue
        src_pts = np.float32([kp[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp_ref[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
        M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
        if M is None:
            aligned.append(img)
            continue
        h, w = ref.shape[:2]
        warped = cv2.warpPerspective(img, M, (w, h), flags=cv2.INTER_LINEAR,
                                     borderMode=cv2.BORDER_REFLECT)
        aligned.append(warped)
        inliers = int(mask.sum()) if mask is not None else len(good)
        print(f"  Alignment: Frame {i} aligned ({inliers} inliers)")
    return aligned


def extract_exif(image_path):
    """Extract EXIF metadata for bracket grouping."""
    try:
        img = Image.open(image_path)
        exif_data = img._getexif()
        if not exif_data:
            return None
        tags = {ExifTags.TAGS.get(k, k): v for k, v in exif_data.items()}
        timestamp = tags.get('DateTimeOriginal', tags.get('DateTime', ''))
        focal_length = tags.get('FocalLength', 0)
        exposure_time = tags.get('ExposureTime', 0)
        iso = tags.get('ISOSpeedRatings', tags.get('ISOSpeedRatings', 0))
        return {
            'timestamp': str(timestamp) if timestamp else '',
            'focal_length': float(focal_length) if focal_length else 0,
            'exposure_time': float(exposure_time) if exposure_time else 0,
            'iso': int(iso) if iso else 0,
        }
    except Exception as e:
        print(f"  EXIF extraction failed for {image_path}: {e}")
        return None


def _parse_timestamp(ts_str):
    if not ts_str:
        return None
    for fmt in ('%Y:%m:%d %H:%M:%S', '%Y-%m-%d %H:%M:%S'):
        try:
            return datetime.datetime.strptime(ts_str, fmt)
        except (ValueError, TypeError):
            continue
    return None


def group_brackets_by_exif(file_info_list, time_threshold_seconds=10):
    """Group files into bracket clusters by EXIF metadata."""
    if not file_info_list:
        return []
    def get_ts(info):
        exif = info.get('exif') or {}
        return _parse_timestamp(exif.get('timestamp', '')) or datetime.datetime.min
    sorted_files = sorted(file_info_list, key=get_ts)
    groups = []
    current_group = [sorted_files[0]]
    for i in range(1, len(sorted_files)):
        prev, curr = sorted_files[i - 1], sorted_files[i]
        prev_ts, curr_ts = get_ts(prev), get_ts(curr)
        time_diff = (curr_ts - prev_ts).total_seconds()
        prev_fl = (prev.get('exif') or {}).get('focal_length', 0)
        curr_fl = (curr.get('exif') or {}).get('focal_length', 0)
        same_fl = (prev_fl == curr_fl) or curr_fl == 0 or prev_fl == 0
        if time_diff <= time_threshold_seconds and same_fl:
            current_group.append(curr)
        else:
            groups.append(current_group)
            current_group = [curr]
    groups.append(current_group)
    result = []
    for i, group in enumerate(groups):
        result.append({
            'bracket_id': i, 'name': f"Bracket_{i + 1}",
            'files': [f['url'] for f in group], 'bracket_size': len(group),
            'exif_data': [f.get('exif') for f in group],
        })
    return result


# ---------------------------------------------------------------------------
# STAGE B: Window segmentation + targeted exposure fusion
# ---------------------------------------------------------------------------

def differential_window_mask(ambient_bgr, darkest_bgr):
    """Detect windows by pixel drop between bright ambient and darkest bracket."""
    h, w = ambient_bgr.shape[:2]
    gray_ambient = cv2.cvtColor(ambient_bgr, cv2.COLOR_BGR2GRAY).astype(np.int16)
    gray_darkest = cv2.cvtColor(darkest_bgr, cv2.COLOR_BGR2GRAY).astype(np.int16)
    pixel_delta = np.clip(gray_ambient - gray_darkest, 0, 255).astype(np.uint8)
    _, window_zones = cv2.threshold(pixel_delta, 120, 255, cv2.THRESH_BINARY)
    kernel_size = max(int(max(h, w) * 0.015), 3)
    if kernel_size % 2 == 0:
        kernel_size += 1
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
    closed_mask = cv2.morphologyEx(window_zones, cv2.MORPH_CLOSE, kernel)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(closed_mask, connectivity=8)
    geom_mask = np.zeros((h, w), dtype=np.uint8)
    min_area = int((h * w) * 0.005)
    max_area = int((h * w) * 0.60)
    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        if min_area <= area <= max_area:
            geom_mask[labels == i] = 255
    feathered = cv2.GaussianBlur(geom_mask.astype(np.float32), (21, 21), 0)
    return np.clip(feathered / 255.0, 0, 1)


def build_window_mask(dark_raw, bright_raw, sigma=12.0):
    """Hybrid geometric + photometric window mask."""
    h, w = dark_raw.shape[:2]
    f = dark_raw.astype(np.float32) / 255.0
    lum = 0.299 * f[:, :, 2] + 0.587 * f[:, :, 1] + 0.114 * f[:, :, 0]
    hard_thresh = (lum > 0.55).astype(np.uint8) * 255
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (35, 35))
    hard_closed = cv2.morphologyEx(hard_thresh, cv2.MORPH_CLOSE, kernel)
    dilate_k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (25, 25))
    hard_closed = cv2.dilate(hard_closed, dilate_k)
    contours, _ = cv2.findContours(hard_closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    photo_mask = np.zeros((h, w), dtype=np.float32)
    min_window_area = (h * w) * 0.002
    for cnt in contours:
        if cv2.contourArea(cnt) >= min_window_area:
            cv2.drawContours(photo_mask, [cnt], -1, 1.0, thickness=cv2.FILLED)
    spill = np.clip((lum - 0.60) / (1.0 - 0.60 + 1e-6), 0, 1) ** 1.2
    photo_combined = np.clip(photo_mask + spill * 0.7, 0, 1)
    delta_mask = differential_window_mask(bright_raw, dark_raw)
    combined = np.clip(photo_combined + delta_mask, 0, 1)
    combined = cv2.GaussianBlur(combined.astype(np.float32), (0, 0), sigmaX=sigma, sigmaY=sigma)
    combined = np.clip(combined, 0, 1)
    print(f"  Window mask: {combined.mean()*100:.1f}% of image covered")
    return combined


def select_best_window_frame(images_raw, win_mask):
    """Select the bracket with the best window detail (highest edge density)."""
    best_idx, best_score = 0, -1.0
    mask_bool = win_mask > 0.3
    for i, img in enumerate(images_raw):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)
        lap = np.abs(cv2.Laplacian(gray, cv2.CV_32F))
        score = float(lap[mask_bool].mean()) if mask_bool.any() else 0.0
        print(f"  Window frame {i} detail score: {score:.2f}")
        if score > best_score:
            best_score = score
            best_idx = i
    print(f"  Selected frame {best_idx} as best window source")
    return best_idx


def window_pull_fusion(ambient_base, window_source, win_mask, feather_sigma=0.5):
    """Composite dark bracket window detail onto bright ambient base."""
    feathered_mask = cv2.GaussianBlur(win_mask.astype(np.float32), (5, 5), feather_sigma)
    mask3 = feathered_mask[:, :, np.newaxis]
    result = ambient_base.astype(np.float32) * (1 - mask3) + window_source.astype(np.float32) * mask3
    return np.clip(result, 0, 255).astype(np.uint8)


def detect_window_mask_single(img_bgr):
    """Detect blown window panes from a single photo (for sky replacement)."""
    h, w = img_bgr.shape[:2]
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    l = lab[:, :, 0].astype(np.float32)
    blown = (l > 235).astype(np.uint8) * 255
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 25))
    closed = cv2.morphologyEx(blown, cv2.MORPH_CLOSE, kernel)
    dilate_k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    closed = cv2.dilate(closed, dilate_k)
    num, labels, stats, _ = cv2.connectedComponentsWithStats(closed, 8)
    mask = np.zeros((h, w), dtype=np.uint8)
    min_area = int((h * w) * 0.003)
    for i in range(1, num):
        if stats[i, cv2.CC_STAT_AREA] >= min_area:
            mask[labels == i] = 255
    feathered = cv2.GaussianBlur(mask.astype(np.float32), (21, 21), 0)
    return np.clip(feathered / 255.0, 0, 1)


def generate_sky(h, w):
    """Procedural natural blue sky: vertical gradient + soft clouds. BGR float32."""
    top = np.array([170, 130, 70], dtype=np.float32)
    horz = np.array([225, 205, 185], dtype=np.float32)
    grad = np.zeros((h, w, 3), dtype=np.float32)
    ys = np.linspace(0, 1, h)[:, None] ** 0.7
    grad[:] = top * (1 - ys[:, :, None]) + horz * ys[:, :, None]
    noise = np.random.rand(h, w).astype(np.float32)
    cloud = cv2.GaussianBlur(noise, (0, 0), sigmaX=w / 6)
    cloud = (cloud - cloud.min()) / (cloud.max() - cloud.min() + 1e-6)
    cloud = np.clip((cloud - 0.58) / 0.25, 0, 1)[:, :, None]
    white = np.array([245, 245, 245], dtype=np.float32)
    grad = grad * (1 - cloud * 0.45) + white * (cloud * 0.45)
    return np.clip(grad, 0, 255)


# ---------------------------------------------------------------------------
# STAGE C: Geometry — vertical correction
# ---------------------------------------------------------------------------

def correct_verticals(img, max_correction_deg=3.0):
    """Detect vertical lines via Hough Transform and correct perspective tilt."""
    try:
        h, w = img.shape[:2]
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150, apertureSize=3)
        lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=100,
                                minLineLength=int(h * 0.15), maxLineGap=15)
        if lines is None:
            return img
        angles = []
        for line in lines:
            x1, y1, x2, y2 = line[0]
            dx, dy = abs(x1 - x2), abs(y2 - y1)
            if dx < 20 and dy > 50:
                angle = np.arctan2(y2 - y1, x2 - x1) * 180.0 / np.pi
                tilt = angle - 90.0 if angle > 45 else angle + 90.0
                if abs(tilt) < max_correction_deg:
                    angles.append(tilt)
        if len(angles) < 5:
            return img
        median_tilt = float(np.median(angles))
        if abs(median_tilt) < 0.1:
            return img
        M = cv2.getRotationMatrix2D((w // 2, h // 2), median_tilt, 1.0)
        corrected = cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_CUBIC,
                                   borderMode=cv2.BORDER_REPLICATE)
        print(f"  Vertical correction: {median_tilt:.2f}° tilt corrected ({len(angles)} lines)")
        return corrected
    except Exception as e:
        print(f"  Vertical correction skipped: {e}")
        return img


# ---------------------------------------------------------------------------
# STAGE D: White point protection + finish
# ---------------------------------------------------------------------------

def protect_white_points(img_bgr):
    """Lock white points on smooth surfaces to prevent the 'grey wall' effect."""
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    l = lab[:, :, 0]
    bright = np.clip((l - 195.0) / 30.0, 0, 1)
    local_mean = cv2.blur(l, (15, 15))
    local_sq_mean = cv2.blur(l ** 2, (15, 15))
    local_std = np.sqrt(np.maximum(local_sq_mean - local_mean ** 2, 0))
    smooth = np.clip(1.0 - local_std / 15.0, 0, 1)
    white_mask = bright * smooth
    white_mask = cv2.GaussianBlur(white_mask, (0, 0), sigmaX=10, sigmaY=10)
    l_protected = l + white_mask * (250.0 - l) * 0.5
    lab[:, :, 0] = np.clip(l_protected, 0, 255)
    pull = 0.80
    lab[:, :, 1] = lab[:, :, 1] * (1.0 - white_mask * pull) + 128.0 * (white_mask * pull)
    lab[:, :, 2] = lab[:, :, 2] * (1.0 - white_mask * pull) + 128.0 * (white_mask * pull)
    return cv2.cvtColor(np.clip(lab, 0, 255).astype(np.uint8), cv2.COLOR_LAB2BGR)


def apply_finish(img_bgr):
    """Clarity + light sharpen + dither. No CLAHE (causes tile banding)."""
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    l = lab[:, :, 0] / 255.0
    blurred = cv2.GaussianBlur(l, (0, 0), sigmaX=25, sigmaY=25)
    l = np.clip(l + 0.25 * (l - blurred), 0, 1)
    lab[:, :, 0] = l * 255.0
    img_f = cv2.cvtColor(np.clip(lab, 0, 255).astype(np.uint8),
                         cv2.COLOR_LAB2BGR).astype(np.float32)
    blur_s = cv2.GaussianBlur(img_f, (0, 0), sigmaX=1.5, sigmaY=1.5)
    img_f = np.clip(img_f + 0.3 * (img_f - blur_s), 0, 255)
    noise = np.random.randint(-1, 2, img_f.shape[:2], dtype=np.int16)
    img_f = np.clip(img_f + noise[:, :, None], 0, 255)
    result = img_f.astype(np.uint8)
    print(f"  Finish applied. Final mean: {cv2.cvtColor(result, cv2.COLOR_BGR2GRAY).mean():.1f}/255")
    return result


# ---------------------------------------------------------------------------
# BRACKET MERGE PIPELINE
# ---------------------------------------------------------------------------

def bracket_merge(file_urls: List[str]) -> np.ndarray:
    tmp_paths = []
    try:
        print(f"Downloading {len(file_urls)} frames...")
        for url in file_urls:
            ext = url.split("?")[0].rsplit(".", 1)[-1]
            ext = f".{ext.lower()}" if ext else ".jpg"
            tmp_paths.append(download_file(url, ext))

        raw_images = []
        for p in tmp_paths:
            img = load_image_bgr(p)
            img = smart_resize(img)
            raw_images.append(img)
        print(f"Loaded {len(raw_images)} frames at {raw_images[0].shape[1]}×{raw_images[0].shape[0]}")

        if len(raw_images) == 1:
            raw_images = synthesize_brackets(raw_images[0])

        if len(raw_images) > 1:
            print("Aligning brackets (SIFT)...")
            ref_idx = len(raw_images) // 2
            raw_images = align_brackets(raw_images, ref_idx=ref_idx)
            gc.collect()

        print("Denoising (bilateral)...")
        raw_images = [cv2.bilateralFilter(img, d=5, sigmaColor=45, sigmaSpace=45) for img in raw_images]
        gc.collect()

        print("Running Enfuse...")
        merged = enfuse_merge(raw_images)
        if merged is None:
            print("Mertens fallback fusion...")
            fused = cv2.createMergeMertens(contrast_weight=1.5, saturation_weight=1.2,
                                           exposure_weight=0.0).process(raw_images)
            merged = np.clip(fused * 255, 0, 255).astype(np.uint8)
            del fused; gc.collect()

        print(f"Merge complete. Mean brightness: {cv2.cvtColor(merged, cv2.COLOR_BGR2GRAY).mean():.1f}/255")

        if len(raw_images) >= 2:
            print("Running window pull...")
            win_mask = build_window_mask(raw_images[0], raw_images[-1])
            if win_mask.max() > 0.05:
                best_idx = select_best_window_frame(raw_images, win_mask)
                merged = window_pull_fusion(merged, raw_images[best_idx], win_mask)
                print("Window pull applied")
            else:
                print("No significant windows detected, skipping window pull")

        print("Correcting verticals...")
        merged = correct_verticals(merged)

        print("Protecting white points...")
        merged = protect_white_points(merged)

        print("Applying finish...")
        merged = apply_finish(merged)
        return merged
    finally:
        for p in tmp_paths:
            try: os.unlink(p)
            except Exception: pass


# ---------------------------------------------------------------------------
# SINGLE-PHOTO ENHANCEMENT
# ---------------------------------------------------------------------------

def enhance_single(file_url: str, replace_sky: bool = False) -> np.ndarray:
    """Lightroom-style enhancement — SINGLE float32 LAB pass. No banding."""
    ext = file_url.split("?")[0].rsplit(".", 1)[-1]
    ext = f".{ext.lower()}" if ext else ".jpg"
    tmp = download_file(file_url, ext)
    try:
        img = load_image_bgr(tmp)
        img = smart_resize(img)
        h, w = img.shape[:2]
        print(f"Loaded single photo at {w}×{h}")

        img = cv2.bilateralFilter(img, d=4, sigmaColor=30, sigmaSpace=30)
        img = correct_verticals(img)

        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB).astype(np.float32)
        l, a, b = lab[:, :, 0].copy(), lab[:, :, 1].copy(), lab[:, :, 2].copy()

        # Exposure lift
        l = np.clip(np.power(l / 255.0, 0.48) * 255.0, 0, 255)
        # Lower highlights
        hl_mask = np.clip((l - 215.0) / 40.0, 0, 1) ** 1.5
        hl_mask = cv2.GaussianBlur(hl_mask, (0, 0), sigmaX=20, sigmaY=20)
        l = np.clip(l - hl_mask * 45.0, 0, 255)
        # Lift shadows
        sh_mask = np.clip((80.0 - l) / 80.0, 0, 1) ** 1.2
        sh_mask = cv2.GaussianBlur(sh_mask, (0, 0), sigmaX=20, sigmaY=20)
        l = np.clip(l + sh_mask * 65.0, 0, 255)
        # Whites & blacks
        white_mask = np.clip((l - 200.0) / 55.0, 0, 1)
        black_mask = np.clip((40.0 - l) / 40.0, 0, 1)
        l = np.clip(l + white_mask * 18.0 + black_mask * 15.0, 0, 255)
        # White point protection (prevent grey wall)
        bright_smooth = np.clip((l - 195.0) / 30.0, 0, 1)
        local_mean = cv2.blur(l, (15, 15))
        local_sq = cv2.blur(l ** 2, (15, 15))
        local_std = np.sqrt(np.maximum(local_sq - local_mean ** 2, 0))
        smooth = np.clip(1.0 - local_std / 15.0, 0, 1)
        wp_mask = bright_smooth * smooth
        wp_mask = cv2.GaussianBlur(wp_mask, (0, 0), sigmaX=10, sigmaY=10)
        l = np.clip(l + wp_mask * (250.0 - l) * 0.5, 0, 255)
        # White balance
        bright_w = np.clip((l - 160.0) / 40.0, 0, 1)
        bright_w = cv2.GaussianBlur(bright_w, (0, 0), sigmaX=15, sigmaY=15)
        wb_pull = 0.85
        a = a * (1.0 - bright_w * wb_pull) + 128.0 * (bright_w * wb_pull)
        b = b * (1.0 - bright_w * wb_pull) + 128.0 * (bright_w * wb_pull)
        # Saturation (mid/dark zones only)
        sat_mask = 1.0 - bright_w
        a = np.clip(a + (a - 128.0) * 0.30 * sat_mask, 0, 255)
        b = np.clip(b + (b - 128.0) * 0.30 * sat_mask, 0, 255)
        # Clarity
        l_norm = l / 255.0
        l_blur = cv2.GaussianBlur(l_norm, (0, 0), sigmaX=25, sigmaY=25)
        l_norm = np.clip(l_norm + 0.25 * (l_norm - l_blur), 0, 1)
        l = l_norm * 255.0

        lab[:, :, 0], lab[:, :, 1], lab[:, :, 2] = l, a, b
        img = cv2.cvtColor(np.clip(lab, 0, 255).astype(np.uint8), cv2.COLOR_LAB2BGR)
        print(f"LAB pass done. Mean: {cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).mean():.1f}")

        if replace_sky:
            win_mask = detect_window_mask_single(img)
            if win_mask.max() > 0.01:
                sky = generate_sky(h, w)
                mask3 = win_mask[:, :, None]
                img = np.clip(img.astype(np.float32) * (1 - mask3) + sky * mask3, 0, 255).astype(np.uint8)
                print("Sky replaced in window zones")

        img_f = img.astype(np.float32)
        blur_s = cv2.GaussianBlur(img_f, (0, 0), sigmaX=1.5, sigmaY=1.5)
        img_f = np.clip(img_f + 0.3 * (img_f - blur_s), 0, 255)
        noise = np.random.randint(-1, 2, img_f.shape[:2], dtype=np.int16)
        img_f = np.clip(img_f + noise[:, :, None], 0, 255)
        result = img_f.astype(np.uint8)
        print(f"Enhancement complete. Final: {result.shape[1]}×{result.shape[0]}")
        return result
    finally:
        try: os.unlink(tmp)
        except Exception: pass


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

class MergeRequest(BaseModel):
    file_urls: List[str]
    bracket_name: str = "bracket"
    replace_sky: bool = False


@app.post("/merge")
async def merge_hdr(req: MergeRequest):
    if not req.file_urls:
        raise HTTPException(400, "No file URLs provided")
    if len(req.file_urls) not in (1, 3, 5):
        raise HTTPException(400, f"Expected 1, 3, or 5 files, got {len(req.file_urls)}")
    try:
        print(f"Starting HDR merge for '{req.bracket_name}' ({len(req.file_urls)} frames)...")
        result = bracket_merge(req.file_urls)
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
        print(f"ERROR in /merge: {tb}")
        raise HTTPException(500, detail=f"{str(e)}\n\nTraceback:\n{tb}")


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


class GroupRequest(BaseModel):
    file_urls: List[str]


@app.post("/group")
async def group_files(req: GroupRequest):
    if not req.file_urls:
        raise HTTPException(400, "No file URLs provided")
    try:
        print(f"Grouping {len(req.file_urls)} files by EXIF metadata...")
        tmp_paths = []
        file_info_list = []
        try:
            for url in req.file_urls:
                ext = url.split("?")[0].rsplit(".", 1)[-1]
                ext = f".{ext.lower()}" if ext else ".jpg"
                path = download_file(url, ext)
                tmp_paths.append(path)
                exif = extract_exif(path)
                file_info_list.append({'url': url, 'path': path, 'exif': exif})
            groups = group_brackets_by_exif(file_info_list)
            print(f"Found {len(groups)} bracket groups")
            return {"success": True, "groups": groups}
        finally:
            for p in tmp_paths:
                try: os.unlink(p)
                except Exception: pass
    except Exception as e:
        tb = traceback.format_exc()
        print(f"ERROR in /group: {tb}")
        raise HTTPException(500, detail=f"{str(e)}\n\nTraceback:\n{tb}")


@app.get("/health")
def health():
    return {"status": "ok"}
