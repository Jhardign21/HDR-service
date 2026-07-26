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
from PIL import Image
import io
import base64
import gc
import traceback
import subprocess
import shutil

from pipeline.alignment import align_brackets
from pipeline.window_pull import (
    build_window_mask, select_best_window_frame, window_pull_fusion,
    detect_window_mask_single, generate_sky
)
from pipeline.geometry import correct_verticals
from pipeline.finish import protect_white_points, apply_finish
from pipeline.ingestion import extract_exif, group_brackets_by_exif

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
    """Resize so the longest side is at most max_dim, preserving aspect ratio."""
    h, w = img.shape[:2]
    longest = max(h, w)
    if longest <= max_dim:
        return img
    scale = max_dim / longest
    new_w, new_h = int(w * scale), int(h * scale)
    return cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)


def synthesize_brackets(img: np.ndarray) -> List[np.ndarray]:
    """Create virtual brackets from a single shot (for single-photo fallback)."""
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
    """Run Enfuse on BGR uint8 frames. Returns fused result or None."""
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
        result = cv2.resize(result, (w, h), interpolation=cv2.INTER_AREA)
        print(f"  Enfuse OK. Output mean: {cv2.cvtColor(result, cv2.COLOR_BGR2GRAY).mean():.1f}")
        return result
    except Exception as e:
        print(f"  Enfuse error: {e}")
        return None
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


# ---------------------------------------------------------------------------
# BRACKET MERGE PIPELINE (multi-stage)
# ---------------------------------------------------------------------------

def bracket_merge(file_urls: List[str]) -> np.ndarray:
    """
    Multi-stage HDR merge pipeline:
      Stage A: Download, decode, align (SIFT)
      Stage B: Enfuse exposure fusion + window pull (dark bracket window detail)
      Stage C: Vertical correction (Hough transform)
      Stage D: White point protection + finish (clarity, sharpen, dither)
    """
    tmp_paths = []
    try:
        # Stage A: Download & decode
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

        single_shot = len(raw_images) == 1
        if single_shot:
            raw_images = synthesize_brackets(raw_images[0])

        # Stage A: Align brackets (SIFT)
        if len(raw_images) > 1:
            print("Aligning brackets (SIFT)...")
            ref_idx = len(raw_images) // 2
            raw_images = align_brackets(raw_images, ref_idx=ref_idx)
            gc.collect()

        # Denoise
        print("Denoising (bilateral)...")
        raw_images = [cv2.bilateralFilter(img, d=5, sigmaColor=45, sigmaSpace=45)
                      for img in raw_images]
        gc.collect()

        # Stage B: Enfuse merge (ambient base)
        print("Running Enfuse...")
        merged = enfuse_merge(raw_images)

        if merged is None:
            print("Mertens fallback fusion...")
            fused = cv2.createMergeMertens(
                contrast_weight=1.5,
                saturation_weight=1.2,
                exposure_weight=0.0,
            ).process(raw_images)
            merged = np.clip(fused * 255, 0, 255).astype(np.uint8)
            del fused; gc.collect()

        mean_out = cv2.cvtColor(merged, cv2.COLOR_BGR2GRAY).mean()
        print(f"Merge complete. Mean brightness: {mean_out:.1f}/255")

        # Stage B: Window pull (dark bracket window detail onto ambient base)
        if len(raw_images) >= 2:
            print("Running window pull...")
            dark_frame = raw_images[0]
            bright_frame = raw_images[-1]
            win_mask = build_window_mask(dark_frame, bright_frame)

            if win_mask.max() > 0.05:
                best_idx = select_best_window_frame(raw_images, win_mask)
                window_source = raw_images[best_idx]
                merged = window_pull_fusion(merged, window_source, win_mask)
                print("Window pull applied")
            else:
                print("No significant windows detected, skipping window pull")

        # Stage C: Vertical correction
        print("Correcting verticals...")
        merged = correct_verticals(merged)

        # Stage D: White point protection (prevent grey walls)
        print("Protecting white points...")
        merged = protect_white_points(merged)

        # Stage D: Finish (clarity + sharpen + dither)
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
    """
    Lightroom-style real estate enhancement — SINGLE float32 LAB pass.
    All adjustments in one LAB round-trip to avoid banding.
    Includes vertical correction and white point protection (grey wall fix).
    """
    ext = file_url.split("?")[0].rsplit(".", 1)[-1]
    ext = f".{ext.lower()}" if ext else ".jpg"
    tmp = download_file(file_url, ext)
    try:
        img = load_image_bgr(tmp)
        img = smart_resize(img)
        h, w = img.shape[:2]
        print(f"Loaded single photo at {w}×{h}")

        # 1. Light denoise
        img = cv2.bilateralFilter(img, d=4, sigmaColor=30, sigmaSpace=30)

        # 2. Vertical correction (before color adjustments)
        img = correct_verticals(img)

        # 3. SINGLE float32 LAB pass — ALL tone & color adjustments
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB).astype(np.float32)
        l = lab[:, :, 0].copy()
        a = lab[:, :, 1].copy()
        b = lab[:, :, 2].copy()

        # 3a. Exposure lift (gamma on L — high-key brightness)
        l = np.clip(np.power(l / 255.0, 0.48) * 255.0, 0, 255)

        # 3b. Lower highlights (masked to bright zones, feathered)
        hl_mask = np.clip((l - 215.0) / 40.0, 0, 1) ** 1.5
        hl_mask = cv2.GaussianBlur(hl_mask, (0, 0), sigmaX=20, sigmaY=20)
        l = np.clip(l - hl_mask * 45.0, 0, 255)

        # 3c. Lift shadows (masked to dark zones, feathered)
        sh_mask = np.clip((80.0 - l) / 80.0, 0, 1) ** 1.2
        sh_mask = cv2.GaussianBlur(sh_mask, (0, 0), sigmaX=20, sigmaY=20)
        l = np.clip(l + sh_mask * 65.0, 0, 255)

        # 3d. Whites & blacks (clean white ceilings, no crushed blacks)
        white_mask = np.clip((l - 200.0) / 55.0, 0, 1)
        black_mask = np.clip((40.0 - l) / 40.0, 0, 1)
        l = np.clip(l + white_mask * 18.0 + black_mask * 15.0, 0, 255)

        # 3e. White point protection — lock whites on smooth surfaces (prevent grey wall)
        bright_smooth = np.clip((l - 195.0) / 30.0, 0, 1)
        local_mean = cv2.blur(l, (15, 15))
        local_sq = cv2.blur(l ** 2, (15, 15))
        local_std = np.sqrt(np.maximum(local_sq - local_mean ** 2, 0))
        smooth = np.clip(1.0 - local_std / 15.0, 0, 1)
        wp_mask = bright_smooth * smooth
        wp_mask = cv2.GaussianBlur(wp_mask, (0, 0), sigmaX=10, sigmaY=10)
        l = np.clip(l + wp_mask * (250.0 - l) * 0.5, 0, 255)

        # 3f. White balance — neutralize green/yellow in bright zones (ceilings)
        bright_w = np.clip((l - 160.0) / 40.0, 0, 1)
        bright_w = cv2.GaussianBlur(bright_w, (0, 0), sigmaX=15, sigmaY=15)
        wb_pull = 0.85
        a = a * (1.0 - bright_w * wb_pull) + 128.0 * (bright_w * wb_pull)
        b = b * (1.0 - bright_w * wb_pull) + 128.0 * (bright_w * wb_pull)

        # 3g. Saturation — boost only in mid/dark zones (plants, furniture)
        sat_mask = 1.0 - bright_w
        sat_boost = 0.30
        a = np.clip(a + (a - 128.0) * sat_boost * sat_mask, 0, 255)
        b = np.clip(b + (b - 128.0) * sat_boost * sat_mask, 0, 255)

        # 3h. Clarity — large-radius unsharp on L (midtone pop, no CLAHE)
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

        # 4. Sky replacement (optional, off by default)
        if replace_sky:
            win_mask = detect_window_mask_single(img)
            if win_mask.max() > 0.01:
                sky = generate_sky(h, w)
                mask3 = win_mask[:, :, None]
                img = np.clip(img.astype(np.float32) * (1 - mask3) + sky * mask3, 0, 255).astype(np.uint8)
                print("Sky replaced in window zones")

        # 5. Light sharpen (BGR float32 — no LAB round-trip)
        img_f = img.astype(np.float32)
        blur_s = cv2.GaussianBlur(img_f, (0, 0), sigmaX=1.5, sigmaY=1.5)
        img_f = np.clip(img_f + 0.3 * (img_f - blur_s), 0, 255)

        # 6. Dither — ±1 noise breaks residual 8-bit gradient banding
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
    """Stage A: Auto-group files into brackets by EXIF metadata."""
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
