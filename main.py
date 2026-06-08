from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
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

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

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
    """Decode a RAW file to 8-bit BGR at full resolution."""
    import rawpy
    with rawpy.imread(path) as raw:
        rgb = raw.postprocess(
            use_camera_wb=True,
            no_auto_bright=False,
            output_bps=8,
            half_size=False,
            median_filter_passes=0,
        )
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    del rgb
    gc.collect()
    return bgr  # do NOT resize here — alignment needs matched sizes


def load_image_bgr(path: str) -> np.ndarray:
    """Load any image (RAW or JPEG/PNG) as BGR uint8."""
    ext = os.path.splitext(path)[1].lower()
    raw_exts = {".cr3", ".cr2", ".nef", ".arw", ".dng", ".raf", ".rw2", ".orf", ".raw"}
    if ext in raw_exts:
        return decode_raw(path)
    img = cv2.imread(path)
    if img is None:
        raise ValueError(f"Could not read image: {path}")
    return img


# ---------------------------------------------------------------------------
# Core HDR pipeline: AlignMTB → Debevec → window composite → Reinhard tonemap
# ---------------------------------------------------------------------------

def build_window_mask(dark_frame: np.ndarray, sigma: float = 10.0) -> np.ndarray:
    """
    Build a soft window mask from the DARKEST bracket frame.
    Only pixels EXTREMELY bright in the underexposed shot are true windows.
    Returns float32 mask [0..1], shape H×W×1 for broadcasting.
    """
    lum = cv2.cvtColor(dark_frame, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
    WIN_THRESH = 0.72  # raised — only true window glass, not bright walls/ceiling
    mask = np.clip((lum - WIN_THRESH) / (1.0 - WIN_THRESH), 0, 1) ** 2.0  # sharper falloff
    mask = cv2.GaussianBlur(mask, (0, 0), sigmaX=sigma, sigmaY=sigma)
    return np.clip(mask, 0, 1)[:, :, np.newaxis]


def bracket_merge(file_urls: List[str]) -> np.ndarray:
    """
    Professional HDR merge pipeline (no Debevec — Mertens is more robust):
      1. Download & decode all bracket frames
      2. Resize all to OUTPUT dimensions
      3. AlignMTB — correct camera shake between shots
      4. Sort frames dark→bright by mean luminance
      5. MergeMertens exposure fusion → clean 8-bit LDR base
         (Mertens needs no exposure times and handles RAW perfectly)
      6. Window composite: pull dark-frame window pixels BACK over the fusion
         result so glass/sky retains detail instead of being blown or grey
    """
    tmp_paths = []
    try:
        # ── 1. Download ────────────────────────────────────────────────────────
        print(f"Downloading {len(file_urls)} bracket frames...")
        for url in file_urls:
            ext = url.split("?")[0].rsplit(".", 1)[-1]
            ext = f".{ext.lower()}" if ext else ".jpg"
            tmp_paths.append(download_file(url, ext))

        # ── 2. Decode & resize ─────────────────────────────────────────────────
        images = []
        for p in tmp_paths:
            img = load_image_bgr(p)
            img = cv2.resize(img, (OUTPUT_WIDTH, OUTPUT_HEIGHT), interpolation=cv2.INTER_LANCZOS4)
            images.append(img)
        print(f"Loaded {len(images)} frames at {OUTPUT_WIDTH}x{OUTPUT_HEIGHT}")
        gc.collect()

        # ── 3. AlignMTB ────────────────────────────────────────────────────────
        if len(images) > 1:
            print("Aligning frames with AlignMTB...")
            align = cv2.createAlignMTB(max_bits=6, exclude_range=4, cut=True)
            align.process(images, images)
            gc.collect()

        # ── 4. Sort dark → bright ──────────────────────────────────────────────
        means = [cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).mean() for img in images]
        order = list(np.argsort(means))
        images = [images[i] for i in order]
        print(f"Frame means dark→bright: {[round(means[i], 1) for i in order]}")
        dark_frame = images[0]   # underexposed — best window detail

        # ── 5. Mertens exposure fusion ─────────────────────────────────────────
        if len(images) > 1:
            print("Fusing with Mertens...")
            fused = cv2.createMergeMertens(
                contrast_weight=1.4,
                saturation_weight=0.8,
                exposure_weight=0.0,  # 0 = purely contrast/saturation driven, no exposure bias
            ).process(images)
            merged = np.clip(fused * 255, 0, 255).astype(np.uint8)
            del fused
        else:
            merged = images[0].copy()
        gc.collect()

        # ── 6. Window composite ────────────────────────────────────────────────
        # Use the mid-exposure frame (not dark frame) to recover window detail.
        # This preserves natural colours (green foliage) without colour casts.
        if len(images) > 1:
            print("Compositing window detail...")
            win_mask = build_window_mask(dark_frame, sigma=8.0)  # H×W×1

            # Use middle exposure frame for windows — better colour balance than dark frame
            mid_idx = len(images) // 2
            mid_frame = images[mid_idx].astype(np.float32) / 255.0
            # Slightly tone down the mid frame's window area to avoid overexposure
            mid_frame = np.clip(mid_frame * 0.85, 0, 1)
            mid_u8 = (mid_frame * 255).astype(np.uint8)

            merged_f = merged.astype(np.float32)
            mid_win  = mid_u8.astype(np.float32)
            composited = merged_f * (1.0 - win_mask) + mid_win * win_mask
            composited = np.clip(composited, 0, 255).astype(np.uint8)
            del merged_f, mid_win, win_mask, mid_frame, mid_u8
            gc.collect()
        else:
            composited = merged

        print("Bracket merge complete.")
        return composited

    finally:
        for p in tmp_paths:
            try:
                os.unlink(p)
            except Exception:
                pass


# ---------------------------------------------------------------------------
# AutoHDR-style post-processing (finish layer — applied AFTER merge)
# ---------------------------------------------------------------------------

def apply_autohdr_finish(img_bgr: np.ndarray) -> np.ndarray:
    """
    AutoHDR "Classic" finish pipeline — 3-zone masked approach:
      ZONE A: Windows      — fully protected, dark-frame composited in bracket_merge
      ZONE B: Walls/Ceiling (bright interior) — desaturated to clean beige/white, light brightness push
      ZONE C: Dark interior (floors, furniture) — gentle fill light only

    Pipeline:
    1.  Window mask  — protect bright exterior pixels from any interior processing
    2.  Gamma        — applied to interior only (zones B+C)
    3.  Window pull  — soft cap on remaining blown highlights
    4.  Wall/ceiling desaturation mask — reduce saturation on bright interior zones → clean beige/white
    5.  Wall brightness push — gently lift walls/ceiling toward clean white
    6.  Fill light   — lift only deep shadows (zone C), preserve dark floors
    7.  Colour grade — warm cast, cut blue
    8.  Vibrance     — boost only low-saturation areas (floors/wood), skip walls
    9.  Denoise + Sharpen
    """
    img = img_bgr.astype(np.float32) / 255.0

    GAMMA              = 0.88   # slight pull-back — Mertens already lifts
    HI_START_RAW       = 0.80   # window pull threshold
    HI_CAP             = 0.88   # cap blown windows
    HI_STRENGTH        = 0.65
    # Wall/ceiling zone (bright interior, NOT windows)
    WALL_LUM_LOW       = 0.55   # walls start above this luminance
    WALL_LUM_HIGH      = 0.90   # above this = probably window, skip
    WALL_DESAT         = 0.55   # how much to desaturate walls/ceiling (0=none, 1=full grey)
    WALL_BRIGHT        = 0.08   # gentle brightness push on walls toward white
    # Shadow fill (dark interior only)
    FILL_CUTOFF        = 0.22   # only pixels below this lum get fill
    FILL_STRENGTH      = 0.14
    R_MULT             = 1.03
    G_MULT             = 0.98
    B_MULT             = 0.93
    VIBRANCE           = 14.0
    SHARPEN_AMT        = 0.60
    SHARPEN_RADIUS     = 1.0

    # ── 1. WINDOW MASK ────────────────────────────────────────────────────────
    # Tight mask: only true window glass (very high lum in raw image)
    lum_raw = 0.299 * img[:, :, 2] + 0.587 * img[:, :, 1] + 0.114 * img[:, :, 0]
    WIN_THRESH = 0.88
    win_protect = np.clip((lum_raw - WIN_THRESH) / (1.0 - WIN_THRESH), 0, 1) ** 1.2
    win_protect = cv2.GaussianBlur(win_protect, (0, 0), sigmaX=6, sigmaY=6)
    win_protect = np.clip(win_protect, 0, 1)
    win_protect3 = win_protect[:, :, np.newaxis]
    interior3    = 1.0 - win_protect3   # everything that is NOT a window

    # ── 2. GAMMA (interior only) ──────────────────────────────────────────────
    img_gamma = np.power(np.clip(img, 1e-6, 1.0), GAMMA)
    img = img_gamma * interior3 + img * win_protect3
    img = np.clip(img, 0, 1)

    # ── 3. WINDOW PULL ────────────────────────────────────────────────────────
    hi_mask = np.clip((lum_raw - HI_START_RAW) / (1.0 - HI_START_RAW), 0, 1) ** 1.5
    hi_mask3 = hi_mask[:, :, np.newaxis]
    img = img - hi_mask3 * np.clip(img - HI_CAP, 0, None) * HI_STRENGTH
    img = np.clip(img, 0, 1)

    # ── 4. WALL/CEILING MASK — bright interior pixels, excluding windows ──────
    # After gamma, recompute luminance to find wall/ceiling zone
    lum2 = 0.299 * img[:, :, 2] + 0.587 * img[:, :, 1] + 0.114 * img[:, :, 0]
    # Soft mask: pixels in [WALL_LUM_LOW .. WALL_LUM_HIGH] AND not a window
    wall_mask = np.clip((lum2 - WALL_LUM_LOW) / (WALL_LUM_HIGH - WALL_LUM_LOW), 0, 1)
    wall_mask = wall_mask * (1.0 - win_protect)  # exclude windows
    wall_mask = cv2.GaussianBlur(wall_mask.astype(np.float32), (0, 0), sigmaX=4, sigmaY=4)
    wall_mask = np.clip(wall_mask, 0, 1)
    wall_mask3 = wall_mask[:, :, np.newaxis]

    # ── 5a. DESATURATE walls/ceiling → clean beige/white ─────────────────────
    img_u8_tmp = (img * 255).astype(np.uint8)
    hsv_tmp = cv2.cvtColor(img_u8_tmp, cv2.COLOR_BGR2HSV).astype(np.float32)
    # Reduce saturation in wall zone
    hsv_tmp[:, :, 1] = hsv_tmp[:, :, 1] * (1.0 - wall_mask * WALL_DESAT)
    img_u8_tmp = cv2.cvtColor(np.clip(hsv_tmp, 0, 255).astype(np.uint8), cv2.COLOR_HSV2BGR)
    img = img_u8_tmp.astype(np.float32) / 255.0

    # ── 5b. BRIGHTNESS PUSH on walls/ceiling (gentle lift toward white) ───────
    img = img + wall_mask3 * WALL_BRIGHT * (1.0 - img)
    img = np.clip(img, 0, 1)

    # ── 6. FILL LIGHT — deep shadow zone (floors, dark furniture) only ────────
    lum3 = 0.299 * img[:, :, 2] + 0.587 * img[:, :, 1] + 0.114 * img[:, :, 0]
    fill_mask = np.clip(1.0 - lum3 / FILL_CUTOFF, 0, 1) ** 1.8
    fill_mask3 = fill_mask[:, :, np.newaxis]
    img = img + fill_mask3 * FILL_STRENGTH * (1.0 - img) * interior3
    img = np.clip(img, 0, 1)

    # ── 7. COLOUR GRADE ──────────────────────────────────────────────────────
    img[:, :, 2] = np.clip(img[:, :, 2] * R_MULT, 0, 1)
    img[:, :, 1] = np.clip(img[:, :, 1] * G_MULT, 0, 1)
    img[:, :, 0] = np.clip(img[:, :, 0] * B_MULT, 0, 1)

    # ── 8. VIBRANCE — skip walls (already desaturated), boost floors/wood ─────
    img_u8 = (img * 255).astype(np.uint8)
    hsv = cv2.cvtColor(img_u8, cv2.COLOR_BGR2HSV).astype(np.float32)
    sat_norm = hsv[:, :, 1] / 255.0
    # Apply vibrance only to non-wall zones to avoid re-saturating walls
    vib_boost = VIBRANCE * (1.0 - sat_norm) ** 1.5 * (1.0 - wall_mask)
    hsv[:, :, 1] = np.clip(hsv[:, :, 1] + vib_boost, 0, 255)
    img_u8 = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
    img = img_u8.astype(np.float32) / 255.0

    # ── 9. LUMINANCE-AWARE DENOISE ────────────────────────────────────────────
    img_u8 = (img * 255).astype(np.uint8)
    denoised = cv2.bilateralFilter(img_u8, d=9, sigmaColor=25, sigmaSpace=12)
    lum_denoise = 0.299 * img[:, :, 2] + 0.587 * img[:, :, 1] + 0.114 * img[:, :, 0]
    denoise_alpha = np.clip(1.0 - lum_denoise * 1.4, 0, 1)[:, :, np.newaxis]
    img_u8 = (img_u8 * (1.0 - denoise_alpha) + denoised * denoise_alpha).astype(np.uint8)

    # ── 10. SHARPEN ───────────────────────────────────────────────────────────
    pil_img = Image.fromarray(cv2.cvtColor(img_u8, cv2.COLOR_BGR2RGB))
    blurred  = pil_img.filter(ImageFilter.GaussianBlur(radius=SHARPEN_RADIUS))
    orig_f   = np.array(pil_img).astype(np.float32)
    blur_f   = np.array(blurred).astype(np.float32)
    sharpened = np.clip(orig_f + SHARPEN_AMT * (orig_f - blur_f), 0, 255).astype(np.uint8)
    result_bgr = cv2.cvtColor(sharpened, cv2.COLOR_RGB2BGR)

    print("AutoHDR finish applied.")
    return result_bgr


# ---------------------------------------------------------------------------
# Request model & endpoint
# ---------------------------------------------------------------------------

class MergeRequest(BaseModel):
    file_urls: List[str]
    bracket_name: str = "bracket"
    imagine_api_key: Optional[str] = None
    photomatix_api_key: Optional[str] = None   # kept for API compatibility, unused
    replace_sky: bool = False


@app.post("/merge")
async def merge_hdr(req: MergeRequest):
    if not req.file_urls:
        raise HTTPException(400, "No file URLs provided")
    if len(req.file_urls) not in (1, 3, 5):
        raise HTTPException(400, f"Expected 1, 3, or 5 files, got {len(req.file_urls)}")

    try:
        print(f"Starting bracket merge for '{req.bracket_name}' ({len(req.file_urls)} frames)...")
        merged = bracket_merge(req.file_urls)
        merged = apply_autohdr_finish(merged)

        pil = Image.fromarray(cv2.cvtColor(merged, cv2.COLOR_BGR2RGB))
        del merged
        gc.collect()

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


@app.get("/health")
def health():
    return {"status": "ok"}
