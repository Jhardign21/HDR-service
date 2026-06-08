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
    return bgr


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
# Tone curve: maps input [0..1] float array through a spline curve
# ---------------------------------------------------------------------------

def apply_tone_curve(img_f: np.ndarray, curve_in: List[float], curve_out: List[float]) -> np.ndarray:
    """
    Apply a tone curve to a float32 image [0..1].
    curve_in / curve_out are control points (0..1).
    Uses numpy linear interpolation across 256 LUT steps.
    """
    lut = np.interp(np.linspace(0, 1, 256), curve_in, curve_out).astype(np.float32)
    img_u8 = np.clip(img_f * 255, 0, 255).astype(np.uint8)
    result = lut[img_u8]
    return result.astype(np.float32)


# ---------------------------------------------------------------------------
# Window mask: luminance + blue-channel dominance
# ---------------------------------------------------------------------------

def build_window_mask(img: np.ndarray, sigma: float = 10.0) -> np.ndarray:
    """
    Detect bright exterior windows. Returns float32 mask [0..1] H×W×1.
    Uses the darkest available frame for best window isolation.
    """
    img_f = img.astype(np.float32) / 255.0
    lum   = 0.299 * img_f[:, :, 2] + 0.587 * img_f[:, :, 1] + 0.114 * img_f[:, :, 0]

    # Luminance: very bright zones
    lum_mask = np.clip((lum - 0.70) / (1.0 - 0.70), 0, 1) ** 1.5

    # Blue dominance: exterior sky/daylight is cooler than warm interior
    b_chan = img_f[:, :, 0]
    r_chan = img_f[:, :, 2]
    g_chan = img_f[:, :, 1]
    blue_dom = np.clip((b_chan - np.maximum(r_chan, g_chan) + 0.05) / 0.15, 0, 1)
    blue_dom = blue_dom * (lum > 0.40).astype(np.float32)

    combined = np.clip(lum_mask * 0.75 + blue_dom * 0.25, 0, 1)
    combined = cv2.GaussianBlur(combined.astype(np.float32), (0, 0), sigmaX=sigma, sigmaY=sigma)
    return np.clip(combined, 0, 1)[:, :, np.newaxis]


# ---------------------------------------------------------------------------
# Ghost detection
# ---------------------------------------------------------------------------

def detect_ghost_mask(images: List[np.ndarray], ref_idx: int) -> np.ndarray:
    ref = cv2.cvtColor(images[ref_idx], cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
    diffs = []
    for i, img in enumerate(images):
        if i == ref_idx:
            continue
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
        diffs.append(np.abs(gray - ref))
    if not diffs:
        return np.ones(ref.shape, dtype=np.float32)
    max_diff = np.max(np.stack(diffs, axis=0), axis=0)
    ghost_prob = np.clip((max_diff - 0.15) / 0.20, 0, 1)
    ghost_prob = cv2.GaussianBlur(ghost_prob, (0, 0), sigmaX=6, sigmaY=6)
    return (1.0 - np.clip(ghost_prob, 0, 0.85)).astype(np.float32)


# ---------------------------------------------------------------------------
# Synthetic bracket generation
# ---------------------------------------------------------------------------

def synthesize_brackets(img: np.ndarray) -> List[np.ndarray]:
    img_f  = img.astype(np.float32) / 255.0
    dark   = np.clip(np.power(img_f, 0.40), 0, 1)   # simulate -2EV
    bright = np.clip(np.power(img_f, 2.20), 0, 1)   # simulate +2EV
    print("Synthesized dark/bright virtual brackets from single exposure.")
    return [(dark * 255).astype(np.uint8), img, (bright * 255).astype(np.uint8)]


# ---------------------------------------------------------------------------
# FLASH SIMULATION — the core technique used by Autoenhance/flambient pros
#
# The key insight from research: professional results use the BRIGHT exposure
# as a "simulated flash frame" for walls/ceilings, blended over the ambient
# base using a luminosity mask. This is what makes interiors pop open without
# halos or the artificial HDR look.
# ---------------------------------------------------------------------------

def simulate_flash_blend(images: List[np.ndarray], win_mask3: np.ndarray) -> np.ndarray:
    """
    Flambient-style flash simulation using the brightest bracket frame.

    Pro technique (from Esoft/Autoenhance research):
      1. Use the BRIGHTEST frame as the "flash layer" (simulates bounced flash)
      2. Create a luminosity mask from that frame — bright interior zones only
      3. Blend flash layer OVER the Mertens base using the mask
      4. The mask feathers naturally at edges → no halos, no artifacts
      5. Window zones are EXCLUDED from flash layer (protected by win_mask)

    This is exactly what manual flambient editors do in Photoshop:
    - Ambient layer on bottom (natural shadows, mood)
    - Flash layer on top with luminosity mask painted on walls/ceiling/furniture
    """
    # Sort frames dark → bright
    means = [cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).mean() for img in images]
    order = np.argsort(means)
    bright_frame = images[order[-1]]  # brightest = flash equivalent
    dark_frame   = images[order[0]]   # darkest  = window detail source

    return bright_frame, dark_frame


# ---------------------------------------------------------------------------
# Local contrast enhancement via large-radius unsharp mask on L-channel
# (Creates the "3D pop" look without noise — superior to CLAHE for this task)
# ---------------------------------------------------------------------------

def local_contrast_enhance(img_bgr: np.ndarray, radius: float = 40.0, amount: float = 0.25) -> np.ndarray:
    """
    Large-radius unsharp mask on LAB L-channel.

    This is the technique that creates the professional "dimensional" look:
    - Large radius (40px) targets macro contrast zones (walls vs ceiling vs floor)
    - Small amount (0.25) is subtle — just enough to separate depth planes
    - Applied on L-channel only → zero colour contamination
    - No noise amplification (unlike CLAHE which pumps up local grain)
    """
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    l   = lab[:, :, 0] / 255.0

    # Gaussian blur at large radius to extract low-frequency luminance
    blurred = cv2.GaussianBlur(l, (0, 0), sigmaX=radius, sigmaY=radius)

    # Unsharp mask: enhance the difference between pixel and low-freq version
    l_enhanced = np.clip(l + amount * (l - blurred), 0, 1)

    lab[:, :, 0] = l_enhanced * 255.0
    return cv2.cvtColor(np.clip(lab, 0, 255).astype(np.uint8), cv2.COLOR_LAB2BGR)


# ---------------------------------------------------------------------------
# Core HDR merge pipeline
# ---------------------------------------------------------------------------

def bracket_merge(file_urls: List[str]) -> np.ndarray:
    """
    Professional HDR merge pipeline using flambient-inspired compositing:

      1.  Download & decode frames
      2.  Resize to output dimensions
      3.  Early NLMeans denoise (prevents noise amplification downstream)
      4.  AlignMTB alignment
      5.  Sort frames dark → bright
      6.  Synthesize brackets for single-shot input
      7.  Mertens fusion (base/ambient layer)
      8.  Flash composite: blend bright frame over Mertens base via luminosity mask
      9.  Window composite: pull dark frame into window zones
    """
    tmp_paths = []
    try:
        print(f"Downloading {len(file_urls)} bracket frames...")
        for url in file_urls:
            ext = url.split("?")[0].rsplit(".", 1)[-1]
            ext = f".{ext.lower()}" if ext else ".jpg"
            tmp_paths.append(download_file(url, ext))

        images = []
        for p in tmp_paths:
            img = load_image_bgr(p)
            img = cv2.resize(img, (OUTPUT_WIDTH, OUTPUT_HEIGHT), interpolation=cv2.INTER_LANCZOS4)
            images.append(img)
        print(f"Loaded {len(images)} frames at {OUTPUT_WIDTH}x{OUTPUT_HEIGHT}")
        gc.collect()

        # ── Early denoise — before any enhancement ────────────────────────────
        print("Denoising bracket frames early...")
        images = [cv2.fastNlMeansDenoisingColored(img, None, h=5, hColor=5,
                  templateWindowSize=7, searchWindowSize=21) for img in images]
        gc.collect()

        # ── AlignMTB ──────────────────────────────────────────────────────────
        if len(images) > 1:
            print("Aligning frames...")
            align = cv2.createAlignMTB(max_bits=6, exclude_range=4, cut=True)
            align.process(images, images)
            gc.collect()

        # ── Sort dark → bright ────────────────────────────────────────────────
        means = [cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).mean() for img in images]
        order = list(np.argsort(means))
        images = [images[i] for i in order]
        print(f"Frame means dark→bright: {[round(means[i], 1) for i in order]}")

        # ── Synthesize brackets for single-shot input ─────────────────────────
        single_shot = len(images) == 1
        if single_shot:
            print("Single exposure — synthesizing virtual brackets...")
            images = synthesize_brackets(images[0])

        # ── Ghost detection & deghosting ──────────────────────────────────────
        mid_idx = len(images) // 2
        if len(images) > 1 and not single_shot:
            print("Ghost detection...")
            clean_weight = detect_ghost_mask(images, ref_idx=mid_idx)
            ref_f    = images[mid_idx].astype(np.float32)
            clean_w3 = clean_weight[:, :, np.newaxis]
            deghosted = []
            for i, img in enumerate(images):
                if i == mid_idx:
                    deghosted.append(img)
                    continue
                blended = img.astype(np.float32) * clean_w3 + ref_f * (1.0 - clean_w3)
                deghosted.append(np.clip(blended, 0, 255).astype(np.uint8))
            images = deghosted
            del deghosted, ref_f
            gc.collect()

        # ── Mertens fusion (ambient base layer) ───────────────────────────────
        print("Mertens fusion (ambient base)...")
        fused  = cv2.createMergeMertens(
            contrast_weight=1.0,
            saturation_weight=0.8,
            exposure_weight=0.0,
        ).process(images)
        mertens_base = np.clip(fused * 255, 0, 255).astype(np.uint8)
        del fused
        gc.collect()

        # ── Flash composite: brightest frame → interior surfaces ──────────────
        # Key insight from flambient research: use bright frame as "flash layer"
        # blended over the Mertens base via a luminosity mask.
        # This is what makes walls/ceilings pop open without HDR artifacts.
        print("Flash composite (flambient simulation)...")
        bright_frame, dark_frame = simulate_flash_blend(images, None)

        # Build window mask from dark frame (best isolation of exterior zones)
        win_mask3 = build_window_mask(dark_frame, sigma=10.0)
        interior3 = 1.0 - win_mask3

        # Luminosity mask for flash blend: mid-bright interior zones
        # (not shadows which should retain ambient depth, not windows)
        bright_f   = bright_frame.astype(np.float32) / 255.0
        mertens_f  = mertens_base.astype(np.float32) / 255.0
        lum_bright = 0.299 * bright_f[:, :, 2] + 0.587 * bright_f[:, :, 1] + 0.114 * bright_f[:, :, 0]

        # Flash mask: mid-tones to highlights of the bright frame, interior only
        # Low end cutoff at 0.20 so deep shadows keep ambient depth (not flashed flat)
        flash_mask = np.clip((lum_bright - 0.20) / (0.85 - 0.20), 0, 1) ** 1.2
        flash_mask = flash_mask * interior3[:, :, 0]
        flash_mask = cv2.GaussianBlur(flash_mask.astype(np.float32), (0, 0), sigmaX=8, sigmaY=8)
        flash_mask = np.clip(flash_mask, 0, 1)[:, :, np.newaxis]

        # Blend: ambient base + flash layer weighted by flash_mask
        # Flash blend strength 0.72 — strong enough to open the room, soft enough to keep depth
        FLASH_STRENGTH = 0.72
        composited_f = mertens_f * (1.0 - flash_mask * FLASH_STRENGTH) + bright_f * (flash_mask * FLASH_STRENGTH)
        composited_f = np.clip(composited_f, 0, 1)

        # ── Window composite: dark frame for window zones ─────────────────────
        dark_f = dark_frame.astype(np.float32) / 255.0
        composited_f = composited_f * (1.0 - win_mask3) + dark_f * win_mask3
        composited = np.clip(composited_f * 255, 0, 255).astype(np.uint8)
        del mertens_f, bright_f, dark_f, flash_mask, win_mask3
        gc.collect()

        print("Bracket merge complete.")
        return composited

    finally:
        for p in tmp_paths:
            try:
                os.unlink(p)
            except Exception:
                pass


# ---------------------------------------------------------------------------
# Post-processing finish
# ---------------------------------------------------------------------------

def apply_autohdr_finish(img_bgr: np.ndarray) -> np.ndarray:
    """
    Professional finish pass:

      1. S-curve tone mapping (shadows lift, highlights protect) — NOT just gamma
      2. Local contrast via large-radius unsharp mask on L-channel (3D pop, no grain)
      3. Window highlight recovery
      4. Wall/ceiling zone: desaturate + grey-world cast removal + brightness push
      5. Shadow fill for deep corners
      6. Neutral colour grade (5500K interior standard)
      7. Moderate vibrance (skip walls)
      8. Mild sharpening

    KEY CHANGES vs previous version:
      - Replaced gamma with proper S-CURVE: lifts shadows more aggressively while
        keeping highlights anchored. This is the #1 technique pros use in Lightroom.
      - Replaced post-gamma CLAHE with large-radius unsharp mask on L-channel:
        produces the same local contrast "pop" WITHOUT amplifying grain.
      - Removed all bilateral filter (was re-blurring after sharpen).
      - Wall grey-world caps are tighter (±8%) to prevent over-bluing.
    """
    img = img_bgr.astype(np.float32) / 255.0

    # ── TUNING PARAMETERS ─────────────────────────────────────────────────────
    # S-curve control points: [input] → [output]
    # Lifts shadows aggressively, protects highlights, keeps mid-tones natural
    SCURVE_IN  = [0.0,  0.10, 0.25, 0.50, 0.75, 0.90, 1.0]
    SCURVE_OUT = [0.0,  0.18, 0.38, 0.62, 0.80, 0.92, 1.0]

    HI_START_RAW       = 0.82
    HI_CAP             = 0.92
    HI_STRENGTH        = 0.55
    WALL_LUM_LOW       = 0.45
    WALL_LUM_HIGH      = 0.93
    WALL_DESAT         = 0.50
    WALL_BRIGHT        = 0.10
    WALL_CAST_STRENGTH = 0.30   # tighter — ±8% cap prevents over-bluing
    FILL_CUTOFF        = 0.38
    FILL_STRENGTH      = 0.38
    R_MULT             = 0.99
    G_MULT             = 1.00
    B_MULT             = 0.97
    VIBRANCE           = 10.0
    SHARPEN_AMT        = 0.50
    SHARPEN_RADIUS     = 0.8
    LOCAL_CONTRAST_R   = 40.0   # large-radius unsharp mask radius (px)
    LOCAL_CONTRAST_A   = 0.22   # amount — subtle but visible depth separation

    # ── 1. WINDOW MASK ────────────────────────────────────────────────────────
    lum_raw = 0.299 * img[:, :, 2] + 0.587 * img[:, :, 1] + 0.114 * img[:, :, 0]
    b_chan   = img[:, :, 0]
    r_chan   = img[:, :, 2]
    g_chan   = img[:, :, 1]
    lum_win  = np.clip((lum_raw - 0.85) / (1.0 - 0.85), 0, 1) ** 1.2
    blue_dom = np.clip((b_chan - np.maximum(r_chan, g_chan) + 0.05) / 0.12, 0, 1)
    blue_dom = blue_dom * (lum_raw > 0.50).astype(np.float32)
    win_raw  = np.clip(lum_win * 0.7 + blue_dom * 0.3, 0, 1)
    win_protect  = cv2.GaussianBlur(win_raw.astype(np.float32), (0, 0), sigmaX=8, sigmaY=8)
    win_protect  = np.clip(win_protect, 0, 1)
    win_protect3 = win_protect[:, :, np.newaxis]
    interior3    = 1.0 - win_protect3

    # ── 2. S-CURVE TONE MAPPING (interior only) ───────────────────────────────
    # This replaces simple gamma. An S-curve lifts shadows aggressively while
    # anchoring highlights — exactly what Lightroom's "Shadows +80, Whites -20" does.
    # It's the single most important change to get the bright/open interior look.
    img_curved  = apply_tone_curve(img, SCURVE_IN, SCURVE_OUT)
    img = img_curved * interior3 + img * win_protect3
    img = np.clip(img, 0, 1)

    # ── 3. LOCAL CONTRAST — large-radius unsharp mask on L-channel ───────────
    # Creates the professional "dimensional" pop:
    # - Large radius (40px) = macro contrast, not micro texture
    # - Applied on L only = zero colour shift
    # - No noise (unlike CLAHE which amplifies grain in dark zones)
    img_u8 = (img * 255).astype(np.uint8)
    img_u8 = local_contrast_enhance(img_u8, radius=LOCAL_CONTRAST_R, amount=LOCAL_CONTRAST_A)
    img    = img_u8.astype(np.float32) / 255.0

    # ── 4. WINDOW HIGHLIGHT PULL ─────────────────────────────────────────────
    hi_mask  = np.clip((lum_raw - HI_START_RAW) / (1.0 - HI_START_RAW), 0, 1) ** 1.5
    hi_mask3 = hi_mask[:, :, np.newaxis]
    img = img - hi_mask3 * np.clip(img - HI_CAP, 0, None) * HI_STRENGTH
    img = np.clip(img, 0, 1)

    # ── 5. WALL/CEILING ZONE ─────────────────────────────────────────────────
    lum2 = 0.299 * img[:, :, 2] + 0.587 * img[:, :, 1] + 0.114 * img[:, :, 0]
    wall_mask = np.clip((lum2 - WALL_LUM_LOW) / (WALL_LUM_HIGH - WALL_LUM_LOW), 0, 1)
    wall_mask = wall_mask * (1.0 - win_protect)
    wall_mask = cv2.GaussianBlur(wall_mask.astype(np.float32), (0, 0), sigmaX=6, sigmaY=6)
    wall_mask = np.clip(wall_mask, 0, 1)
    wall_mask3 = wall_mask[:, :, np.newaxis]

    # Desaturate walls → clean neutral tone
    img_u8_tmp = (img * 255).astype(np.uint8)
    hsv_tmp = cv2.cvtColor(img_u8_tmp, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv_tmp[:, :, 1] = hsv_tmp[:, :, 1] * (1.0 - wall_mask * WALL_DESAT)
    img = cv2.cvtColor(np.clip(hsv_tmp, 0, 255).astype(np.uint8), cv2.COLOR_HSV2BGR).astype(np.float32) / 255.0

    # Grey-world cast removal on wall zone (tight ±8% cap)
    wall_sum = wall_mask.sum() + 1e-6
    mean_r   = (img[:, :, 2] * wall_mask).sum() / wall_sum
    mean_g   = (img[:, :, 1] * wall_mask).sum() / wall_sum
    mean_b   = (img[:, :, 0] * wall_mask).sum() / wall_sum
    mean_all = (mean_r + mean_g + mean_b) / 3.0 + 1e-6
    cor_r = np.clip(mean_all / (mean_r + 1e-6), 0.92, 1.08)
    cor_g = np.clip(mean_all / (mean_g + 1e-6), 0.92, 1.08)
    cor_b = np.clip(mean_all / (mean_b + 1e-6), 0.92, 1.08)
    img[:, :, 2] = np.clip(img[:, :, 2] * (1.0 + (cor_r - 1.0) * wall_mask * WALL_CAST_STRENGTH), 0, 1)
    img[:, :, 1] = np.clip(img[:, :, 1] * (1.0 + (cor_g - 1.0) * wall_mask * WALL_CAST_STRENGTH), 0, 1)
    img[:, :, 0] = np.clip(img[:, :, 0] * (1.0 + (cor_b - 1.0) * wall_mask * WALL_CAST_STRENGTH), 0, 1)

    # Brightness push on walls → toward clean bright tone
    img = img + wall_mask3 * WALL_BRIGHT * (1.0 - img)
    img = np.clip(img, 0, 1)

    # ── 6. SHADOW FILL — deep corners ────────────────────────────────────────
    lum3 = 0.299 * img[:, :, 2] + 0.587 * img[:, :, 1] + 0.114 * img[:, :, 0]
    fill_mask  = np.clip(1.0 - lum3 / FILL_CUTOFF, 0, 1) ** 1.5
    fill_mask3 = fill_mask[:, :, np.newaxis]
    img = img + fill_mask3 * FILL_STRENGTH * (1.0 - img) * interior3
    img = np.clip(img, 0, 1)

    # ── 7. COLOUR GRADE (neutral 5500K interior standard) ────────────────────
    img[:, :, 2] = np.clip(img[:, :, 2] * R_MULT, 0, 1)
    img[:, :, 1] = np.clip(img[:, :, 1] * G_MULT, 0, 1)
    img[:, :, 0] = np.clip(img[:, :, 0] * B_MULT, 0, 1)

    # ── 8. VIBRANCE (moderate, skip walls — already neutralised) ─────────────
    img_u8 = (img * 255).astype(np.uint8)
    hsv    = cv2.cvtColor(img_u8, cv2.COLOR_BGR2HSV).astype(np.float32)
    sat_norm  = hsv[:, :, 1] / 255.0
    vib_zone  = np.clip(1.0 - wall_mask, 0, 1)
    vib_boost = VIBRANCE * (1.0 - sat_norm) ** 1.5 * vib_zone
    hsv[:, :, 1] = np.clip(hsv[:, :, 1] + vib_boost, 0, 255)
    img = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR).astype(np.float32) / 255.0

    # ── 9. SHARPEN ───────────────────────────────────────────────────────────
    img_u8   = (img * 255).astype(np.uint8)
    pil_img  = Image.fromarray(cv2.cvtColor(img_u8, cv2.COLOR_BGR2RGB))
    blurred  = pil_img.filter(ImageFilter.GaussianBlur(radius=SHARPEN_RADIUS))
    orig_f   = np.array(pil_img).astype(np.float32)
    blur_f   = np.array(blurred).astype(np.float32)
    sharpened   = np.clip(orig_f + SHARPEN_AMT * (orig_f - blur_f), 0, 255).astype(np.uint8)
    result_bgr  = cv2.cvtColor(sharpened, cv2.COLOR_RGB2BGR)

    print("AutoHDR finish applied.")
    return result_bgr


# ---------------------------------------------------------------------------
# Request model & endpoint
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
