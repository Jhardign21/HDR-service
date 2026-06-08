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
    """Decode RAW → 8-bit BGR. Let rawpy auto-brighten so we start at a useful exposure."""
    import rawpy
    with rawpy.imread(path) as raw:
        rgb = raw.postprocess(
            use_camera_wb=True,
            no_auto_bright=False,   # allow auto-brightening
            bright=1.0,
            output_bps=8,
            half_size=False,
            median_filter_passes=0,
        )
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    del rgb
    gc.collect()
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
# Ensures each frame is at a visually usable brightness before processing.
# Without this, dark frames stay dark and every downstream operation fails.
# ---------------------------------------------------------------------------

def normalise_exposure(img: np.ndarray, target_mean: float = 110.0) -> np.ndarray:
    """
    Scale image so its mean grey value hits target_mean.
    Caps the scale factor to avoid extreme amplification.
    This is the equivalent of Lightroom's Auto Exposure button — the very
    first thing every professional editor does before any other adjustment.
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    current_mean = float(gray.mean()) + 1e-6
    scale = min(target_mean / current_mean, 3.5)   # never amplify more than ×3.5
    if scale <= 1.02:
        return img  # already bright enough, no change
    img_f = img.astype(np.float32) * scale
    return np.clip(img_f, 0, 255).astype(np.uint8)


# ---------------------------------------------------------------------------
# Tone-curve helper (LUT based, float32 [0..1])
# ---------------------------------------------------------------------------

def apply_tone_curve(img_f: np.ndarray, curve_in: List[float], curve_out: List[float]) -> np.ndarray:
    lut = np.interp(np.linspace(0, 1, 256), curve_in, curve_out).astype(np.float32)
    img_u8 = np.clip(img_f * 255, 0, 255).astype(np.uint8)
    return lut[img_u8].astype(np.float32)


# ---------------------------------------------------------------------------
# Window mask  (luminance + blue-channel dominance)
# ---------------------------------------------------------------------------

def build_window_mask(img_bgr: np.ndarray, sigma: float = 12.0) -> np.ndarray:
    """
    Detect bright exterior-window zones.
    Returns float32 mask [0..1] H×W×1.
    We use the DARKEST frame so interior mid-tones don't bleed into the mask.
    """
    f = img_bgr.astype(np.float32) / 255.0
    lum   = 0.299 * f[:, :, 2] + 0.587 * f[:, :, 1] + 0.114 * f[:, :, 0]
    b, g, r = f[:, :, 0], f[:, :, 1], f[:, :, 2]

    # Very bright → likely window
    lum_mask = np.clip((lum - 0.72) / (1.0 - 0.72), 0, 1) ** 1.5

    # Blue-dominant → exterior daylight
    blue_dom = np.clip((b - np.maximum(r, g) + 0.05) / 0.15, 0, 1)
    blue_dom *= (lum > 0.45).astype(np.float32)

    raw = np.clip(lum_mask * 0.75 + blue_dom * 0.25, 0, 1)
    raw = cv2.GaussianBlur(raw.astype(np.float32), (0, 0), sigmaX=sigma, sigmaY=sigma)
    return np.clip(raw, 0, 1)[:, :, np.newaxis]


# ---------------------------------------------------------------------------
# Ghost detection / deghosting
# ---------------------------------------------------------------------------

def deghost(images: List[np.ndarray], ref_idx: int) -> List[np.ndarray]:
    ref = cv2.cvtColor(images[ref_idx], cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
    diffs = []
    for i, img in enumerate(images):
        if i == ref_idx:
            diffs.append(np.zeros_like(ref))
            continue
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
        diffs.append(np.abs(gray - ref))
    max_diff = np.max(np.stack(diffs, axis=0), axis=0)
    ghost_prob = np.clip((max_diff - 0.12) / 0.20, 0, 1)
    ghost_prob = cv2.GaussianBlur(ghost_prob, (0, 0), sigmaX=6, sigmaY=6)
    clean_w = (1.0 - np.clip(ghost_prob, 0, 0.85))[:, :, np.newaxis]
    ref_f = images[ref_idx].astype(np.float32)
    result = []
    for i, img in enumerate(images):
        if i == ref_idx:
            result.append(img)
            continue
        blended = img.astype(np.float32) * clean_w + ref_f * (1.0 - clean_w)
        result.append(np.clip(blended, 0, 255).astype(np.uint8))
    return result


# ---------------------------------------------------------------------------
# Synthetic brackets for single-shot input
# ---------------------------------------------------------------------------

def synthesize_brackets(img: np.ndarray) -> List[np.ndarray]:
    f = img.astype(np.float32) / 255.0
    # dark: simulate -2 EV  (power > 1 darkens)
    dark   = np.clip(np.power(f, 1.8),  0, 1)
    # bright: simulate +2 EV (power < 1 brightens)
    bright = np.clip(np.power(f, 0.45), 0, 1)
    print("Synthesized virtual brackets from single exposure.")
    return [(dark * 255).astype(np.uint8), img, (bright * 255).astype(np.uint8)]


# ---------------------------------------------------------------------------
# Large-radius unsharp mask on L-channel ("3D pop", no noise)
# ---------------------------------------------------------------------------

def local_contrast_enhance(img_bgr: np.ndarray, radius: float = 45.0, amount: float = 0.20) -> np.ndarray:
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    l   = lab[:, :, 0] / 255.0
    blurred = cv2.GaussianBlur(l, (0, 0), sigmaX=radius, sigmaY=radius)
    lab[:, :, 0] = np.clip(l + amount * (l - blurred), 0, 1) * 255.0
    return cv2.cvtColor(np.clip(lab, 0, 255).astype(np.uint8), cv2.COLOR_LAB2BGR)


# ---------------------------------------------------------------------------
# CORE MERGE  (flambient-inspired)
# ---------------------------------------------------------------------------

def bracket_merge(file_urls: List[str]) -> np.ndarray:
    """
    Pipeline:
      1. Download & decode
      2. Resize
      3. EXPOSURE NORMALISE each frame  ← critical new step
      4. Early NLMeans denoise
      5. AlignMTB
      6. Sort dark→bright
      7. Synthesize brackets if single-shot
      8. Mertens fusion (ambient base)
      9. Flash composite: brightest frame blended in via luminosity mask
     10. Window composite: darkest frame pulled into window zones
    """
    tmp_paths = []
    try:
        print(f"Downloading {len(file_urls)} frames...")
        for url in file_urls:
            ext = url.split("?")[0].rsplit(".", 1)[-1]
            ext = f".{ext.lower()}" if ext else ".jpg"
            tmp_paths.append(download_file(url, ext))

        images = []
        for p in tmp_paths:
            img = load_image_bgr(p)
            img = cv2.resize(img, (OUTPUT_WIDTH, OUTPUT_HEIGHT), interpolation=cv2.INTER_LANCZOS4)
            images.append(img)
        print(f"Loaded {len(images)} frames at {OUTPUT_WIDTH}×{OUTPUT_HEIGHT}")

        # ── Step 3: Exposure normalisation ────────────────────────────────────
        # THE most important step: if input frames are dark, every single
        # downstream algorithm (Mertens, CLAHE, masking) will fail.
        # Professional software (Lightroom, Capture One, Autoenhance) applies
        # Auto Exposure as the very first step before any HDR merge.
        # Target mean 110 ≈ properly exposed indoor JPEG.
        print("Normalising frame exposures...")
        means_before = [cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).mean() for img in images]
        print(f"  Means before: {[round(m, 1) for m in means_before]}")
        images = [normalise_exposure(img, target_mean=110.0) for img in images]
        means_after = [cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).mean() for img in images]
        print(f"  Means after:  {[round(m, 1) for m in means_after]}")
        gc.collect()

        # ── Step 4: Early denoise ─────────────────────────────────────────────
        print("Denoising...")
        images = [cv2.fastNlMeansDenoisingColored(img, None, h=5, hColor=5,
                  templateWindowSize=7, searchWindowSize=21) for img in images]
        gc.collect()

        # ── Step 5: Align ─────────────────────────────────────────────────────
        if len(images) > 1:
            print("Aligning...")
            align = cv2.createAlignMTB(max_bits=6, exclude_range=4, cut=True)
            align.process(images, images)
            gc.collect()

        # ── Step 6: Sort dark→bright ──────────────────────────────────────────
        means = [cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).mean() for img in images]
        order = list(np.argsort(means))
        images = [images[i] for i in order]
        dark_frame   = images[0]
        bright_frame = images[-1]
        print(f"Frame means dark→bright: {[round(means[i], 1) for i in order]}")

        # ── Step 7: Synthesize brackets for single-shot ───────────────────────
        single_shot = len(images) == 1
        if single_shot:
            images = synthesize_brackets(images[0])
            dark_frame   = images[0]
            bright_frame = images[-1]

        # ── Step 8: Brightest frame as base + window pull from darkest ───────────
        # This is exactly how professional flambient editors work:
        #   - Start with the BRIGHTEST frame → interior is already open and bright
        #   - Composite window zones from the DARKEST frame → clean exterior view
        #   - For multi-bracket: blend Mertens only into deep shadows to add depth
        #
        # Mertens as the primary base was the problem — it averages exposures
        # and pulls brightness DOWN. Starting from the bright frame is the fix.
        print("Using brightest frame as interior base...")
        bright_f  = bright_frame.astype(np.float32) / 255.0
        dark_f    = dark_frame.astype(np.float32) / 255.0

        # Window mask from darkest frame (best exterior isolation before normalisation lifts it)
        win_mask3 = build_window_mask(dark_frame, sigma=12.0)
        interior3 = 1.0 - win_mask3

        # Base composite: bright interior + dark window zones
        composited_f = bright_f * (1.0 - win_mask3) + dark_f * win_mask3
        composited_f = np.clip(composited_f, 0, 1)

        # For real multi-bracket: blend Mertens ONLY into deep shadow zones
        # so corners retain natural depth without dragging down bright areas
        if not single_shot and len(images) > 1:
            images = deghost(images, ref_idx=len(images) // 2)
            gc.collect()
            fused = cv2.createMergeMertens(
                contrast_weight=1.0,
                saturation_weight=0.8,
                exposure_weight=0.0,
            ).process(images)
            mertens_f = np.clip(fused * 255, 0, 255).astype(np.uint8).astype(np.float32) / 255.0
            del fused
            gc.collect()

            # Shadow blend mask: only very dark interior zones (lum < 0.25)
            lum_bright = 0.299 * bright_f[:, :, 2] + 0.587 * bright_f[:, :, 1] + 0.114 * bright_f[:, :, 0]
            shadow_blend = np.clip(1.0 - lum_bright / 0.25, 0, 1) ** 2.0
            shadow_blend = shadow_blend * interior3[:, :, 0]
            shadow_blend = cv2.GaussianBlur(shadow_blend.astype(np.float32), (0, 0), sigmaX=8, sigmaY=8)
            shadow_blend = np.clip(shadow_blend, 0, 0.50)[:, :, np.newaxis]  # cap at 50%

            composited_f = composited_f * (1.0 - shadow_blend) + mertens_f * shadow_blend
            composited_f = np.clip(composited_f, 0, 1)
            del mertens_f, shadow_blend

        composited = np.clip(composited_f * 255, 0, 255).astype(np.uint8)
        del bright_f, dark_f, win_mask3, composited_f
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
# POST-PROCESSING FINISH
# ---------------------------------------------------------------------------

def apply_autohdr_finish(img_bgr: np.ndarray) -> np.ndarray:
    """
    Finish pass — assumes input is NOW properly exposed (mean ~100-130).

    Steps:
      1.  Gentle S-curve: subtle lift of remaining shadows, anchor highlights
      2.  Large-radius unsharp mask on L-channel (3D depth pop, zero noise)
      3.  Window highlight protection
      4.  Wall/ceiling zone: desaturate, cast removal, brightness push
      5.  Shadow fill for residual dark corners
      6.  Colour grade (neutral warm 5500K)
      7.  Vibrance
      8.  Sharpening

    NOTE: S-curve is now GENTLE because exposure normalisation already did the
    heavy lifting. A strong curve on an already-bright image would clip highlights.
    """
    img = img_bgr.astype(np.float32) / 255.0

    # ── PARAMETERS ────────────────────────────────────────────────────────────
    # Gentle S-curve: input is now bright, so we only need a soft lift
    SCURVE_IN  = [0.0,  0.08, 0.25, 0.50, 0.75, 0.92, 1.0]
    SCURVE_OUT = [0.0,  0.13, 0.32, 0.56, 0.76, 0.93, 1.0]

    HI_START_RAW       = 0.82
    HI_CAP             = 0.93
    HI_STRENGTH        = 0.50
    WALL_LUM_LOW       = 0.30   # lowered from 0.45 so more surface area is caught
    WALL_LUM_HIGH      = 0.95
    WALL_DESAT         = 0.45
    WALL_BRIGHT        = 0.08
    WALL_CAST_STRENGTH = 0.25
    FILL_CUTOFF        = 0.30   # only truly dark corners
    FILL_STRENGTH      = 0.40
    R_MULT             = 1.00
    G_MULT             = 1.00
    B_MULT             = 0.97
    VIBRANCE           = 12.0
    SHARPEN_AMT        = 0.45
    SHARPEN_RADIUS     = 0.8
    LOCAL_CONTRAST_R   = 45.0
    LOCAL_CONTRAST_A   = 0.20

    # ── 1. Window mask ────────────────────────────────────────────────────────
    lum_raw = 0.299 * img[:, :, 2] + 0.587 * img[:, :, 1] + 0.114 * img[:, :, 0]
    b_chan, g_chan, r_chan = img[:, :, 0], img[:, :, 1], img[:, :, 2]

    lum_win  = np.clip((lum_raw - 0.85) / (1.0 - 0.85), 0, 1) ** 1.2
    blue_dom = np.clip((b_chan - np.maximum(r_chan, g_chan) + 0.05) / 0.12, 0, 1)
    blue_dom = blue_dom * (lum_raw > 0.50).astype(np.float32)
    win_raw  = np.clip(lum_win * 0.7 + blue_dom * 0.3, 0, 1)

    win_protect  = cv2.GaussianBlur(win_raw.astype(np.float32), (0, 0), sigmaX=10, sigmaY=10)
    win_protect  = np.clip(win_protect, 0, 1)
    win_protect3 = win_protect[:, :, np.newaxis]
    interior3    = 1.0 - win_protect3

    # ── 2. Gentle S-curve on interior ────────────────────────────────────────
    img_curved = apply_tone_curve(img, SCURVE_IN, SCURVE_OUT)
    img = img_curved * interior3 + img * win_protect3
    img = np.clip(img, 0, 1)

    # ── 3. Large-radius unsharp mask on L (3D pop) ───────────────────────────
    img_u8 = (img * 255).astype(np.uint8)
    img_u8 = local_contrast_enhance(img_u8, radius=LOCAL_CONTRAST_R, amount=LOCAL_CONTRAST_A)
    img    = img_u8.astype(np.float32) / 255.0

    # ── 4. Window highlight pull ──────────────────────────────────────────────
    hi_mask  = np.clip((lum_raw - HI_START_RAW) / (1.0 - HI_START_RAW), 0, 1) ** 1.5
    hi_mask3 = hi_mask[:, :, np.newaxis]
    img = img - hi_mask3 * np.clip(img - HI_CAP, 0, None) * HI_STRENGTH
    img = np.clip(img, 0, 1)

    # ── 5. Wall / ceiling zone ────────────────────────────────────────────────
    lum2 = 0.299 * img[:, :, 2] + 0.587 * img[:, :, 1] + 0.114 * img[:, :, 0]
    wall_mask = np.clip((lum2 - WALL_LUM_LOW) / (WALL_LUM_HIGH - WALL_LUM_LOW), 0, 1)
    wall_mask = wall_mask * (1.0 - win_protect)
    wall_mask = cv2.GaussianBlur(wall_mask.astype(np.float32), (0, 0), sigmaX=8, sigmaY=8)
    wall_mask = np.clip(wall_mask, 0, 1)
    wall_mask3 = wall_mask[:, :, np.newaxis]

    # Desaturate
    img_u8_tmp = (img * 255).astype(np.uint8)
    hsv_tmp = cv2.cvtColor(img_u8_tmp, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv_tmp[:, :, 1] = hsv_tmp[:, :, 1] * (1.0 - wall_mask * WALL_DESAT)
    img = cv2.cvtColor(np.clip(hsv_tmp, 0, 255).astype(np.uint8), cv2.COLOR_HSV2BGR).astype(np.float32) / 255.0

    # Grey-world cast removal (tight ±8% cap)
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

    # Brightness push
    img = img + wall_mask3 * WALL_BRIGHT * (1.0 - img)
    img = np.clip(img, 0, 1)

    # ── 6. Shadow fill ────────────────────────────────────────────────────────
    lum3 = 0.299 * img[:, :, 2] + 0.587 * img[:, :, 1] + 0.114 * img[:, :, 0]
    fill_mask  = np.clip(1.0 - lum3 / FILL_CUTOFF, 0, 1) ** 1.5
    fill_mask3 = fill_mask[:, :, np.newaxis]
    img = img + fill_mask3 * FILL_STRENGTH * (1.0 - img) * interior3
    img = np.clip(img, 0, 1)

    # ── 7. Colour grade ───────────────────────────────────────────────────────
    img[:, :, 2] = np.clip(img[:, :, 2] * R_MULT, 0, 1)
    img[:, :, 1] = np.clip(img[:, :, 1] * G_MULT, 0, 1)
    img[:, :, 0] = np.clip(img[:, :, 0] * B_MULT, 0, 1)

    # ── 8. Vibrance ───────────────────────────────────────────────────────────
    img_u8 = (img * 255).astype(np.uint8)
    hsv    = cv2.cvtColor(img_u8, cv2.COLOR_BGR2HSV).astype(np.float32)
    sat_norm  = hsv[:, :, 1] / 255.0
    vib_zone  = np.clip(1.0 - wall_mask, 0, 1)
    vib_boost = VIBRANCE * (1.0 - sat_norm) ** 1.5 * vib_zone
    hsv[:, :, 1] = np.clip(hsv[:, :, 1] + vib_boost, 0, 255)
    img = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR).astype(np.float32) / 255.0

    # ── 9. Sharpen ────────────────────────────────────────────────────────────
    img_u8  = (img * 255).astype(np.uint8)
    pil_img = Image.fromarray(cv2.cvtColor(img_u8, cv2.COLOR_BGR2RGB))
    blurred = pil_img.filter(ImageFilter.GaussianBlur(radius=SHARPEN_RADIUS))
    orig_f  = np.array(pil_img).astype(np.float32)
    blur_f  = np.array(blurred).astype(np.float32)
    sharpened  = np.clip(orig_f + SHARPEN_AMT * (orig_f - blur_f), 0, 255).astype(np.uint8)
    result_bgr = cv2.cvtColor(sharpened, cv2.COLOR_RGB2BGR)

    mean_final = cv2.cvtColor(result_bgr, cv2.COLOR_BGR2GRAY).mean()
    print(f"AutoHDR finish applied. Final mean brightness: {mean_final:.1f}/255")
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
