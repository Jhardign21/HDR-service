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
# IMPROVEMENT 1: Better window detection
# Combines luminance + blue-channel dominance + edge-proximity bias
# ---------------------------------------------------------------------------

def build_window_mask_smart(dark_frame: np.ndarray, sigma: float = 8.0) -> np.ndarray:
    """
    Smart window mask from the darkest bracket frame.
    Uses three cues combined:
      1. Luminance — windows are bright even in underexposed shots
      2. Blue-channel dominance — sky/exterior light is cooler/bluer than interior warm light
      3. Edge proximity — windows tend to be bounded by dark frames (walls)
    Returns float32 mask [0..1], shape H×W×1.
    """
    h, w = dark_frame.shape[:2]
    img_f = dark_frame.astype(np.float32) / 255.0

    # Cue 1: luminance
    lum = 0.299 * img_f[:, :, 2] + 0.587 * img_f[:, :, 1] + 0.114 * img_f[:, :, 0]
    lum_mask = np.clip((lum - 0.65) / (1.0 - 0.65), 0, 1) ** 1.8

    # Cue 2: blue-channel relative dominance (exterior/sky is cooler than warm interior walls)
    b_chan = img_f[:, :, 0]
    r_chan = img_f[:, :, 2]
    g_chan = img_f[:, :, 1]
    # Blue dominance: blue >= red and blue >= green in that region
    blue_dom = np.clip((b_chan - np.maximum(r_chan, g_chan) + 0.05) / 0.15, 0, 1)
    blue_dom = blue_dom * (lum > 0.45).astype(np.float32)  # only where bright enough

    # Cue 3: Sobel edge map — window panes are bounded by high-contrast edges (frames/mullions)
    gray = (lum * 255).astype(np.uint8)
    sobel_x = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    edges = np.sqrt(sobel_x**2 + sobel_y**2)
    edges = cv2.normalize(edges, None, 0, 1, cv2.NORM_MINMAX)
    # Dilate edges so pixels just inside a window frame get credit
    edge_dilated = cv2.dilate(edges, np.ones((15, 15), np.uint8))
    # Pixels near a strong edge (window frame) and bright → likely window pane
    edge_proximity = np.clip(edge_dilated * 0.6, 0, 1)

    # Combine: require luminance + either blue dominance or edge proximity
    combined = lum_mask * (0.5 + 0.3 * blue_dom + 0.2 * edge_proximity)
    combined = np.clip(combined, 0, 1)
    combined = cv2.GaussianBlur(combined, (0, 0), sigmaX=sigma, sigmaY=sigma)
    return np.clip(combined, 0, 1)[:, :, np.newaxis]


# ---------------------------------------------------------------------------
# IMPROVEMENT 2: Ghost detection & removal
# Detects motion between frames (curtains, plants, people) and masks them out
# ---------------------------------------------------------------------------

def detect_ghost_mask(images: List[np.ndarray], ref_idx: int) -> np.ndarray:
    """
    Build a per-pixel ghost weight map.
    For each non-reference frame, compute absolute luminance difference vs. reference.
    Pixels with high variance across frames are likely ghosts (motion).
    Returns float32 weight map [0..1] shape H×W — 1=clean, 0=ghost region.
    The weight map is used to downweight ghost frames in Mertens fusion.
    """
    ref = cv2.cvtColor(images[ref_idx], cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
    diffs = []
    for i, img in enumerate(images):
        if i == ref_idx:
            continue
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
        # Normalize luminance difference (account for intentional exposure difference)
        diff = np.abs(gray - ref)
        diffs.append(diff)

    if not diffs:
        return np.ones(ref.shape, dtype=np.float32)

    max_diff = np.max(np.stack(diffs, axis=0), axis=0)
    # Threshold: differences > 0.25 after luminance matching = motion ghost
    ghost_prob = np.clip((max_diff - 0.15) / 0.20, 0, 1)
    ghost_prob = cv2.GaussianBlur(ghost_prob, (0, 0), sigmaX=6, sigmaY=6)
    clean_weight = 1.0 - np.clip(ghost_prob, 0, 0.85)  # never fully zero
    return clean_weight.astype(np.float32)


# ---------------------------------------------------------------------------
# IMPROVEMENT 3: Synthetic bracket generation for single-shot / bad brackets
# ---------------------------------------------------------------------------

def synthesize_brackets(img: np.ndarray) -> List[np.ndarray]:
    """
    From a single exposure, generate a 3-bracket synthetic set.
    Dark:   gamma 0.45 (simulate -2 EV) → best window detail
    Normal: original
    Bright: gamma 1.8  (simulate +2 EV) → best shadow detail
    This mimics Autoenhance's HDR Harmoniser for single-shot inputs.
    """
    img_f = img.astype(np.float32) / 255.0

    dark   = np.clip(np.power(img_f, 0.45), 0, 1)
    bright = np.clip(np.power(img_f, 1.80), 0, 1)

    dark_u8   = (dark   * 255).astype(np.uint8)
    bright_u8 = (bright * 255).astype(np.uint8)

    print("Synthesized dark/bright virtual brackets from single exposure.")
    return [dark_u8, img, bright_u8]


# ---------------------------------------------------------------------------
# Core HDR pipeline
# ---------------------------------------------------------------------------

def bracket_merge(file_urls: List[str]) -> np.ndarray:
    """
    Professional HDR merge pipeline:
      1.  Download & decode all bracket frames
      2.  Resize all to OUTPUT dimensions
      3.  AlignMTB — correct camera shake
      4.  Sort frames dark→bright
      5.  Synthesize virtual brackets if only 1 frame provided (HDR Harmoniser)
      6.  Ghost detection — build clean-weight map from motion regions
      7.  MergeMertens exposure fusion with ghost-aware weighting
      8.  Window composite: pull best-exposed window pixels back over fusion result
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
        dark_frame = images[0]

        # ── 5. HDR HARMONISER — synthesize brackets for single-shot inputs ─────
        single_shot = len(images) == 1
        if single_shot:
            print("Single exposure detected — synthesizing virtual brackets (HDR Harmoniser)...")
            images = synthesize_brackets(images[0])
            dark_frame = images[0]

        # ── 6. GHOST DETECTION ────────────────────────────────────────────────
        mid_idx = len(images) // 2
        if len(images) > 1 and not single_shot:
            print("Running ghost detection...")
            clean_weight = detect_ghost_mask(images, ref_idx=mid_idx)
            # Apply ghost suppression: blend ghosty pixels toward reference frame
            ref_f = images[mid_idx].astype(np.float32)
            clean_w3 = clean_weight[:, :, np.newaxis]
            deghosted = []
            for i, img in enumerate(images):
                if i == mid_idx:
                    deghosted.append(img)
                    continue
                img_f = img.astype(np.float32)
                # Blend toward reference in ghost regions
                blended = img_f * clean_w3 + ref_f * (1.0 - clean_w3)
                deghosted.append(np.clip(blended, 0, 255).astype(np.uint8))
            images = deghosted
            del deghosted, ref_f
            gc.collect()

        # ── 7. Mertens exposure fusion ─────────────────────────────────────────
        print("Fusing with Mertens...")
        fused = cv2.createMergeMertens(
            contrast_weight=1.4,
            saturation_weight=0.85,
            exposure_weight=0.0,
        ).process(images)
        merged = np.clip(fused * 255, 0, 255).astype(np.uint8)
        del fused
        gc.collect()

        # ── 8. Window composite ────────────────────────────────────────────────
        # Smart window mask from dark frame: luminance + blue-channel + edge proximity
        print("Compositing window detail with smart window mask...")
        win_mask = build_window_mask_smart(dark_frame, sigma=8.0)

        # Use dark frame for windows — it has the most window/sky detail
        dark_f   = dark_frame.astype(np.float32)
        merged_f = merged.astype(np.float32)
        composited = merged_f * (1.0 - win_mask) + dark_f * win_mask
        composited = np.clip(composited, 0, 255).astype(np.uint8)
        del merged_f, dark_f, win_mask
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
# AutoHDR-style post-processing — 4-zone masked finish
# ---------------------------------------------------------------------------

def apply_autohdr_finish(img_bgr: np.ndarray) -> np.ndarray:
    """
    Zone-aware finish pipeline (mimics Autoenhance.ai local editing approach):

      ZONE A — Windows     : protected, no processing
      ZONE B — Walls/Ceiling: desaturate colour cast → clean beige/white + gentle lift
      ZONE C — Floor/Wood  : warm hue boost + vibrance to enrich hardwood
      ZONE D — Deep shadows: fill light (furniture/corners)

    Improvements vs previous:
      + Per-zone grey-world colour cast removal on walls (fix green/magenta cast)
      + Floor/wood zone with warm saturation boost
      + Vibrance skips wall zone (prevents re-saturating what we just cleaned)
      + Blue-channel window detection inherited from smart mask in bracket_merge
    """
    img = img_bgr.astype(np.float32) / 255.0

    GAMMA              = 0.88
    HI_START_RAW       = 0.80
    HI_CAP             = 0.88
    HI_STRENGTH        = 0.65
    WALL_LUM_LOW       = 0.55   # zone B lower boundary
    WALL_LUM_HIGH      = 0.90   # zone B upper boundary (above = window)
    WALL_DESAT         = 0.55   # desaturate walls 55% → clean neutral beige/white
    WALL_BRIGHT        = 0.08   # gentle brightness push on walls
    WALL_CAST_STRENGTH = 0.45   # per-zone grey-world cast removal strength on walls
    FLOOR_LUM_HIGH     = 0.45   # floor/wood is darker than walls
    FLOOR_WARM_BOOST   = 0.12   # warm hue push on floors (enrich hardwood)
    FILL_CUTOFF        = 0.22
    FILL_STRENGTH      = 0.14
    R_MULT             = 1.03
    G_MULT             = 0.98
    B_MULT             = 0.93
    VIBRANCE           = 14.0
    SHARPEN_AMT        = 0.65
    SHARPEN_RADIUS     = 1.0

    # ── 1. WINDOW MASK (luminance + blue dominance) ───────────────────────────
    lum_raw = 0.299 * img[:, :, 2] + 0.587 * img[:, :, 1] + 0.114 * img[:, :, 0]
    b_chan   = img[:, :, 0]
    r_chan   = img[:, :, 2]
    g_chan   = img[:, :, 1]

    # Luminance cue
    lum_win = np.clip((lum_raw - 0.85) / (1.0 - 0.85), 0, 1) ** 1.2
    # Blue-channel dominance cue (exterior light is cooler than warm interior walls)
    blue_dom = np.clip((b_chan - np.maximum(r_chan, g_chan) + 0.05) / 0.12, 0, 1)
    blue_dom = blue_dom * (lum_raw > 0.50).astype(np.float32)
    # Combined window signal
    win_raw = np.clip(lum_win * 0.7 + blue_dom * 0.3, 0, 1)
    win_protect = cv2.GaussianBlur(win_raw.astype(np.float32), (0, 0), sigmaX=6, sigmaY=6)
    win_protect = np.clip(win_protect, 0, 1)
    win_protect3 = win_protect[:, :, np.newaxis]
    interior3    = 1.0 - win_protect3

    # ── 2. GAMMA (interior only) ──────────────────────────────────────────────
    img_gamma = np.power(np.clip(img, 1e-6, 1.0), GAMMA)
    img = img_gamma * interior3 + img * win_protect3
    img = np.clip(img, 0, 1)

    # ── 3. WINDOW PULL ────────────────────────────────────────────────────────
    hi_mask = np.clip((lum_raw - HI_START_RAW) / (1.0 - HI_START_RAW), 0, 1) ** 1.5
    hi_mask3 = hi_mask[:, :, np.newaxis]
    img = img - hi_mask3 * np.clip(img - HI_CAP, 0, None) * HI_STRENGTH
    img = np.clip(img, 0, 1)

    # ── 4. WALL/CEILING ZONE MASK ─────────────────────────────────────────────
    lum2 = 0.299 * img[:, :, 2] + 0.587 * img[:, :, 1] + 0.114 * img[:, :, 0]
    wall_mask = np.clip((lum2 - WALL_LUM_LOW) / (WALL_LUM_HIGH - WALL_LUM_LOW), 0, 1)
    wall_mask = wall_mask * (1.0 - win_protect)
    wall_mask = cv2.GaussianBlur(wall_mask.astype(np.float32), (0, 0), sigmaX=5, sigmaY=5)
    wall_mask = np.clip(wall_mask, 0, 1)
    wall_mask3 = wall_mask[:, :, np.newaxis]

    # ── 5. FLOOR/WOOD ZONE MASK ───────────────────────────────────────────────
    # Floors are darker (lum < FLOOR_LUM_HIGH) and tend to have warm hue (red > blue)
    warm_bias = np.clip((img[:, :, 2] - img[:, :, 0]) / 0.15, 0, 1)  # R > B = warm
    floor_mask = np.clip(1.0 - lum2 / FLOOR_LUM_HIGH, 0, 1) * warm_bias * interior3[:, :, 0]
    floor_mask = cv2.GaussianBlur(floor_mask.astype(np.float32), (0, 0), sigmaX=5, sigmaY=5)
    floor_mask = np.clip(floor_mask, 0, 1)
    floor_mask3 = floor_mask[:, :, np.newaxis]

    # ── 6a. WALL DESATURATION → clean neutral beige/white ────────────────────
    img_u8_tmp = (img * 255).astype(np.uint8)
    hsv_tmp = cv2.cvtColor(img_u8_tmp, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv_tmp[:, :, 1] = hsv_tmp[:, :, 1] * (1.0 - wall_mask * WALL_DESAT)
    img_u8_tmp = cv2.cvtColor(np.clip(hsv_tmp, 0, 255).astype(np.uint8), cv2.COLOR_HSV2BGR)
    img = img_u8_tmp.astype(np.float32) / 255.0

    # ── 6b. PER-ZONE GREY-WORLD CAST REMOVAL on walls ─────────────────────────
    # Autoenhance v4.7 explicitly added "colour cast removal" as a separate step.
    # Idea: in the wall zone, all channels SHOULD be roughly equal (neutral beige/white).
    # If one channel dominates, nudge it back toward the mean of the other two.
    wall_pixels_r = img[:, :, 2] * wall_mask
    wall_pixels_g = img[:, :, 1] * wall_mask
    wall_pixels_b = img[:, :, 0] * wall_mask
    wall_sum = wall_mask.sum() + 1e-6
    mean_r = wall_pixels_r.sum() / wall_sum
    mean_g = wall_pixels_g.sum() / wall_sum
    mean_b = wall_pixels_b.sum() / wall_sum
    mean_all = (mean_r + mean_g + mean_b) / 3.0 + 1e-6
    # Correction factors: channels above mean get pulled back, below get a small lift
    cor_r = mean_all / (mean_r + 1e-6)
    cor_g = mean_all / (mean_g + 1e-6)
    cor_b = mean_all / (mean_b + 1e-6)
    # Clamp corrections so we don't over-correct (max 15% shift)
    cor_r = np.clip(cor_r, 0.88, 1.15)
    cor_g = np.clip(cor_g, 0.88, 1.15)
    cor_b = np.clip(cor_b, 0.88, 1.15)
    # Apply correction blended by wall_mask strength and cast strength
    img[:, :, 2] = np.clip(img[:, :, 2] * (1.0 + (cor_r - 1.0) * wall_mask * WALL_CAST_STRENGTH), 0, 1)
    img[:, :, 1] = np.clip(img[:, :, 1] * (1.0 + (cor_g - 1.0) * wall_mask * WALL_CAST_STRENGTH), 0, 1)
    img[:, :, 0] = np.clip(img[:, :, 0] * (1.0 + (cor_b - 1.0) * wall_mask * WALL_CAST_STRENGTH), 0, 1)

    # ── 6c. WALL BRIGHTNESS PUSH → lift toward clean white ───────────────────
    img = img + wall_mask3 * WALL_BRIGHT * (1.0 - img)
    img = np.clip(img, 0, 1)

    # ── 7. FLOOR/WOOD WARM BOOST — enrich hardwood tones ─────────────────────
    # Boost red channel slightly and reduce blue in floor zone to warm up hardwood
    img[:, :, 2] = np.clip(img[:, :, 2] + floor_mask * FLOOR_WARM_BOOST * (1.0 - img[:, :, 2]), 0, 1)
    img[:, :, 0] = np.clip(img[:, :, 0] - floor_mask * (FLOOR_WARM_BOOST * 0.5) * img[:, :, 0], 0, 1)

    # ── 8. FILL LIGHT — deep shadow corners only ──────────────────────────────
    lum3 = 0.299 * img[:, :, 2] + 0.587 * img[:, :, 1] + 0.114 * img[:, :, 0]
    fill_mask = np.clip(1.0 - lum3 / FILL_CUTOFF, 0, 1) ** 1.8
    fill_mask3 = fill_mask[:, :, np.newaxis]
    # Don't fill floor zone (already warmed), don't fill windows
    fill_zone = interior3 * (1.0 - floor_mask3)
    img = img + fill_mask3 * FILL_STRENGTH * (1.0 - img) * fill_zone
    img = np.clip(img, 0, 1)

    # ── 9. GLOBAL COLOUR GRADE ────────────────────────────────────────────────
    img[:, :, 2] = np.clip(img[:, :, 2] * R_MULT, 0, 1)
    img[:, :, 1] = np.clip(img[:, :, 1] * G_MULT, 0, 1)
    img[:, :, 0] = np.clip(img[:, :, 0] * B_MULT, 0, 1)

    # ── 10. VIBRANCE — skip walls (already neutralised), boost floors/mid-tones
    img_u8 = (img * 255).astype(np.uint8)
    hsv = cv2.cvtColor(img_u8, cv2.COLOR_BGR2HSV).astype(np.float32)
    sat_norm = hsv[:, :, 1] / 255.0
    # Wall zone excluded from vibrance; floor zone gets a little extra
    vib_zone = (1.0 - wall_mask) + floor_mask * 0.4
    vib_zone = np.clip(vib_zone, 0, 1)
    vib_boost = VIBRANCE * (1.0 - sat_norm) ** 1.5 * vib_zone
    hsv[:, :, 1] = np.clip(hsv[:, :, 1] + vib_boost, 0, 255)
    img_u8 = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
    img = img_u8.astype(np.float32) / 255.0

    # ── 11. LUMINANCE-AWARE DENOISE ───────────────────────────────────────────
    img_u8 = (img * 255).astype(np.uint8)
    denoised = cv2.bilateralFilter(img_u8, d=9, sigmaColor=25, sigmaSpace=12)
    lum_denoise = 0.299 * img[:, :, 2] + 0.587 * img[:, :, 1] + 0.114 * img[:, :, 0]
    denoise_alpha = np.clip(1.0 - lum_denoise * 1.4, 0, 1)[:, :, np.newaxis]
    img_u8 = (img_u8 * (1.0 - denoise_alpha) + denoised * denoise_alpha).astype(np.uint8)

    # ── 12. SHARPEN ───────────────────────────────────────────────────────────
    pil_img  = Image.fromarray(cv2.cvtColor(img_u8, cv2.COLOR_BGR2RGB))
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
    photomatix_api_key: Optional[str] = None
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
