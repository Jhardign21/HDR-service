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
import time

app = FastAPI(title="HDR Merge Service")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

OUTPUT_WIDTH  = 2048
OUTPUT_HEIGHT = 1536

PHOTOMATIX_API = "https://api.hdrsoft.com"


# ---------------------------------------------------------------------------
# Photomatix API — HDR merge via cloud
# ---------------------------------------------------------------------------

def photomatix_merge(file_urls: List[str], api_key: str) -> np.ndarray:
    headers = {"x-pm-token": api_key}

    print("Photomatix: creating HDR engine...")
    r = requests.post(
        f"{PHOTOMATIX_API}/hdrengines",
        headers=headers,
        params={
            "type": "multi",
            "alignment": "yes",
            "deghosting": "on",
            "noise-reduction": "underexposed",
            "lens-correction": "yes",
            "output-bit-depth": "8",
        },
        timeout=30,
    )
    if r.status_code not in (200, 201):
        raise Exception(f"Photomatix engine creation failed ({r.status_code}): {r.text[:300]}")

    data = r.json()
    engine_uri = data.get("data", {}).get("location") or data.get("location")
    if not engine_uri:
        raise Exception(f"No engine URI returned: {data}")
    print(f"Photomatix engine: {engine_uri}")

    for i, url in enumerate(file_urls):
        base = url.split("?")[0].rsplit("/", 1)[-1]
        ext = base.rsplit(".", 1)[-1].lower() if "." in base else "jpg"
        filename = f"image_{i+1}.{ext}"
        print(f"Photomatix: adding image {filename}...")
        add_r = requests.post(
            f"{PHOTOMATIX_API}{engine_uri}/images/{filename}",
            headers=headers,
            data={"url": url},
            timeout=60,
        )
        if add_r.status_code not in (200, 201):
            raise Exception(f"Photomatix add image failed ({add_r.status_code}): {add_r.text[:300]}")

    print("Photomatix: processing with Real Estate preset...")
    process_r = requests.post(
        f"{PHOTOMATIX_API}{engine_uri}/processed/preset",
        headers=headers,
        params={
            "preset": "Real Estate",
            "output-format": "jpeg",
            "output-bit-depth": "8",
        },
        timeout=180,
    )
    if process_r.status_code not in (200, 201):
        raise Exception(f"Photomatix process failed ({process_r.status_code}): {process_r.text[:300]}")

    process_data = process_r.json()
    result_url = process_data.get("data", {}).get("location") or process_data.get("location")
    if not result_url:
        raise Exception(f"No result URL from Photomatix: {process_data}")

    print(f"Photomatix: downloading result from {result_url}...")
    for attempt in range(30):
        dl = requests.get(f"{PHOTOMATIX_API}{result_url}", headers=headers, timeout=60)
        if dl.status_code == 200:
            break
        elif dl.status_code == 202:
            print(f"Photomatix: still processing... attempt {attempt+1}")
            time.sleep(5)
        else:
            raise Exception(f"Photomatix download failed ({dl.status_code}): {dl.text[:300]}")
    else:
        raise Exception("Photomatix processing timed out after 150s")

    img_arr = np.frombuffer(dl.content, np.uint8)
    img = cv2.imdecode(img_arr, cv2.IMREAD_COLOR)
    if img is None:
        raise Exception("Could not decode Photomatix result image")

    print(f"Photomatix merge done: {img.shape[1]}x{img.shape[0]}")
    return img


# ---------------------------------------------------------------------------
# AutoHDR-style post-processing
# ---------------------------------------------------------------------------

def apply_autohdr_finish(img_bgr: np.ndarray) -> np.ndarray:
    """
    AutoHDR/Autoenhance.ai "Classic" pipeline — window-aware masking.

    Key principle: build a WINDOW PROTECTION MASK from the raw input luminance
    BEFORE any lifting. Every subsequent brightness operation is multiplied by
    (1 - window_protect_mask) so windows are never re-brightened after the pull.

    Pipeline:
    1.  Build window protection mask from raw luminance (pre-lift)
    2.  Pre-lift: gamma to bring interior to working brightness
    3.  Window pull: compress highlights back to target cap
    4.  Fill light: lift dark interior zones — MASKED away from windows
    5.  Whites push: clean white on ceilings/walls — MASKED away from windows
    6.  Colour grade: neutral-to-slightly-warm (suits white-walled interiors)
    7.  S-curve contrast
    8.  Vibrance
    9.  Sharpen
    """
    img = img_bgr.astype(np.float32) / 255.0

    # ── AI-CALIBRATED PARAMS ──────────────────────────────────────────────────
    GAMMA            = 0.42   # Reinhard + gamma for clean bright interior lift
    # Window pull operates on RAW pre-lift luminance — only pull back true blown glass
    HI_START_RAW     = 0.72   # raw lum threshold: pixels > 0.72 pre-lift are windows
    HI_CAP           = 0.88   # window pull target value post-processing
    HI_STRENGTH      = 0.65   # moderate pull on windows
    FILL_CUTOFF      = 0.45   # fill light touches shadows and lower midtones
    FILL_STRENGTH    = 0.38   # moderate fill to open shadows
    WHITES_START     = 0.78   # whites push starts lower — catch more grey walls/ceiling
    WHITES_STRENGTH  = 0.72   # strong whites push → clean white walls
    R_MULT           = 1.04   # warm — match the warm white in target
    G_MULT           = 1.02
    B_MULT           = 0.98   # slightly pull blue for warm interior
    VIBRANCE         = 22.0
    SHARPEN_AMT      = 0.45
    SHARPEN_RADIUS   = 1.0

    # ── 1. WINDOW PROTECTION MASK (built from PRE-LIFT luminance) ────────────
    # Tight mask: only protect true window glass (very high luminance pixels).
    # Use a high threshold so curtains, walls near windows are NOT masked out.
    lum_raw = 0.299 * img[:, :, 2] + 0.587 * img[:, :, 1] + 0.114 * img[:, :, 0]
    WIN_THRESH = 0.78   # only pixels above this are "window glass" — tight protection
    win_protect = np.clip((lum_raw - WIN_THRESH) / (1.0 - WIN_THRESH), 0, 1) ** 0.8
    # Small blur to feather edge of window frame
    win_protect_blurred = cv2.GaussianBlur(win_protect, (0, 0), sigmaX=5, sigmaY=5)
    win_protect_blurred = np.clip(win_protect_blurred, 0, 1)
    win_protect3 = win_protect_blurred[:, :, np.newaxis]   # broadcast-ready
    interior3    = 1.0 - win_protect3                      # where we CAN lift

    # ── 2. PRE-LIFT ───────────────────────────────────────────────────────────
    # Filmic tone-lift: maps [0,1] dark input to [0,1] bright output
    # Uses a modified Reinhard-style curve: out = x / (x + c) * (1 + c)
    # c=0.12 → shadow pixels get ~4x boost, highlights compress naturally
    c = 0.08
    img = img / (img + c) * (1.0 + c)
    img = np.clip(img, 0, 1)
    # Apply gamma to lift remaining dark areas further
    img = np.power(np.clip(img, 1e-6, 1.0), GAMMA)
    img = np.clip(img, 0, 1)

    # ── 3. WINDOW PULL (uses PRE-LIFT raw lum to identify true windows) ──────
    # Use lum_raw (calculated before any lifting) so only actual window pixels
    # are pulled — not walls/curtains that got lifted to bright by the pipeline.
    hi_mask_raw = np.clip((lum_raw - HI_START_RAW) / (1.0 - HI_START_RAW), 0, 1) ** 1.5
    hi_mask3 = hi_mask_raw[:, :, np.newaxis]
    img = img - hi_mask3 * np.clip(img - HI_CAP, 0, None) * HI_STRENGTH
    img = np.clip(img, 0, 1)

    # ── 4. ZONE-AWARE FILL LIGHT (interior only — windows excluded) ───────────
    lum2 = 0.299 * img[:, :, 2] + 0.587 * img[:, :, 1] + 0.114 * img[:, :, 0]
    fill_mask = np.clip(1.0 - lum2 / FILL_CUTOFF, 0, 1) ** 1.8
    fill_mask3 = fill_mask[:, :, np.newaxis]
    # Apply fill ONLY to interior — window pixels kept as-is
    delta_fill = fill_mask3 * FILL_STRENGTH * (1.0 - img)
    img = img + delta_fill * interior3
    img = np.clip(img, 0, 1)

    # ── 5. WHITES PUSH (interior only — ceilings/walls → clean white) ─────────
    lum3 = 0.299 * img[:, :, 2] + 0.587 * img[:, :, 1] + 0.114 * img[:, :, 0]
    white_mask = np.clip((lum3 - WHITES_START) / (1.0 - WHITES_START), 0, 1) ** 2.0
    white_mask3 = white_mask[:, :, np.newaxis]
    delta_white = white_mask3 * (1.0 - img) * WHITES_STRENGTH
    img = img + delta_white * interior3
    img = np.clip(img, 0, 1)

    # ── 5b. GLOBAL BRIGHTNESS BOOST (interior only) ───────────────────────────
    # After fill+whites, push ALL interior pixels brighter with a soft curve
    # This is the "exposure +" equivalent that AutoHDR applies globally
    img = img + 0.18 * (1.0 - img) * interior3
    img = np.clip(img, 0, 1)

    # ── 6. COLOUR GRADE ──────────────────────────────────────────────────────
    img[:, :, 2] = np.clip(img[:, :, 2] * R_MULT, 0, 1)
    img[:, :, 1] = np.clip(img[:, :, 1] * G_MULT, 0, 1)
    img[:, :, 0] = np.clip(img[:, :, 0] * B_MULT, 0, 1)

    # ── 7. MIDTONE BRIGHTNESS LIFT CURVE ─────────────────────────────────────
    # Gentle lift curve that pushes midtones brighter (AutoHDR "clarity" feel)
    # without crushing shadows or clipping highlights.
    # f(x) = x + lift * sin(π*x)  → max lift at x=0.5, zero at x=0 and x=1
    img = np.clip(img + 0.08 * np.sin(np.pi * img), 0, 1)

    # ── 8. VIBRANCE ───────────────────────────────────────────────────────────
    img_u8 = (img * 255).astype(np.uint8)
    hsv = cv2.cvtColor(img_u8, cv2.COLOR_BGR2HSV).astype(np.float32)
    sat_norm = hsv[:, :, 1] / 255.0
    vib_boost = VIBRANCE * (1.0 - sat_norm) ** 1.5
    hsv[:, :, 1] = np.clip(hsv[:, :, 1] + vib_boost, 0, 255)
    img_u8 = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
    img = img_u8.astype(np.float32) / 255.0

    # ── 9. LUMINANCE-AWARE DENOISE ────────────────────────────────────────────
    # Denoise more aggressively in shadows (where lifted noise is worst),
    # and gently in highlights (where detail must be preserved).
    img_u8 = (img * 255).astype(np.uint8)
    # Fast bilateral filter: preserves edges while smoothing flat noisy areas
    denoised = cv2.bilateralFilter(img_u8, d=9, sigmaColor=25, sigmaSpace=12)
    # Blend: shadow pixels get more denoise, bright pixels keep original detail
    lum_denoise = 0.299 * img[:, :, 2] + 0.587 * img[:, :, 1] + 0.114 * img[:, :, 0]
    # alpha=1 in shadows (full denoise), alpha=0 in highlights (no denoise)
    denoise_alpha = np.clip(1.0 - lum_denoise * 1.4, 0, 1)[:, :, np.newaxis]
    img_u8 = (img_u8 * (1.0 - denoise_alpha) + denoised * denoise_alpha).astype(np.uint8)

    # ── 10. SHARPEN ───────────────────────────────────────────────────────────
    pil_img = Image.fromarray(cv2.cvtColor(img_u8, cv2.COLOR_BGR2RGB))
    blurred = pil_img.filter(ImageFilter.GaussianBlur(radius=SHARPEN_RADIUS))
    orig_f   = np.array(pil_img).astype(np.float32)
    blur_f   = np.array(blurred).astype(np.float32)
    sharpened = np.clip(orig_f + SHARPEN_AMT * (orig_f - blur_f), 0, 255).astype(np.uint8)
    result_bgr = cv2.cvtColor(sharpened, cv2.COLOR_RGB2BGR)

    print("AutoHDR window-aware finish applied.")
    return result_bgr


# ---------------------------------------------------------------------------
# Fallback: local Mertens merge (no Photomatix key)
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
        rgb = raw.postprocess(
            use_camera_wb=True,
            no_auto_bright=False,
            output_bps=8,
            half_size=False,          # full resolution — no interpolation softness
            median_filter_passes=0,
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


def local_merge(file_urls: List[str]) -> np.ndarray:
    tmp_paths = []
    try:
        for url in file_urls:
            ext = url.split("?")[0].rsplit(".", 1)[-1]
            ext = f".{ext.lower()}" if ext else ".jpg"
            tmp_paths.append(download_file(url, ext))

        images = [load_image_bgr(p) for p in tmp_paths]
        gc.collect()

        if len(images) > 1:
            fused = cv2.createMergeMertens(
                contrast_weight=1.4, saturation_weight=0.9, exposure_weight=0.2
            ).process(images)
            merged = np.clip(fused * 255, 0, 255).astype(np.uint8)
            del fused
        else:
            merged = images[0]

        gc.collect()
        return merged
    finally:
        for p in tmp_paths:
            try:
                os.unlink(p)
            except Exception:
                pass


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

    # Accept key from request body (preferred) or fall back to env var
    photomatix_key = req.photomatix_api_key or os.environ.get("PHOTOMATIX_API_KEY")

    try:
        if photomatix_key:
            print("Using Photomatix API for HDR merge...")
            merged = photomatix_merge(req.file_urls, photomatix_key)
            merged = cv2.resize(merged, (OUTPUT_WIDTH, OUTPUT_HEIGHT), interpolation=cv2.INTER_LANCZOS4)
        else:
            print("No Photomatix key — falling back to local Mertens merge...")
            merged = local_merge(req.file_urls)

        # Apply AutoHDR-style natural finish
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
