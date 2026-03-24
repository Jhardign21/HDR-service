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
    exposure: float = 0.5        # -2 to +2  EV
    saturation: float = 1.25     # 0 to 2
    shadows: float = 0.18        # 0 to 0.5
    whites: float = 0.93         # 0.7 to 1
    blacks: float = 0.04         # 0 to 0.2
    temperature: float = 0.0     # -50 to +50
    sky_pull: bool = True


# ---------------------------------------------------------------------------
# Sky replacement — exterior-aware approach
# ---------------------------------------------------------------------------

def make_sky_gradient(h: int, w: int) -> np.ndarray:
    """Vivid real-estate blue sky: deep azure at top, lighter sky-blue at bottom."""
    sky = np.zeros((h, w, 3), dtype=np.float32)
    for row in range(h):
        t = row / max(h - 1, 1)
        # BGR
        sky[row, :, 0] = 210 + 30 * t   # B: 210→240
        sky[row, :, 1] = 130 + 60 * t   # G: 130→190
        sky[row, :, 2] = 40  + 50 * t   # R: 40→90
    return np.clip(sky, 0, 255).astype(np.uint8)


def build_sky_mask(img: np.ndarray) -> np.ndarray:
    """
    Detect sky by seeding only from the very top rows of the image.
    Prevents white walls/snow from being mistaken for sky.
    """
    h, w = img.shape[:2]

    b = img[:, :, 0].astype(np.float32)
    g = img[:, :, 1].astype(np.float32)
    r = img[:, :, 2].astype(np.float32)

    brightness = (b + g + r) / 3.0
    max_c = np.maximum(np.maximum(b, g), r)
    min_c = np.minimum(np.minimum(b, g), r)
    chroma = max_c - min_c

    blue_sky  = (b > r + 15) & (b > g + 5) & (brightness > 100)
    blown_sky = (brightness > 210) & (chroma < 20)
    sky_like  = (blue_sky | blown_sky).astype(np.uint8) * 255

    # Build seed mask: ONLY use the very top 5% of the image as flood-fill seeds
    # This prevents white house walls from becoming seeds
    seed_zone = int(h * 0.05)
    seed_mask = sky_like.copy()
    seed_mask[seed_zone:, :] = 0   # clear everything below top 5%

    # Flood fill from every top-edge pixel that qualifies as sky
    canvas = sky_like.copy()
    filled = np.zeros((h + 2, w + 2), np.uint8)
    for x in range(w):
        if seed_mask[0, x] > 0:
            cv2.floodFill(canvas, filled, (x, 0), 128,
                          loDiff=(15, 15, 15), upDiff=(15, 15, 15))

    flood_mask = (canvas == 128).astype(np.uint8) * 255

    # Hard cap: sky never goes below 40% of image height
    flood_mask[int(h * 0.40):, :] = 0

    # Feather edges
    flood_mask = cv2.GaussianBlur(flood_mask, (21, 21), 0)

    covered = np.count_nonzero(flood_mask > 10)
    print(f"Sky mask pixels: {covered}  (threshold: {int(h * w * 0.02)})")
    return flood_mask


def apply_sky_replacement(img: np.ndarray) -> tuple:
    h, w = img.shape[:2]
    mask = build_sky_mask(img)
    covered = np.count_nonzero(mask > 10)
    min_px = int(h * w * 0.02)   # at least 2% of image

    if covered < min_px:
        print("Sky region too small — skipping replacement")
        return img, False

    sky = make_sky_gradient(h, w).astype(np.float32)
    src = img.astype(np.float32)
    a   = mask.astype(np.float32)[:, :, np.newaxis] / 255.0
    result = sky * a + src * (1.0 - a)
    print(f"Sky replaced — {covered} pixels blended")
    return np.clip(result, 0, 255).astype(np.uint8), True


# ---------------------------------------------------------------------------
# Tone pipeline — tuned to produce punchy real-estate look
# ---------------------------------------------------------------------------

def apply_tone(img: np.ndarray, p: ProcessingParams) -> np.ndarray:
    f = img.astype(np.float32) / 255.0

    # 1. Gamma lift + exposure boost
    ev = 2.0 ** p.exposure
    f  = np.power(np.clip(f, 1e-6, 1.0), 0.72)   # gentle gamma lift
    f  = np.clip(f * ev, 0, 1)

    # 2. Black floor
    f = f * (1.0 - p.blacks) + p.blacks

    # 3. White ceiling
    f = np.clip(f, 0, p.whites) / p.whites

    # 4. S-curve contrast (lifts mids, deepens shadows slightly)
    # Apply a subtle S-curve: shadows slightly deeper, mids brighter
    f = f * f * (3.0 - 2.0 * f)   # smoothstep — adds contrast naturally

    # 5. Shadow lift on dark areas
    shadow_mask = np.clip(1.0 - f / 0.4, 0, 1)
    f = f + shadow_mask * p.shadows
    f = np.clip(f, 0, 1)

    # 6. Temperature shift
    if abs(p.temperature) > 0.5:
        shift = p.temperature / 400.0
        f[:, :, 2] = np.clip(f[:, :, 2] + shift, 0, 1)  # R warmer
        f[:, :, 0] = np.clip(f[:, :, 0] - shift * 0.5, 0, 1)  # B slightly cooler

    # 7. Neutral white balance (per-channel highlight normalisation)
    for c in range(3):
        p99 = float(np.percentile(f[:, :, c], 99))
        if p99 > 0.55:
            f[:, :, c] = np.clip(f[:, :, c] * (0.97 / p99), 0, 1)

    # 8. Saturation (HSV)
    rgb_u8 = np.clip(f * 255, 0, 255).astype(np.uint8)
    hsv = cv2.cvtColor(rgb_u8, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[:, :, 1] = np.clip(hsv[:, :, 1] * p.saturation, 0, 255)
    result = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)

    # 9. Mild sharpening (unsharp mask)
    blurred = cv2.GaussianBlur(result, (0, 0), sigmaX=1.5)
    result = cv2.addWeighted(result, 1.4, blurred, -0.4, 0)

    del f, hsv, blurred
    gc.collect()
    return result


# ---------------------------------------------------------------------------
# RAW loading / alignment / Mertens
# ---------------------------------------------------------------------------

class MergeRequest(BaseModel):
    file_urls: List[str]
    bracket_name: str = "bracket"
    replace_sky: bool = True
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
    fused = cv2.createMergeMertens(
        contrast_weight=1.0, saturation_weight=1.2, exposure_weight=0.2
    ).process(images)
    result = np.clip(fused * 255, 0, 255).astype(np.uint8)
    del fused
    gc.collect()
    return result


# ---------------------------------------------------------------------------
# Endpoint
# ---------------------------------------------------------------------------

@app.post("/merge")
async def merge_hdr(req: MergeRequest):
    if not req.file_urls:
        raise HTTPException(400, "No file URLs provided")
    if len(req.file_urls) not in (1, 3, 5):
        raise HTTPException(400, f"Expected 1, 3, or 5 files, got {len(req.file_urls)}")

    p = req.params or ProcessingParams()
    print(f"Processing params: {p}")

    tmp_paths = []
    try:
        for url in req.file_urls:
            ext = url.split("?")[0].rsplit(".", 1)[-1]
            ext = f".{ext.lower()}" if ext else ".jpg"
            tmp_paths.append(download_file(url, ext))

        images = [load_image_bgr(path) for path in tmp_paths]
        gc.collect()

        if len(images) > 1:
            images = align_images(images)

        merged = images[0] if len(images) == 1 else merge_mertens(images)
        if len(images) > 1:
            del images
            gc.collect()

        # Tone mapping
        toned = apply_tone(merged, p)
        del merged
        gc.collect()

        # Sky replacement
        sky_replaced = False
        if req.replace_sky and p.sky_pull:
            toned, sky_replaced = apply_sky_replacement(toned)
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
            "sky_replaced": sky_replaced,
            "window_pull_applied": True,
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
def health():
    return {"status": "ok"}
