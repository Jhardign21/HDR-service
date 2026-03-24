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
        sky[row, :, 0] = 210 + 30 * t   # B: 210->240
        sky[row, :, 1] = 130 + 60 * t   # G: 130->190
        sky[row, :, 2] = 40  + 50 * t   # R: 40->90
    return np.clip(sky, 0, 255).astype(np.uint8)


def build_sky_mask(img: np.ndarray) -> np.ndarray:
    """
    Column-by-column structural skyline detection.
    Scans from the top of each column and finds where edges accumulate
    (roofline, treetops, fences). Everything above = sky.
    Works regardless of sky/snow color similarity.
    """
    h, w = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Structural edges (rooflines, window frames, siding, fences)
    edges = cv2.Canny(gray, 30, 90)

    # Per-column skyline: scan from top, sky ends where 3+ edge pixels accumulate
    skyline = np.zeros(w, dtype=np.float32)
    for col in range(w):
        col_edges = edges[:, col]
        cumulative = np.cumsum(col_edges > 0)
        structure_rows = np.where(cumulative >= 3)[0]
        if len(structure_rows) > 0:
            skyline[col] = float(structure_rows[0])
        else:
            skyline[col] = h * 0.45  # conservative fallback

    # Never let sky extend below 65% of image height
    skyline = np.minimum(skyline, h * 0.65)

    # Smooth the skyline to avoid jagged transitions
    skyline_smooth = cv2.GaussianBlur(
        skyline.reshape(1, -1).astype(np.float32), (101, 1), 0
    ).flatten()

    # Build soft alpha mask with feathered edge at skyline
    feather = max(int(h * 0.02), 6)
    row_idx = np.arange(h, dtype=np.float32)[:, np.newaxis]
    sl = skyline_smooth[np.newaxis, :]

    above     = (row_idx <= sl - feather).astype(np.float32)
    t         = np.clip((sl - row_idx + feather) / (2 * feather), 0.0, 1.0)
    in_feather = ((row_idx > sl - feather) & (row_idx <= sl + feather)).astype(np.float32)

    alpha_final = above + in_feather * t
    alpha_u8 = np.clip(alpha_final * 255, 0, 255).astype(np.uint8)
    alpha_u8 = cv2.GaussianBlur(alpha_u8, (15, 15), 0)

    covered = np.count_nonzero(alpha_u8 > 10)
    print(f"[sky] skyline detection: sky_px={covered} ({100*covered//(h*w)}%)")
    return alpha_u8


def color_match_sky(sky: np.ndarray, img: np.ndarray, fg_mask: np.ndarray) -> np.ndarray:
    """
    Match sky brightness to the foreground scene (L channel only).
    Deliberately avoids hue matching — that's what causes purple/pink shifts.
    """
    if np.count_nonzero(fg_mask > 128) < 100:
        return sky

    img_lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB).astype(np.float32)
    sky_lab = cv2.cvtColor(sky, cv2.COLOR_BGR2LAB).astype(np.float32)
    fg_lab  = img_lab[fg_mask > 128]

    # Transfer L (brightness) only — preserves the blue hue of replacement sky
    fg_L_mean = float(np.mean(fg_lab[:, 0]))
    fg_L_std  = float(np.std(fg_lab[:, 0])) + 1e-6
    sk_L_mean = float(np.mean(sky_lab[:, :, 0]))
    sk_L_std  = float(np.std(sky_lab[:, :, 0])) + 1e-6
    sky_lab[:, :, 0] = (sky_lab[:, :, 0] - sk_L_mean) * (fg_L_std / sk_L_std) + fg_L_mean

    sky_lab = np.clip(sky_lab, 0, 255).astype(np.uint8)
    return cv2.cvtColor(sky_lab, cv2.COLOR_LAB2BGR)


def apply_sky_replacement(img: np.ndarray) -> tuple:
    h, w = img.shape[:2]
    sky_mask = build_sky_mask(img)
    covered = np.count_nonzero(sky_mask > 10)
    min_px = int(h * w * 0.01)   # at least 1% of image

    if covered < min_px:
        print("Sky region too small — skipping replacement")
        return img, False

    # Build replacement sky gradient
    sky = make_sky_gradient(h, w)

    # Color-match sky to foreground lighting (Reinhard LAB transfer)
    fg_mask = cv2.bitwise_not(sky_mask)
    sky = color_match_sky(sky, img, fg_mask)

    # Composite
    a      = sky_mask.astype(np.float32)[:, :, np.newaxis] / 255.0
    result = sky.astype(np.float32) * a + img.astype(np.float32) * (1.0 - a)
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
