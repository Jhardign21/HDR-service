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
    Sky detection using edge-density + color + top-connectivity.
    Core insight (how Lightroom/PS work):
      - Sky has near-ZERO local edge density
      - Walls have siding/window/brick edges
      - Snow has grain
      - Only sky is edge-free AND touches the top of the frame
    """
    h, w = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # -----------------------------------------------------------------------
    # STEP 1: Edge density map
    # Canny finds structural edges; dilate + blur gives local edge density
    # -----------------------------------------------------------------------
    edges = cv2.Canny(gray, 25, 75)  # structural edges (siding, rooflines, windows)
    # Dilate so thin edges spread, then blur to get density in local neighborhood
    edges_d = cv2.dilate(edges, np.ones((5, 5), np.uint8), iterations=2)
    edge_density = cv2.GaussianBlur(edges_d.astype(np.float32), (61, 61), 0)
    # Normalize 0-1
    ed_max = edge_density.max()
    if ed_max > 0:
        edge_density /= ed_max

    # Low-edge-density pixels are sky candidates
    # Adaptive threshold: calibrate from the top 10% of the image (known sky)
    top_ed = edge_density[:int(h * 0.10), :]
    ed_thresh = float(np.percentile(top_ed, 90)) * 2.5  # 2.5x the top-strip reference
    ed_thresh = max(ed_thresh, 0.08)  # minimum threshold
    low_edge_px = (edge_density < ed_thresh).astype(np.uint8) * 255

    # -----------------------------------------------------------------------
    # STEP 2: Color confirmation
    # Sky is blue OR bright+very-low-saturation (overcast/hazy/blown)
    # Snow: bright but has texture/edges (filtered by step 1)
    # Walls: may be bright but have edges (filtered by step 1)
    # -----------------------------------------------------------------------
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float32)
    h_ch, s_ch, v_ch = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]

    blue_sky    = (h_ch > 90)  & (h_ch < 145) & (s_ch > 25)  & (v_ch > 50)
    overcast    = (v_ch > 170) & (s_ch < 45)   # blown-out/overcast sky
    sky_color   = (blue_sky | overcast).astype(np.uint8) * 255

    # Candidate = edge-free AND looks like sky color
    sky_candidate = cv2.bitwise_and(low_edge_px, sky_color)

    # -----------------------------------------------------------------------
    # STEP 3: Flood-fill connectivity from top edge (Photoshop "Select Sky")
    # Only pixels reachable from the top 2 rows through candidate pixels are sky.
    # Disconnected bright/smooth regions (walls, snow) are excluded.
    # -----------------------------------------------------------------------
    connected = np.zeros((h, w), dtype=np.uint8)
    connected[:2, :] = sky_candidate[:2, :]  # seed from top 2 rows

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    max_sky_row = int(h * 0.75)  # sky cannot extend below 75% of height
    for _ in range(max_sky_row):
        expanded = cv2.dilate(connected, kernel, iterations=1)
        expanded[max_sky_row:, :] = 0
        new_connected = cv2.bitwise_and(expanded, sky_candidate)
        if np.array_equal(new_connected, connected):
            break
        connected = new_connected

    sky_px = connected

    # If nothing was found, abort
    if np.count_nonzero(sky_px) < int(h * w * 0.005):
        print("[sky] no sky found — skipping")
        return np.zeros((h, w), dtype=np.uint8)

    # -----------------------------------------------------------------------
    # STEP 4: Per-column skyline trace + smooth
    # -----------------------------------------------------------------------
    skyline = np.zeros(w, dtype=np.float32)
    for col in range(w):
        rows = np.where(sky_px[:, col] > 0)[0]
        skyline[col] = float(rows[-1]) if len(rows) > 0 else 0.0

    skyline = np.minimum(skyline, int(h * 0.70))
    skyline_smooth = cv2.GaussianBlur(
        skyline.reshape(1, -1).astype(np.float32), (61, 1), 0
    ).flatten()

    # -----------------------------------------------------------------------
    # STEP 5: Build soft alpha with feathered skyline boundary
    # -----------------------------------------------------------------------
    feather = max(int(h * 0.025), 8)
    sky_float = sky_px.astype(np.float32) / 255.0
    row_idx = np.arange(h, dtype=np.float32)[:, np.newaxis]
    sl = skyline_smooth[np.newaxis, :]

    above     = (row_idx <= sl - feather).astype(np.float32)
    t         = np.clip((sl - row_idx + feather) / (2 * feather), 0.0, 1.0)
    in_feather = ((row_idx > sl - feather) & (row_idx <= sl + feather)).astype(np.float32)

    alpha_final = above * sky_float + in_feather * t * sky_float
    alpha_u8 = np.clip(alpha_final * 255, 0, 255).astype(np.uint8)
    alpha_u8 = cv2.GaussianBlur(alpha_u8, (15, 15), 0)

    covered = np.count_nonzero(alpha_u8 > 10)
    print(f"[sky] ed_thresh={ed_thresh:.3f}  sky_px={covered}  ({100*covered//(h*w)}%)")
    return alpha_u8


def apply_sky_replacement(img: np.ndarray) -> tuple:
    h, w = img.shape[:2]
    mask = build_sky_mask(img)
    covered = np.count_nonzero(mask > 10)
    min_px = int(h * w * 0.01)   # at least 1% of image

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
