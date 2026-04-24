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
    """
    Use Photomatix Online API to merge bracketed exposures.
    Steps: create engine → add images by URL → process with Real Estate preset → download result.
    Returns BGR uint8 ndarray.
    """
    headers = {"x-pm-token": api_key}

    # Step 1: Create HDR engine
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

    # Step 2: Add images by URL
    for i, url in enumerate(file_urls):
        # Extract filename — must have a valid image extension
        base = url.split("?")[0].rsplit("/", 1)[-1]
        ext = base.rsplit(".", 1)[-1].lower() if "." in base else "jpg"
        # Photomatix requires common extensions; RAW formats map to their extension
        filename = f"image_{i+1}.{ext}"
        print(f"Photomatix: adding image {filename} from URL...")
        add_r = requests.post(
            f"{PHOTOMATIX_API}{engine_uri}/images/{filename}",
            headers=headers,
            data={"url": url},
            timeout=60,
        )
        if add_r.status_code not in (200, 201):
            raise Exception(f"Photomatix add image failed ({add_r.status_code}): {add_r.text[:300]}")

    # Step 3: Process with Real Estate preset and get JPEG back
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
    # Poll until ready (Photomatix may return 202 while processing)
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
# Fallback: local RAW loading + Mertens (used if no Photomatix key)
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
        rgb = raw.postprocess(use_camera_wb=True, no_auto_bright=False, output_bps=8, half_size=True)
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


def local_merge(file_urls: List[str]) -> tuple:
    """Download, decode RAW, align, Mertens merge. Returns (merged, dark_img)."""
    tmp_paths = []
    try:
        for url in file_urls:
            ext = url.split("?")[0].rsplit(".", 1)[-1]
            ext = f".{ext.lower()}" if ext else ".jpg"
            tmp_paths.append(download_file(url, ext))

        images = [load_image_bgr(p) for p in tmp_paths]
        gc.collect()

        dark_img = min(images, key=lambda img: float(np.mean(img))).copy()

        if len(images) > 1:
            aligned = [images[0]]
            ref_gray = cv2.cvtColor(images[0], cv2.COLOR_BGR2GRAY)
            for img in images[1:]:
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                warp = np.eye(2, 3, dtype=np.float32)
                try:
                    _, warp = cv2.findTransformECC(
                        ref_gray, gray, warp, cv2.MOTION_TRANSLATION,
                        (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.001),
                    )
                    aligned.append(cv2.warpAffine(img, warp, (OUTPUT_WIDTH, OUTPUT_HEIGHT),
                                                  flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP))
                except Exception:
                    aligned.append(img)
                del gray
            del ref_gray
            fused = cv2.createMergeMertens(
                contrast_weight=1.4, saturation_weight=0.9, exposure_weight=0.2
            ).process(aligned)
            merged = np.clip(fused * 255, 0, 255).astype(np.uint8)
            del aligned, fused
        else:
            merged = images[0]

        gc.collect()
        return merged, dark_img
    finally:
        for p in tmp_paths:
            try:
                os.unlink(p)
            except Exception:
                pass


# ---------------------------------------------------------------------------
# Geometry correction
# ---------------------------------------------------------------------------

def correct_geometry(img: np.ndarray) -> np.ndarray:
    h, w = img.shape[:2]
    cx, cy = w / 2.0, h / 2.0

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray_eq = cv2.equalizeHist(gray)
    edges = cv2.Canny(gray_eq, 30, 100, apertureSize=3)
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=60,
                            minLineLength=h // 5, maxLineGap=15)

    vert_lines = []
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            dx, dy = float(x2 - x1), float(y2 - y1)
            if abs(dy) > abs(dx) * 2.5 and abs(dy) > h // 5:
                vert_lines.append((float(x1), float(y1), float(x2), float(y2)))

    print(f"Vertical lines: {len(vert_lines)}")
    if len(vert_lines) < 6:
        print("Not enough vertical lines — skipping VP correction")
        del gray, gray_eq, edges, lines
        gc.collect()
        return img

    import random
    sample = vert_lines if len(vert_lines) <= 40 else random.sample(vert_lines, 40)
    vp_candidates = []
    for i in range(len(sample)):
        for j in range(i + 1, len(sample)):
            x1, y1, x2, y2 = sample[i]
            x3, y3, x4, y4 = sample[j]
            denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
            if abs(denom) < 1e-6:
                continue
            t = ((x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)) / denom
            ix = x1 + t * (x2 - x1)
            iy = y1 + t * (y2 - y1)
            if (iy < -h * 0.05 or iy > h * 1.05) and (-w * 0.3 < ix < w * 1.3):
                vp_candidates.append((ix, iy))

    print(f"VP candidates: {len(vp_candidates)}")
    if len(vp_candidates) < 6:
        print("Not enough VP candidates — skipping correction")
        del gray, gray_eq, edges, lines
        gc.collect()
        return img

    vpy = float(np.median([v[1] for v in vp_candidates]))
    vpx = float(np.median([v[0] for v in vp_candidates]))
    print(f"VP: ({vpx:.0f}, {vpy:.0f})")

    if h * 0.10 <= vpy <= h * 0.90:
        print("VP inside image — skipping")
        del gray, gray_eq, edges, lines
        gc.collect()
        return img

    vpy_c = vpy - cy
    p = -1.0 / vpy_c
    max_p = 1.0 / (0.25 * h)
    p = float(np.clip(p, -max_p, max_p))

    H = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, p, 1.0]], dtype=np.float64)
    T = np.array([[1, 0, -cx], [0, 1, -cy], [0, 0, 1]], dtype=np.float64)
    T_inv = np.array([[1, 0, cx], [0, 1, cy], [0, 0, 1]], dtype=np.float64)
    H_full = T_inv @ H @ T

    result = cv2.warpPerspective(img, H_full, (w, h), flags=cv2.INTER_LINEAR,
                                 borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))

    corners_src = np.array([[0, 0, 1], [w, 0, 1], [w, h, 1], [0, h, 1]], dtype=np.float64).T
    corners_dst = H_full @ corners_src
    corners_dst /= corners_dst[2, :]
    dst_x, dst_y = corners_dst[0, :], corners_dst[1, :]

    x0 = max(int(np.ceil(max(dst_x[0], dst_x[3]))) + 2, 0)
    x1 = min(int(np.floor(min(dst_x[1], dst_x[2]))) - 2, w)
    y0 = max(int(np.ceil(max(dst_y[0], dst_y[1]))) + 2, 0)
    y1 = min(int(np.floor(min(dst_y[2], dst_y[3]))) - 2, h)

    if x1 > x0 + 100 and y1 > y0 + 100:
        result = cv2.resize(result[y0:y1, x0:x1], (w, h), interpolation=cv2.INTER_LANCZOS4)

    del gray, gray_eq, edges, lines
    gc.collect()
    return result


# ---------------------------------------------------------------------------
# Window detection & pull
# ---------------------------------------------------------------------------

def detect_window_mask(img: np.ndarray) -> np.ndarray:
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    L = lab[:, :, 0]
    _, blown = cv2.threshold(L, 200, 255, cv2.THRESH_BINARY)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    low_sat = (hsv[:, :, 1] < 25).astype(np.uint8) * 255
    bright_l = (L > 185).astype(np.uint8) * 255
    seed = cv2.bitwise_or(blown, cv2.bitwise_and(low_sat, bright_l))

    close_k = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 25))
    open_k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    seed = cv2.morphologyEx(seed, cv2.MORPH_CLOSE, close_k)
    seed = cv2.morphologyEx(seed, cv2.MORPH_OPEN, open_k)

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(seed, connectivity=8)
    min_area = (OUTPUT_WIDTH * OUTPUT_HEIGHT) * 0.0025
    filtered = np.zeros_like(seed)
    for i in range(1, num_labels):
        if stats[i, cv2.CC_STAT_AREA] >= min_area:
            filtered[labels == i] = 255

    dilate_k = cv2.getStructuringElement(cv2.MORPH_RECT, (60, 60))
    expanded = cv2.dilate(filtered, dilate_k, iterations=2)
    height = expanded.shape[0]
    expanded[:int(height * 0.10), :] = 0
    expanded[int(height * 0.55):, :] = 0

    erode_k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (22, 22))
    shrunk = cv2.erode(expanded, erode_k, iterations=5)

    feathered = cv2.GaussianBlur(shrunk.astype(np.float32), (0, 0), sigmaX=8)
    max_val = feathered.max()
    if max_val > 0:
        feathered /= max_val

    del lab, hsv, blown, low_sat, bright_l, seed, filtered, expanded, shrunk
    gc.collect()
    return feathered


def apply_window_pull(merged: np.ndarray, dark_img: np.ndarray,
                      mask: np.ndarray, strength: float = 0.70) -> np.ndarray:
    merged_f = merged.astype(np.float32)
    dark_f = np.clip(dark_img.astype(np.float32) * 2.0, 0, 255)
    mask3 = np.stack([mask * strength] * 3, axis=2)
    result = np.clip(merged_f * (1.0 - mask3) + dark_f * mask3, 0, 255).astype(np.uint8)

    hsv = cv2.cvtColor(result, cv2.COLOR_BGR2HSV).astype(np.float32)
    brightness = hsv[:, :, 2] / 255.0
    saturation = hsv[:, :, 1] / 255.0
    sky_pixel = (brightness > 0.55) & (saturation < 0.25)
    sky_strength = mask * sky_pixel.astype(np.float32) * 0.4
    result_f = result.astype(np.float32)
    result_f[:, :, 0] = np.clip(result_f[:, :, 0] + sky_strength * 10, 0, 255)
    result_f[:, :, 1] = np.clip(result_f[:, :, 1] + sky_strength * 3, 0, 255)
    result_f[:, :, 2] = np.clip(result_f[:, :, 2] - sky_strength * 6, 0, 255)

    del hsv, brightness, saturation, sky_pixel, sky_strength
    return np.clip(result_f, 0, 255).astype(np.uint8)


# ---------------------------------------------------------------------------
# Imagine Art — sky & window generative fill
# ---------------------------------------------------------------------------

def imagine_generative_fill(img: np.ndarray, mask: np.ndarray, prompt: str, api_key: str) -> np.ndarray:
    _, img_buf = cv2.imencode('.jpg', img, [cv2.IMWRITE_JPEG_QUALITY, 95])
    mask_u8 = (np.clip(mask, 0, 1) * 255).astype(np.uint8)
    mask_bgr = cv2.merge([mask_u8, mask_u8, mask_u8])
    _, mask_buf = cv2.imencode('.png', mask_bgr)
    try:
        response = requests.post(
            'https://api.vyro.ai/v2/image/edits/generative-fill',
            headers={'Authorization': f'Bearer {api_key}'},
            files={
                'image': ('image.jpg', img_buf.tobytes(), 'image/jpeg'),
                'mask': ('mask.png', mask_buf.tobytes(), 'image/png'),
            },
            data={'prompt': prompt},
            timeout=120,
        )
        if response.status_code == 200:
            result_arr = np.frombuffer(response.content, np.uint8)
            result_img = cv2.imdecode(result_arr, cv2.IMREAD_COLOR)
            if result_img is not None:
                return cv2.resize(result_img, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_LANCZOS4)
        print(f'Imagine fill error {response.status_code}: {response.text[:200]}')
    except Exception as e:
        print(f'Imagine fill exception: {e}')
    return img


def detect_sky_mask(img: np.ndarray) -> np.ndarray:
    h, w = img.shape[:2]
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    L = lab[:, :, 0]
    blown = (L > 210).astype(np.uint8) * 255
    sky_zone = np.zeros_like(blown)
    sky_zone[:int(h * 0.45), :] = blown[:int(h * 0.45), :]
    close_k = cv2.getStructuringElement(cv2.MORPH_RECT, (30, 30))
    sky_zone = cv2.morphologyEx(sky_zone, cv2.MORPH_CLOSE, close_k)
    open_k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    sky_zone = cv2.morphologyEx(sky_zone, cv2.MORPH_OPEN, open_k)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(sky_zone, connectivity=8)
    min_area = (w * h) * 0.02
    filtered = np.zeros_like(sky_zone)
    for i in range(1, num_labels):
        if stats[i, cv2.CC_STAT_AREA] >= min_area:
            filtered[labels == i] = 255
    feathered = cv2.GaussianBlur(filtered.astype(np.float32), (0, 0), sigmaX=10)
    max_val = feathered.max()
    if max_val > 0:
        feathered /= max_val
    del lab, L, blown, sky_zone, filtered
    gc.collect()
    return feathered


# ---------------------------------------------------------------------------
# Light post-processing on top of Photomatix result
# ---------------------------------------------------------------------------

def post_process(img: np.ndarray) -> np.ndarray:
    """
    Light touch post-processing on the Photomatix-merged result:
    - Gentle brightness lift if needed
    - Mild sharpening
    Photomatix handles tone mapping/colour so we keep this minimal.
    """
    f = img.astype(np.float32) / 255.0

    # Gentle brightness check — only lift if image is dark
    mean_lum = float(np.mean(0.2126 * f[:,:,2] + 0.7152 * f[:,:,1] + 0.0722 * f[:,:,0]))
    if mean_lum < 0.45:
        ev = min(2.0 ** (np.log2(0.55 / max(mean_lum, 0.01)) * 0.6), 1.5)
        f = np.clip(f * ev, 0, 1)
        print(f"Post-process: brightness lift applied (mean was {mean_lum:.2f})")

    result = np.clip(f * 255, 0, 255).astype(np.uint8)

    # Gentle sharpening
    blurred = cv2.GaussianBlur(result, (0, 0), sigmaX=1.0)
    result = cv2.addWeighted(result, 1.2, blurred, -0.2, 0)

    del f, blurred
    gc.collect()
    return result


# ---------------------------------------------------------------------------
# Request model & endpoint
# ---------------------------------------------------------------------------

class MergeRequest(BaseModel):
    file_urls: List[str]
    bracket_name: str = "bracket"
    imagine_api_key: Optional[str] = None
    replace_sky: bool = True


@app.post("/merge")
async def merge_hdr(req: MergeRequest):
    if not req.file_urls:
        raise HTTPException(400, "No file URLs provided")
    if len(req.file_urls) not in (1, 3, 5):
        raise HTTPException(400, f"Expected 1, 3, or 5 files, got {len(req.file_urls)}")

    photomatix_key = os.environ.get("PHOTOMATIX_API_KEY")

    dark_img = None
    tmp_paths = []

    try:
        # --- Merge ---
        if photomatix_key:
            print("Using Photomatix API for HDR merge...")
            merged = photomatix_merge(req.file_urls, photomatix_key)
            merged = cv2.resize(merged, (OUTPUT_WIDTH, OUTPUT_HEIGHT), interpolation=cv2.INTER_LANCZOS4)

            # For window pull we need the darkest bracket locally
            print("Downloading darkest bracket for window pull...")
            try:
                url = req.file_urls[0]  # fallback: first image
                ext = url.split("?")[0].rsplit(".", 1)[-1]
                ext = f".{ext.lower()}" if ext else ".jpg"
                dark_path = download_file(url, ext)
                tmp_paths.append(dark_path)
                dark_img = load_image_bgr(dark_path)
                # If 3 or 5 shots, try to get actual darkest (first is usually darkest in bracket order)
                # For now use first frame
            except Exception as e:
                print(f"Could not load dark bracket: {e} — window pull may be reduced")
                dark_img = merged.copy()
        else:
            print("No Photomatix key — falling back to local Mertens merge...")
            merged, dark_img = local_merge(req.file_urls)

        # --- Geometry correction ---
        print("Correcting geometry on merged...")
        merged = correct_geometry(merged)
        if dark_img is not None:
            print("Correcting geometry on dark bracket...")
            dark_img = correct_geometry(dark_img)

        # --- Window detection ---
        window_mask = None
        print("Detecting windows...")
        window_mask = detect_window_mask(merged)
        window_coverage = float(np.mean(window_mask))
        print(f"Window mask coverage: {window_coverage:.4f}")
        if window_coverage <= 0.0003:
            print("No significant window regions — skipping window pull")
            window_mask = None

        # --- Light post-processing ---
        toned = post_process(merged)
        del merged
        gc.collect()

        # --- Window pull ---
        if window_mask is not None:
            if req.imagine_api_key:
                print("Window pull via Imagine Art...")
                toned = imagine_generative_fill(
                    toned, window_mask,
                    "bright exterior view through window, blue sky outside, natural daylight, professional real estate photography",
                    req.imagine_api_key,
                )
            elif dark_img is not None:
                print("Window pull via local blend...")
                toned = apply_window_pull(toned, dark_img, window_mask, strength=0.70)

        # --- Sky replacement ---
        if req.imagine_api_key and req.replace_sky:
            print("Detecting sky for replacement...")
            sky_mask = detect_sky_mask(toned)
            sky_coverage = float(np.mean(sky_mask))
            print(f"Sky mask coverage: {sky_coverage:.4f}")
            if sky_coverage > 0.005:
                print("Replacing sky via Imagine Art...")
                toned = imagine_generative_fill(
                    toned, sky_mask,
                    "beautiful blue sky with white clouds, golden hour light, professional real estate exterior photography",
                    req.imagine_api_key,
                )
            else:
                print("No significant sky — skipping replacement")

        if dark_img is not None:
            del dark_img
        gc.collect()

        # --- Encode ---
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
