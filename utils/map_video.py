'''
# utils/map_video.py
import io
import math
from pathlib import Path

import gpxpy
import gpxpy.gpx
import numpy as np
import requests
from PIL import Image, ImageDraw

import imageio
import imageio_ffmpeg  # ensures ffmpeg is bundled

TILE_SIZE = 256
OSM_TILE_URL = "https://tile.openstreetmap.org/{z}/{x}/{y}.png"
HTTP_HEADERS = {"User-Agent": "GPXFlyover/0.1 (demo; contact: you@example.com)"}

class GPXParseError(Exception):
    pass

def _parse_gpx_lonlat(gpx_path: str):
    with open(gpx_path, "r", encoding="utf-8") as f:
        gpx = gpxpy.parse(f)
    lons, lats = [], []
    for trk in gpx.tracks:
        for seg in trk.segments:
            for p in seg.points:
                if p.latitude is None or p.longitude is None:
                    continue
                lons.append(float(p.longitude))
                lats.append(float(p.latitude))
    if len(lons) < 2:
        raise GPXParseError("Not enough GPX points.")
    return np.asarray(lons), np.asarray(lats)

def _lonlat_to_global_px(lon_deg: np.ndarray, lat_deg: np.ndarray, z: int):
    lat_deg = np.clip(lat_deg, -85.05112878, 85.05112878)
    n = 2 ** z
    lat_rad = np.deg2rad(lat_deg)
    x = (lon_deg + 180.0) / 360.0 * n * TILE_SIZE
    y = (1.0 - np.log(np.tan(lat_rad) + 1.0 / np.cos(lat_rad)) / math.pi) / 2.0 * n * TILE_SIZE
    return x, y

def _choose_zoom(lon: np.ndarray, lat: np.ndarray, max_mosaic_px: int = 4096):
    lat0 = float(np.median(lat))
    def mpp(z):  # meters per pixel
        return 156543.03392 * math.cos(math.radians(lat0)) / (2 ** z)
    dlon = float(np.max(lon) - np.min(lon))
    dlat = float(np.max(lat) - np.min(lat))
    span_m = max(dlon * 111320 * math.cos(math.radians(lat0)),
                 dlat * 110540)
    best = 12
    for z in range(6, 19):
        if span_m / mpp(z) <= max_mosaic_px:
            best = z
        else:
            break
    return max(6, min(best, 18))

def _tiles_bbox_from_global_px(xg: np.ndarray, yg: np.ndarray, z: int, margin_tiles: int = 1):
    min_tile_x = int(np.floor(np.min(xg) / TILE_SIZE)) - margin_tiles
    max_tile_x = int(np.floor(np.max(xg) / TILE_SIZE)) + margin_tiles
    min_tile_y = int(np.floor(np.min(yg) / TILE_SIZE)) - margin_tiles
    max_tile_y = int(np.floor(np.max(yg) / TILE_SIZE)) + margin_tiles
    n = (2 ** z) - 1
    min_tile_x = max(0, min_tile_x); min_tile_y = max(0, min_tile_y)
    max_tile_x = min(n, max_tile_x); max_tile_y = min(n, max_tile_y)
    return min_tile_x, min_tile_y, max_tile_x, max_tile_y

def _fetch_tile(z: int, x: int, y: int, cache_dir: Path) -> Image.Image:
    cache_path = cache_dir / f"{z}_{x}_{y}.png"
    if cache_path.exists():
        return Image.open(cache_path).convert("RGB")
    url = OSM_TILE_URL.format(z=z, x=x, y=y)
    r = requests.get(url, headers=HTTP_HEADERS, timeout=15)
    r.raise_for_status()
    img = Image.open(io.BytesIO(r.content)).convert("RGB")
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(cache_path)
    return img

def _build_mosaic(z: int, x0: int, y0: int, x1: int, y1: int, cache_dir: Path) -> Image.Image:
    cols = x1 - x0 + 1
    rows = y1 - y0 + 1
    mosaic = Image.new("RGB", (cols * TILE_SIZE, rows * TILE_SIZE))
    for ty in range(y0, y1 + 1):
        for tx in range(x0, x1 + 1):
            tile = _fetch_tile(z, tx, ty, cache_dir)
            mosaic.paste(tile, ((tx - x0) * TILE_SIZE, (ty - y0) * TILE_SIZE))
    return mosaic

def _resample_along_distance(x: np.ndarray, y: np.ndarray, n: int):
    dx = np.diff(x); dy = np.diff(y)
    d = np.hypot(dx, dy)
    cum = np.concatenate([[0.0], np.cumsum(d)])
    if cum[-1] <= 0:
        return x, y
    new_cum = np.linspace(0, float(cum[-1]), n)
    return np.interp(new_cum, cum, x), np.interp(new_cum, cum, y)

def generate_map_flyover_video(
    gpx_file: str,
    out_file: str,
    fps: int = 30,
    target_duration: int = 60,
    zoom_padding_ratio: float = 0.15,  # kept for API shape
    style: str = "osm",                # unused here; kept for API shape
    width: int = 1280,
    height: int = 720,
):
    lon, lat = _parse_gpx_lonlat(gpx_file)
    z = _choose_zoom(lon, lat, max_mosaic_px=4096)
    xg, yg = _lonlat_to_global_px(lon, lat, z)

    frames = max(int(fps * target_duration), 60)
    xg, yg = _resample_along_distance(xg, yg, frames)

    x0_tile, y0_tile, x1_tile, y1_tile = _tiles_bbox_from_global_px(xg, yg, z, margin_tiles=1)
    mosaic = _build_mosaic(z, x0_tile, y0_tile, x1_tile, y1_tile, cache_dir=Path("tile_cache"))
    origin_x = x0_tile * TILE_SIZE
    origin_y = y0_tile * TILE_SIZE

    writer = imageio.get_writer(
        out_file,
        fps=fps,
        codec="libx264",
        macro_block_size=None,
        ffmpeg_log_level="warning",
        ffmpeg_params=["-pix_fmt", "yuv420p", "-movflags", "faststart"],
    )

    px = xg - origin_x
    py = yg - origin_y

    for i in range(frames):
        cx, cy = px[i], py[i]
        left = int(round(cx - width / 2))
        top  = int(round(cy - height / 2))
        left = max(0, min(left, mosaic.width - width))
        top  = max(0, min(top,  mosaic.height - height))

        frame = mosaic.crop((left, top, left + width, top + height)).copy()

        draw = ImageDraw.Draw(frame, "RGBA")
        j0 = max(0, i - 600)  # ~20 s of history
        pts = list(zip((px[j0:i+1] - left), (py[j0:i+1] - top)))
        if len(pts) >= 2:
            draw.line(pts, fill=(25, 118, 210, 255), width=4)
        r = 6
        draw.ellipse((pts[-1][0]-r, pts[-1][1]-r, pts[-1][0]+r, pts[-1][1]+r), fill=(255,109,0,255))

        writer.append_data(np.array(frame))

    writer.close()
    return {"frames": frames, "fps": fps, "duration_s": frames / float(fps)}
'''

'''
# utils/map_video.py
import io
import math
import json
from pathlib import Path

import gpxpy
import numpy as np
import requests
from PIL import Image, ImageDraw, ImageFont

import imageio
import imageio_ffmpeg  # ensure ffmpeg is bundled

TILE_SIZE = 256
OSM_TILE_URL = "https://tile.openstreetmap.org/{z}/{x}/{y}.png"
HTTP_HEADERS = {"User-Agent": "GPXFlyover/0.3"}

class GPXParseError(Exception):
    pass

# ---------- GPX & tiles helpers (unchanged) ----------
def _parse_gpx_lonlat(gpx_path: str):
    with open(gpx_path, "r", encoding="utf-8") as f:
        gpx = gpxpy.parse(f)
    lons, lats, eles, times = [], [], [], []
    for trk in gpx.tracks:
        for seg in trk.segments:
            for p in seg.points:
                if p.latitude is None or p.longitude is None:
                    continue
                lons.append(float(p.longitude))
                lats.append(float(p.latitude))
                eles.append(float(p.elevation) if p.elevation is not None else None)
                times.append(p.time)
    if len(lons) < 2:
        raise GPXParseError("Not enough GPX points.")
    return np.asarray(lons), np.asarray(lats), eles, times

def _lonlat_to_global_px(lon_deg: np.ndarray, lat_deg: np.ndarray, z: int):
    lat_deg = np.clip(lat_deg, -85.05112878, 85.05112878)
    n = 2 ** z
    lat_rad = np.deg2rad(lat_deg)
    x = (lon_deg + 180.0) / 360.0 * n * TILE_SIZE
    y = (1.0 - np.log(np.tan(lat_rad) + 1.0 / np.cos(lat_rad)) / math.pi) / 2.0 * n * TILE_SIZE
    return x, y

def _choose_zoom(lon: np.ndarray, lat: np.ndarray, max_mosaic_px: int = 4096):
    lat0 = float(np.median(lat))
    def mpp(z):  # meters per pixel
        return 156543.03392 * math.cos(math.radians(lat0)) / (2 ** z)
    dlon = float(np.max(lon) - np.min(lon))
    dlat = float(np.max(lat) - np.min(lat))
    span_m = max(dlon * 111320 * math.cos(math.radians(lat0)),
                 dlat * 110540)
    best = 12
    for z in range(6, 19):
        if span_m / mpp(z) <= max_mosaic_px:
            best = z
        else:
            break
    return max(6, min(best, 18))

def _tiles_bbox_from_global_px(xg: np.ndarray, yg: np.ndarray, z: int, margin_tiles: int = 1):
    min_tile_x = int(np.floor(np.min(xg) / TILE_SIZE)) - margin_tiles
    max_tile_x = int(np.floor(np.max(xg) / TILE_SIZE)) + margin_tiles
    min_tile_y = int(np.floor(np.min(yg) / TILE_SIZE)) - margin_tiles
    max_tile_y = int(np.floor(np.max(yg) / TILE_SIZE)) + margin_tiles
    n = (2 ** z) - 1
    min_tile_x = max(0, min_tile_x); min_tile_y = max(0, min_tile_y)
    max_tile_x = min(n, max_tile_x); max_tile_y = min(n, max_tile_y)
    return min_tile_x, min_tile_y, max_tile_x, max_tile_y

def _fetch_tile(z: int, x: int, y: int, cache_dir: Path) -> Image.Image:
    cache_path = cache_dir / f"{z}_{x}_{y}.png"
    if cache_path.exists():
        return Image.open(cache_path).convert("RGB")
    url = OSM_TILE_URL.format(z=z, x=x, y=y)
    r = requests.get(url, headers=HTTP_HEADERS, timeout=20)
    r.raise_for_status()
    img = Image.open(io.BytesIO(r.content)).convert("RGB")
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(cache_path)
    return img

def _build_mosaic(z: int, x0: int, y0: int, x1: int, y1: int, cache_dir: Path) -> Image.Image:
    cols = x1 - x0 + 1
    rows = y1 - y0 + 1
    mosaic = Image.new("RGB", (cols * TILE_SIZE, rows * TILE_SIZE))
    for ty in range(y0, y1 + 1):
        for tx in range(x0, x1 + 1):
            tile = _fetch_tile(z, tx, ty, cache_dir)
            mosaic.paste(tile, ((tx - x0) * TILE_SIZE, (ty - y0) * TILE_SIZE))
    return mosaic

def _resample_along_distance(x: np.ndarray, y: np.ndarray, n: int):
    dx = np.diff(x); dy = np.diff(y)
    d = np.hypot(dx, dy)
    cum = np.concatenate([[0.0], np.cumsum(d)])
    if cum[-1] <= 0:
        return x, y
    new_cum = np.linspace(0, float(cum[-1]), n)
    return np.interp(new_cum, cum, x), np.interp(new_cum, cum, y)

def _heading_unit(x: np.ndarray, y: np.ndarray, i: int, k: int = 3):
    i0 = max(0, i - k); i1 = min(len(x) - 1, i + k)
    vx = x[i1] - x[i0]; vy = y[i1] - y[i0]
    n = math.hypot(vx, vy)
    if n == 0:
        return 0.0, -1.0
    return vx / n, vy / n

# ---------- 2D video generator (unchanged) ----------
def generate_map_flyover_video(
    gpx_file: str,
    out_file: str,
    fps: int = 30,
    target_duration: int = 60,
    zoom_padding_ratio: float = 0.15,
    style: str = "osm",
    width: int = 1280,
    height: int = 720,
    follow_back_meters: float = 350.0,
):
    # --- parse and choose zoom as before ---
    lon, lat, _eles, _times = _parse_gpx_lonlat(gpx_file)
    z = _choose_zoom(lon, lat, max_mosaic_px=4096)
    xg, yg = _lonlat_to_global_px(lon, lat, z)

    # meters per pixel at median latitude
    lat0 = float(np.median(lat))
    mpp = 156543.03392 * math.cos(math.radians(lat0)) / (2 ** z)
    back_px = follow_back_meters / max(1e-6, mpp)

    # frame count and resampled path (so i == frame index)
    frames = max(int(fps * target_duration), 60)
    xg, yg = _resample_along_distance(xg, yg, frames)

    # ---------- NEW: compute all camera centers, then size mosaic to cover ALL crops ----------
    # heading vectors for each frame
    ux = np.zeros(frames, dtype=float)
    uy = np.zeros(frames, dtype=float)
    for i in range(frames):
        ux[i], uy[i] = _heading_unit(xg, yg, i, k=4)

    # chase-cam center (behind the dot)
    cx_all = xg - ux * back_px
    cy_all = yg - uy * back_px

    half_w = width / 2.0
    half_h = height / 2.0

    # full pixel window needed to cover every crop across the animation
    need_min_px = float(np.min(cx_all - half_w))
    need_max_px = float(np.max(cx_all + half_w))
    need_min_py = float(np.min(cy_all - half_h))
    need_max_py = float(np.max(cy_all + half_h))

    # convert to tile indices with a small safety margin (extra tiles to hide seams)
    margin_tiles = 2
    min_tx = int(math.floor(need_min_px / TILE_SIZE)) - margin_tiles
    max_tx = int(math.floor(need_max_px / TILE_SIZE)) + margin_tiles
    min_ty = int(math.floor(need_min_py / TILE_SIZE)) - margin_tiles
    max_ty = int(math.floor(need_max_py / TILE_SIZE)) + margin_tiles

    # ensure the mosaic is at least as large as a single frame
    cols = (max_tx - min_tx + 1)
    rows = (max_ty - min_ty + 1)
    if cols * TILE_SIZE < width:
        extra = math.ceil((width - cols * TILE_SIZE) / TILE_SIZE)
        min_tx -= (extra + 1) // 2
        max_tx += extra // 2
    if rows * TILE_SIZE < height:
        extra = math.ceil((height - rows * TILE_SIZE) / TILE_SIZE)
        min_ty -= (extra + 1) // 2
        max_ty += extra // 2

    # clamp tile indices to valid range
    n = (2 ** z) - 1
    min_tx = max(0, min_tx); min_ty = max(0, min_ty)
    max_tx = min(n, max_tx); max_ty = min(n, max_ty)

    # build mosaic that we GUARANTEE is big enough for every crop
    mosaic = _build_mosaic(z, min_tx, min_ty, max_tx, max_ty, cache_dir=Path("tile_cache"))
    origin_x = min_tx * TILE_SIZE
    origin_y = min_ty * TILE_SIZE

    # track coords relative to mosaic origin (for drawing)
    px = xg - origin_x
    py = yg - origin_y
    cx_all -= origin_x
    cy_all -= origin_y

    # ---------- encode frames ----------
    writer = imageio.get_writer(
        out_file, fps=fps, codec="libx264",
        macro_block_size=None, ffmpeg_log_level="warning",
        ffmpeg_params=["-pix_fmt", "yuv420p", "-movflags", "faststart"],
    )

    for i in range(frames):
        # we already computed camera center for each frame
        cx = cx_all[i]
        cy = cy_all[i]

        left = int(round(cx - half_w))
        top  = int(round(cy - half_h))

        # final safety clamp (should rarely trigger now)
        left = max(0, min(left, mosaic.width  - width))
        top  = max(0, min(top,  mosaic.height - height))

        frame = mosaic.crop((left, top, left + width, top + height)).copy()

        # draw the recent path and the dot
        draw = ImageDraw.Draw(frame, "RGBA")
        j0 = max(0, i - 600)  # ~20s tail at 30 fps
        pts = list(zip((px[j0:i+1] - left), (py[j0:i+1] - top)))
        if len(pts) >= 2:
            draw.line(pts, fill=(25, 118, 210, 255), width=4)
        if pts:
            r = 6
            draw.ellipse((pts[-1][0]-r, pts[-1][1]-r, pts[-1][0]+r, pts[-1][1]+r),
                         fill=(255,109,0,255), outline=(255,255,255,255), width=2)

        writer.append_data(np.array(frame))

    writer.close()
    return {"frames": frames, "fps": fps, "duration_s": frames / float(fps)}


# ---------- NEW: Burn LIVE HUD for downloads ----------
def burn_hud_on_video(video_file: str, profile_json_file: str, out_file: str):
    """
    Burn a LIVE HUD onto the mp4: per-frame speed (mph), elevation (ft), and distance (mi).
    Uses the profile JSON so downloaded video matches the on-page HUD.
    """
    with open(profile_json_file, "r", encoding="utf-8") as f:
        prof = json.load(f)

    dist = np.array(prof["distance_miles"], dtype=float)
    elev = np.array(prof["elevation_feet"], dtype=float)
    gain = np.array(prof["cum_gain_feet"], dtype=float)
    total_miles = float(prof["total_miles"])
    duration_s  = float(prof.get("duration_s", 60.0))
    has_time    = bool(prof.get("has_time", False))
    time_s      = np.array(prof.get("time_s", []), dtype=float)
    moving_time_s = float(prof.get("moving_time_s", duration_s))
    avg_mph = total_miles / max(1e-6, (moving_time_s / 3600.0))

    # segment mph
    seg_mph = []
    if has_time and len(time_s) == len(dist):
        for i in range(len(dist)-1):
            dmi = max(0.0, dist[i+1] - dist[i])
            dt  = max(1e-3, (time_s[i+1] - time_s[i])) / 3600.0
            seg_mph.append(dmi / dt)
    else:
        seg_mph = [avg_mph] * (len(dist)-1)
    seg_mph = np.array(seg_mph, dtype=float)

    def idx_for_miles(mi):
        i = int(np.searchsorted(dist, mi)) - 1
        return max(0, min(i, len(dist)-2))

    def mph_at_miles(mi):
        i = idx_for_miles(mi)
        j0 = max(0, i-2); j1 = min(len(seg_mph)-1, i+2)
        return float(np.mean(seg_mph[j0:j1+1]))

    def elev_at_miles(mi):
        if mi <= dist[0]: return float(elev[0])
        if mi >= dist[-1]: return float(elev[-1])
        i = idx_for_miles(mi)
        t = (mi - dist[i]) / max(1e-9, (dist[i+1] - dist[i]))
        return float(elev[i] + t*(elev[i+1]-elev[i]))

    reader = imageio.get_reader(video_file)
    meta = reader.get_meta_data()
    fps = float(meta.get("fps", 30.0))
    writer = imageio.get_writer(
        out_file, fps=fps, codec="libx264",
        macro_block_size=None, ffmpeg_log_level="warning",
        ffmpeg_params=["-pix_fmt", "yuv420p", "-movflags", "faststart"]
    )

    try:
        font = ImageFont.truetype("Arial.ttf", 18)
        font_b = ImageFont.truetype("Arial Bold.ttf", 18)
    except Exception:
        font = ImageFont.load_default()
        font_b = font

    for i, frame in enumerate(reader):
        t = i / fps
        miles = total_miles * (t / duration_s)
        mph_now  = mph_at_miles(miles)
        elev_now = elev_at_miles(miles)

        pil = Image.fromarray(frame).convert("RGBA")
        draw = ImageDraw.Draw(pil, "RGBA")

        # panel
        panel_w, panel_h = 240, 94
        x0, y0 = 12, 12
        draw.rounded_rectangle((x0, y0, x0+panel_w, y0+panel_h), radius=14, fill=(0,0,0,140))

        # live values
        def txt(x, y, s, bold=False):
            draw.text((x, y), s, fill=(255,255,255,210), font=(font_b if bold else font))

        txt(x0+12, y0+10, "Speed");     txt(x0+140, y0+10, f"{mph_now:.1f} mph", True)
        txt(x0+12, y0+36, "Elevation"); txt(x0+140, y0+36, f"{elev_now:,.0f} ft", True)
        txt(x0+12, y0+62, "Distance");  txt(x0+140, y0+62, f"{miles:.2f} mi", True)

        writer.append_data(np.array(pil.convert("RGB")))

    reader.close()
    writer.close()
'''

import io
import math
import json
from pathlib import Path

import gpxpy
import numpy as np
import requests
from PIL import Image, ImageDraw, ImageFont

import imageio
import imageio_ffmpeg  # ensure ffmpeg is bundled

TILE_SIZE = 256
OSM_TILE_URL = "https://tile.openstreetmap.org/{z}/{x}/{y}.png"
HTTP_HEADERS = {"User-Agent": "GPXFlyover/0.4"}

class GPXParseError(Exception):
    pass


# ---------- GPX & tile helpers ----------
def _parse_gpx_lonlat(gpx_path: str):
    with open(gpx_path, "r", encoding="utf-8") as f:
        gpx = gpxpy.parse(f)
    lons, lats, eles, times = [], [], [], []
    for trk in gpx.tracks:
        for seg in trk.segments:
            for p in seg.points:
                if p.latitude is None or p.longitude is None:
                    continue
                lons.append(float(p.longitude))
                lats.append(float(p.latitude))
                eles.append(float(p.elevation) if p.elevation is not None else None)
                times.append(p.time)
    if len(lons) < 2:
        raise GPXParseError("Not enough GPX points.")
    return np.asarray(lons), np.asarray(lats), eles, times

def _lonlat_to_global_px(lon_deg: np.ndarray, lat_deg: np.ndarray, z: int):
    lat_deg = np.clip(lat_deg, -85.05112878, 85.05112878)
    n = 2 ** z
    lat_rad = np.deg2rad(lat_deg)
    x = (lon_deg + 180.0) / 360.0 * n * TILE_SIZE
    y = (1.0 - np.log(np.tan(lat_rad) + 1.0 / np.cos(lat_rad)) / math.pi) / 2.0 * n * TILE_SIZE
    return x, y

def _choose_zoom(lon: np.ndarray, lat: np.ndarray, max_mosaic_px: int = 4096):
    """Pick a reasonable zoom for OSM tiles; we will still resize mosaic later."""
    lat0 = float(np.median(lat))
    def mpp(z):  # meters per pixel
        return 156543.03392 * math.cos(math.radians(lat0)) / (2 ** z)
    dlon = float(np.max(lon) - np.min(lon))
    dlat = float(np.max(lat) - np.min(lat))
    span_m = max(dlon * 111320 * math.cos(math.radians(lat0)),
                 dlat * 110540)
    best = 12
    for z in range(6, 19):
        if span_m / mpp(z) <= max_mosaic_px:
            best = z
        else:
            break
    return max(6, min(best, 18))

def _fetch_tile(z: int, x: int, y: int, cache_dir: Path) -> Image.Image:
    cache_path = cache_dir / f"{z}_{x}_{y}.png"
    if cache_path.exists():
        return Image.open(cache_path).convert("RGB")
    url = OSM_TILE_URL.format(z=z, x=x, y=y)
    r = requests.get(url, headers=HTTP_HEADERS, timeout=20)
    r.raise_for_status()
    img = Image.open(io.BytesIO(r.content)).convert("RGB")
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(cache_path)
    return img

def _build_mosaic(z: int, x0: int, y0: int, x1: int, y1: int, cache_dir: Path) -> Image.Image:
    cols = x1 - x0 + 1
    rows = y1 - y0 + 1
    mosaic = Image.new("RGB", (cols * TILE_SIZE, rows * TILE_SIZE))
    for ty in range(y0, y1 + 1):
        for tx in range(x0, x1 + 1):
            tile = _fetch_tile(z, tx, ty, cache_dir)
            mosaic.paste(tile, ((tx - x0) * TILE_SIZE, (ty - y0) * TILE_SIZE))
    return mosaic

def _resample_along_distance(x: np.ndarray, y: np.ndarray, n: int):
    dx = np.diff(x); dy = np.diff(y)
    d = np.hypot(dx, dy)
    cum = np.concatenate([[0.0], np.cumsum(d)])
    if cum[-1] <= 0:
        return x, y
    new_cum = np.linspace(0, float(cum[-1]), n)
    return np.interp(new_cum, cum, x), np.interp(new_cum, cum, y)

def _heading_unit(x: np.ndarray, y: np.ndarray, i: int, k: int = 3):
    i0 = max(0, i - k); i1 = min(len(x) - 1, i + k)
    vx = x[i1] - x[i0]; vy = y[i1] - y[i0]
    n = math.hypot(vx, vy)
    if n == 0:
        return 0.0, -1.0
    return vx / n, vy / n


# ---------- 2D video generator (black-bar proof) ----------
def generate_map_flyover_video(
    gpx_file: str,
    out_file: str,
    fps: int = 30,
    target_duration: int = 60,
    zoom_padding_ratio: float = 0.15,   # kept for compatibility; not used in tile math
    style: str = "osm",
    width: int = 1280,
    height: int = 720,
    follow_back_meters: float = 350.0,
    zoom_boost: int = 0,
):
    """
    Build a single large mosaic that covers *all* camera crops, then crop each
    frame from that mosaic. This guarantees no black stripes even with a chase cam.
    """
    lon, lat, _eles, _times = _parse_gpx_lonlat(gpx_file)
    #z = _choose_zoom(lon, lat, max_mosaic_px=4096)
    base_z = _choose_zoom(lon, lat, max_mosaic_px=6144)
    z = min(18, max(6, base_z + int(zoom_boost)))
    xg, yg = _lonlat_to_global_px(lon, lat, z)

    # meters per pixel at median latitude
    lat0 = float(np.median(lat))
    mpp = 156543.03392 * math.cos(math.radians(lat0)) / (2 ** z)
    back_px = follow_back_meters / max(1e-6, mpp)

    # frames = duration * fps; resample path to frames so i == frame index
    frames = max(int(fps * target_duration), 60)
    xg, yg = _resample_along_distance(xg, yg, frames)

    # Precompute camera centers for every frame and size the mosaic to cover all crops
    ux = np.zeros(frames, dtype=float)
    uy = np.zeros(frames, dtype=float)
    for i in range(frames):
        ux[i], uy[i] = _heading_unit(xg, yg, i, k=4)
    cx_all = xg - ux * back_px
    cy_all = yg - uy * back_px

    half_w = width / 2.0
    half_h = height / 2.0

    need_min_px = float(np.min(cx_all - half_w))
    need_max_px = float(np.max(cx_all + half_w))
    need_min_py = float(np.min(cy_all - half_h))
    need_max_py = float(np.max(cy_all + half_h))

    # Convert to tile indices with a safety margin
    is_portrait = height > width
    margin_tiles = 3 if is_portrait else 2
    #margin_tiles = 2
    min_tx = int(math.floor(need_min_px / TILE_SIZE)) - margin_tiles
    max_tx = int(math.floor(need_max_px / TILE_SIZE)) + margin_tiles
    min_ty = int(math.floor(need_min_py / TILE_SIZE)) - margin_tiles
    max_ty = int(math.floor(need_max_py / TILE_SIZE)) + margin_tiles

    # Ensure mosaic >= one frame
    cols = (max_tx - min_tx + 1)
    rows = (max_ty - min_ty + 1)
    if cols * TILE_SIZE < width:
        extra = math.ceil((width - cols * TILE_SIZE) / TILE_SIZE)
        min_tx -= (extra + 1) // 2
        max_tx += extra // 2
    if rows * TILE_SIZE < height:
        extra = math.ceil((height - rows * TILE_SIZE) / TILE_SIZE)
        min_ty -= (extra + 1) // 2
        max_ty += extra // 2

    # Clamp tile indices to valid range
    n = (2 ** z) - 1
    min_tx = max(0, min_tx); min_ty = max(0, min_ty)
    max_tx = min(n, max_tx); max_ty = min(n, max_ty)

    # Build mosaic covering ALL crops
    mosaic = _build_mosaic(z, min_tx, min_ty, max_tx, max_ty, cache_dir=Path("tile_cache"))
    origin_x = min_tx * TILE_SIZE
    origin_y = min_ty * TILE_SIZE

    # Track relative to mosaic origin
    px = xg - origin_x
    py = yg - origin_y
    cx_all -= origin_x
    cy_all -= origin_y

    # Thicker path/dot on portrait so they read larger
    scale_ui = max(1.0, min(width, height) / 720.0)
    line_w = int(round(5 * scale_ui if is_portrait else 4 * scale_ui))
    dot_r  = int(round(8 * scale_ui if is_portrait else 6 * scale_ui))

    writer = imageio.get_writer(
        out_file, fps=fps, codec="libx264",
        macro_block_size=None, ffmpeg_log_level="warning",
        ffmpeg_params=["-pix_fmt", "yuv420p", "-movflags", "faststart"],
    )

    for i in range(frames):
        cx = cx_all[i]; cy = cy_all[i]
        left = int(round(cx - half_w))
        top  = int(round(cy - half_h))

        # Final guard (should rarely trigger now)
        left = max(0, min(left, mosaic.width  - width))
        top  = max(0, min(top,  mosaic.height - height))

        frame = mosaic.crop((left, top, left + width, top + height)).copy()

        # Draw recent tail + dot
        draw = ImageDraw.Draw(frame, "RGBA")
        j0 = max(0, i - 600)  # ~20s trail at 30fps
        pts = list(zip((px[j0:i+1] - left), (py[j0:i+1] - top)))
        if len(pts) >= 2:
            draw.line(pts, fill=(25, 118, 210, 255), width=line_w)
        if pts:
            #r = 6
            x, y = pts[-1]
            draw.ellipse((x-dot_r, y-dot_r, x+dot_r, y+dot_r),
                         fill=(255,109,0,255), outline=(255,255,255,255), width=2)

        writer.append_data(np.array(frame))

    writer.close()
    return {"frames": frames, "fps": fps, "duration_s": frames / float(fps)}


# ---------- Flexible burner: stats HUD and/or mini elevation chart ----------
def burn_overlays_on_video(video_file: str, profile_json_file: str, out_file: str,
                           show_stats: bool = True, show_chart: bool = True):
    """
    Burn overlays onto the mp4:
      - stats HUD (speed/elev/distance), if show_stats
      - mini elevation vs distance chart with live cursor, if show_chart
    Uses the profile JSON so downloaded video matches the on-page HUD/chart.
    """
    with open(profile_json_file, "r", encoding="utf-8") as f:
        prof = json.load(f)

    dist = np.array(prof["distance_miles"], dtype=float)
    elev = np.array(prof["elevation_feet"], dtype=float)
    total_miles = float(prof["total_miles"])
    duration_s  = float(prof.get("duration_s", 60.0))
    has_time    = bool(prof.get("has_time", False))
    time_s      = np.array(prof.get("time_s", []), dtype=float)
    moving_time_s = float(prof.get("moving_time_s", duration_s))
    avg_mph = total_miles / max(1e-6, (moving_time_s / 3600.0))

    # per-segment mph
    if has_time and len(time_s) == len(dist):
        seg_mph = np.maximum(0.0, np.diff(dist)) / np.maximum(1e-3, np.diff(time_s) / 3600.0)
    else:
        seg_mph = np.full(len(dist)-1, avg_mph, dtype=float)

    def idx_for_miles(mi):
        i = int(np.searchsorted(dist, mi)) - 1
        return max(0, min(i, len(dist)-2))

    def mph_at_miles(mi):
        i = idx_for_miles(mi)
        j0 = max(0, i-2); j1 = min(len(seg_mph)-1, i+2)
        return float(np.mean(seg_mph[j0:j1+1]))

    def elev_at_miles(mi):
        if mi <= dist[0]: return float(elev[0])
        if mi >= dist[-1]: return float(elev[-1])
        i = idx_for_miles(mi)
        t = (mi - dist[i]) / max(1e-9, (dist[i+1] - dist[i]))
        return float(elev[i] + t*(elev[i+1]-elev[i]))

    reader = imageio.get_reader(video_file)
    meta = reader.get_meta_data()
    fps = float(meta.get("fps", 30.0))

    # Peek first frame to get dimensions and build chart background once
    first = reader.get_next_data()
    H, W = first.shape[0], first.shape[1]
    reader.close()
    reader = imageio.get_reader(video_file)

    writer = imageio.get_writer(
        out_file, fps=fps, codec="libx264",
        macro_block_size=None, ffmpeg_log_level="warning",
        ffmpeg_params=["-pix_fmt", "yuv420p", "-movflags", "faststart"]
    )

    # Fonts
    try:
        font = ImageFont.truetype("Arial.ttf", 18)
        font_b = ImageFont.truetype("Arial Bold.ttf", 18)
    except Exception:
        font = ImageFont.load_default()
        font_b = font

    # Prepare mini chart background if needed
    mini_bg = None
    #mini_w = max(240, int(W * 0.28))
    #mini_h = max(100, int(H * 0.18))
    #pad = 10
    is_portrait = H > W
    if show_chart:
        if is_portrait:
            mini_w = W - 24                    # full-width (left↔right) with 12px margins
            mini_h = max(140, int(H * 0.22))   # ~22% of height
        else:
            mini_w = max(240, int(W * 0.28))   # the old compact size on landscape
            mini_h = max(100, int(H * 0.18))
        pad = 12

        y_min = float(np.min(elev))
        y_max = float(np.max(elev))
        #if y_max <= y_min:
        #    y_max = y_min + 1.0

        if y_max <= y_min: y_max = y_min + 1.0

        mini_bg = Image.new("RGBA", (mini_w, mini_h), (0,0,0,0))
        dr = ImageDraw.Draw(mini_bg, "RGBA")
        # rounded panel
        dr.rounded_rectangle((0,0,mini_w,mini_h), radius=14, fill=(0,0,0,140))
        # plot line
        w = mini_w - 2*pad
        h = mini_h - 2*pad
        pts = []
        for i in range(len(dist)):
            x = pad + (dist[i] / total_miles) * w
            y = pad + (1.0 - (elev[i] - y_min) / (y_max - y_min)) * h
            pts.append((x, y))
        if len(pts) >= 2:
            dr.line(pts, fill=(144,202,249,255), width=3 if is_portrait else 2)

    # Drawing helpers
    def draw_stats(draw, panel_origin, mph_now, elev_now, miles_now):
        x0, y0 = panel_origin
        panel_w, panel_h = 240, 94
        draw.rounded_rectangle((x0, y0, x0+panel_w, y0+panel_h), radius=14, fill=(0,0,0,140))
        def txt(x, y, s, bold=False):
            draw.text((x, y), s, fill=(255,255,255,210), font=(font_b if bold else font))
        txt(x0+12, y0+10, "Speed");     txt(x0+140, y0+10, f"{mph_now:.1f} mph", True)
        txt(x0+12, y0+36, "Elevation"); txt(x0+140, y0+36, f"{elev_now:,.0f} ft", True)
        txt(x0+12, y0+62, "Distance");  txt(x0+140, y0+62, f"{miles_now:.2f} mi", True)

    for i, frame in enumerate(reader):
        t = i / fps
        miles = total_miles * (t / duration_s)
        mph_now  = mph_at_miles(miles)
        elev_now = elev_at_miles(miles)

        pil = Image.fromarray(frame).convert("RGBA")
        draw = ImageDraw.Draw(pil, "RGBA")

        if show_stats:
            draw_stats(draw, (12,12), mph_now, elev_now, miles)

        if show_chart and mini_bg is not None:
            chart_img = mini_bg.copy()
            d2 = ImageDraw.Draw(chart_img, "RGBA")
            # cursor (orange line)
            w = mini_w - 2*pad
            x = pad + (miles / total_miles) * w
            d2.line([(x, pad), (x, mini_h-pad)], fill=(255,109,0,255), width=2)
            # paste bottom-right
            pil.alpha_composite(chart_img, (W - mini_w - 12, H - mini_h - 12))

        writer.append_data(np.array(pil.convert("RGB")))

    reader.close()
    writer.close()


# Backward-compatible wrapper (kept if older code calls this)
def burn_hud_on_video(video_file: str, profile_json_file: str, out_file: str):
    burn_overlays_on_video(video_file, profile_json_file, out_file,
                           show_stats=True, show_chart=False)
