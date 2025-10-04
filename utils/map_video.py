from __future__ import annotations

import io
import os
import math
import json
import gpxpy
import requests
import numpy as np
import imageio.v3 as iio

from pathlib import Path
from typing import Tuple, Optional
from PIL import Image, ImageDraw
from moviepy.editor import VideoFileClip, VideoClip

# =============================================================================
# Tunables
# =============================================================================
TILE_SIZE = 256
OSM_TILE_URL = "https://tile.openstreetmap.org/{z}/{x}/{y}.png"
HTTP_HEADERS = {"User-Agent": "Plactiv/1.0 (GPX flyover)"}

ROUTE_COLOR = (0, 120, 255)   
ROUTE_WIDTH = 6
DOT_COLOR   = (255, 60, 60)
DOT_RADIUS  = 6

DEFAULT_PADDING_RATIO = 0.15
DEFAULT_FPS = 30
DEFAULT_SPEED_MPS = 4.0       
DEFAULT_TAIL_SECONDS = 2.5    

# =============================================================================
# Basic utils
# =============================================================================
def _haversine_m(lat1, lon1, lat2, lon2) -> float:
    from math import radians, sin, cos, asin, sqrt
    R = 6371000.0
    dlat = radians(lat2 - lat1)
    dlon = radians(lon2 - lon1)
    a = sin(dlat/2)**2 + cos(radians(lat1))*cos(radians(lat2))*sin(dlon/2)**2
    return 2 * R * asin(sqrt(a))

def _lonlat_to_global_px(lon, lat, z):
    lon = np.asarray(lon, dtype=float)
    lat = np.asarray(lat, dtype=float)
    n = 2.0 ** z
    x = (lon + 180.0) / 360.0 * n * TILE_SIZE
    lat_rad = np.radians(lat)
    y = (1.0 - np.log(np.tan(lat_rad) + 1.0/np.cos(lat_rad)) / math.pi) / 2.0 * n * TILE_SIZE
    return x, y

def _fetch_tile(z: int, x: int, y: int, cache_dir: Path) -> Image.Image:
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / f"{z}_{x}_{y}.png"
    if cache_path.exists():
        return Image.open(cache_path).convert("RGB")
    url = OSM_TILE_URL.format(z=z, x=x, y=y)
    r = requests.get(url, headers=HTTP_HEADERS, timeout=25)
    r.raise_for_status()
    img = Image.open(io.BytesIO(r.content)).convert("RGB")
    img.save(cache_path)
    return img

'''
def _build_mosaic(z: int, x0: int, y0: int, x1: int, y1: int, cache_dir: Path) -> Image.Image:
    cols = x1 - x0 + 1
    rows = y1 - y0 + 1
    mosaic = Image.new("RGB", (cols * TILE_SIZE, rows * TILE_SIZE))
    for ty in range(y0, y1 + 1):
        for tx in range(x0, x1 + 1):
            tile = _fetch_tile(z, tx, ty, cache_dir)
            mosaic.paste(tile, ((tx - x0) * TILE_SIZE, (ty - y0) * TILE_SIZE))
    return mosaic
'''

def _build_mosaic(z: int, x0: int, y0: int, x1: int, y1: int, cache_dir: Path) -> Image.Image:
    n = 2 ** z  # number of tiles along one axis at this zoom

    # Clamp Y to [0, n-1], wrap X into [0, n-1]
    def clamp_y(ty): return max(0, min(n - 1, ty))
    def wrap_x(tx):  return tx % n

    cols = x1 - x0 + 1
    rows = y1 - y0 + 1
    mosaic = Image.new("RGB", (cols * TILE_SIZE, rows * TILE_SIZE))

    for ty in range(y0, y1 + 1):
        cy = clamp_y(ty)
        for tx in range(x0, x1 + 1):
            cx = wrap_x(tx)
            try:
                tile = _fetch_tile(z, cx, cy, cache_dir)
            except Exception:
                # Fallback: solid gray tile to avoid jarring blue flash
                tile = Image.new("RGB", (TILE_SIZE, TILE_SIZE), (225, 228, 232))
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

def _heading_unit(x: np.ndarray, y: np.ndarray, i: int, k: int = 4):
    i0 = max(0, i - k); i1 = min(len(x) - 1, i + k)
    vx = x[i1] - x[i0]; vy = y[i1] - y[i0]
    n = math.hypot(vx, vy)
    if n == 0:
        return 0.0, -1.0
    return vx / n, vy / n

# =============================================================================
# GPX helpers
# =============================================================================
def _parse_gpx_lonlat(gpx_file: str):
    lons, lats, eles, times = [], [], [], []
    with open(gpx_file, "r", encoding="utf-8") as f:
        g = gpxpy.parse(f)
    for trk in g.tracks:
        for seg in trk.segments:
            for p in seg.points:
                if p.longitude is None or p.latitude is None:
                    continue
                lons.append(float(p.longitude))
                lats.append(float(p.latitude))
                eles.append(float(p.elevation) if p.elevation is not None else 0.0)
                times.append(p.time)
    if len(lons) < 2:
        raise ValueError("Not enough GPX points.")
    return np.array(lons), np.array(lats), np.array(eles), np.array(times, dtype=object)

def _pick_zoom_to_fit(lon, lat, width, height, pad_ratio=DEFAULT_PADDING_RATIO, max_z=18) -> int:
    for z in range(max_z, 1, -1):
        x, y = _lonlat_to_global_px(lon, lat, z)
        minx, maxx = np.min(x), np.max(x)
        miny, maxy = np.min(y), np.max(y)
        pad_w = width * pad_ratio
        pad_h = height * pad_ratio
        if (maxx - minx + 2*pad_w) <= width and (maxy - miny + 2*pad_h) <= height:
            return z
    return 9

# =============================================================================
# Drawing
# =============================================================================
def _draw_route_on_image(img: Image.Image, px: np.ndarray, py: np.ndarray,
                         color=ROUTE_COLOR, width=ROUTE_WIDTH):
    if len(px) < 2:
        return
    draw = ImageDraw.Draw(img)
    coords = list(zip(px.tolist(), py.tolist()))
    draw.line(coords, fill=color, width=width)

def _draw_dot(img: Image.Image, x: float, y: float, r: int = DOT_RADIUS, color=DOT_COLOR):
    draw = ImageDraw.Draw(img)
    draw.ellipse((x - r, y - r, x + r, y + r), fill=color)

# =============================================================================
# Main generator (MoviePy handles encoding; no brittle writer kwargs)
# =============================================================================
def generate_map_flyover_video(
    gpx_file: str,
    out_file: str,
    fps: int = DEFAULT_FPS,
    target_duration: Optional[float] = None,   # None -> natural / from GPX
    zoom_padding_ratio: float = DEFAULT_PADDING_RATIO,
    style: str = "osm",
    width: int = 1280,
    height: int = 720,
    follow_back_meters: float = 350.0,  # kept for API compat; not used when dot-locked
    zoom_boost: int = 0,
    add_tail: bool = True,
    tail_seconds: float = DEFAULT_TAIL_SECONDS,
    tile_cache_dir: Optional[str] = None,
    max_duration_s: Optional[float] = None,
    start_zoom_override: Optional[int] = 2,   
    end_fit_pad_ratio: Optional[float] = None,   
    end_zoom_override: Optional[int] = None,     
) -> dict:
    """
    Build a video where:
      1) The moving dot is fixed at the screen center (camera follows it).
      2) At the end, we ease to a zoomed-out, centered view of the full route.
    """
    # ---------------- Load GPX ----------------
    lon, lat, _eles, times = _parse_gpx_lonlat(gpx_file)

    # Zoom that fits full route (for the end shot) and main zoom (while following)
    '''
    z_full = _pick_zoom_to_fit(lon, lat, width, height,
                               pad_ratio=zoom_padding_ratio, max_z=18)
    z_main = min(18, z_full + (zoom_boost or 0))
    '''

    # END zoom: fit the full route (with optional different padding)
    end_pad = end_fit_pad_ratio if end_fit_pad_ratio is not None else zoom_padding_ratio
    z_full = _pick_zoom_to_fit(lon, lat, width, height, pad_ratio=end_pad, max_z=18)

    # START zoom: follow-cam zoom (override or boosted from z_full)
    if start_zoom_override is not None:
        z_main = int(start_zoom_override)
    else:
        z_main = min(18, z_full + (zoom_boost or 0))

    cache_dir = Path(tile_cache_dir or (Path(__file__).resolve().parent / ".." / "tile_cache")).resolve()
    cache_dir.mkdir(parents=True, exist_ok=True)

    # Precompute track in main zoom (global/web-mercator pixels)
    xg_main, yg_main = _lonlat_to_global_px(lon, lat, z_main)

    # ---------------- Duration ----------------
    if target_duration is None and times.size > 1 and times[0] is not None and times[-1] is not None:
        dur_main = (times[-1] - times[0]).total_seconds()
        if dur_main <= 0:
            dur_main = None
    else:
        dur_main = target_duration

    if dur_main is None:
        # fallback: distance / speed
        dist_m = 0.0
        for i in range(1, len(lon)):
            dist_m += _haversine_m(lat[i-1], lon[i-1], lat[i], lon[i])
        dur_main = max(30.0, dist_m / max(1e-6, DEFAULT_SPEED_MPS))

    if max_duration_s is not None:
        dur_main = min(dur_main, float(max_duration_s))

    frames_main = max(int(round(fps * float(dur_main))), fps)

    # Resample path to exactly frames_main points (for smoothness)
    xg, yg = _resample_along_distance(xg_main, yg_main, frames_main)

    half_w, half_h = width / 2.0, height / 2.0

    def render_centered_frame(cx: float, cy: float, z: int) -> Tuple[Image.Image, float, float]:
        """
        Render a frame whose *camera center* in global px is (cx,cy).
        Returns (frame, x0, y0) where (x0,y0) is the top-left of the frame in global px.
        """
        x0 = int(math.floor(cx - half_w))
        y0 = int(math.floor(cy - half_h))
        x1 = x0 + width
        y1 = y0 + height

        '''
        min_tx = int(math.floor(x0 / TILE_SIZE)) - 1
        max_tx = int(math.floor((x1 - 1) / TILE_SIZE)) + 1
        min_ty = int(math.floor(y0 / TILE_SIZE)) - 1
        max_ty = int(math.floor((y1 - 1) / TILE_SIZE)) + 1

        mosaic = _build_mosaic(z, min_tx, min_ty, max_tx, max_ty, cache_dir)
        mosaic_origin_x = min_tx * TILE_SIZE
        mosaic_origin_y = min_ty * TILE_SIZE
        '''
        
        n = 2 ** z
        min_tx = int(math.floor(x0 / TILE_SIZE)) - 1
        max_tx = int(math.floor((x1 - 1) / TILE_SIZE)) + 1
        min_ty = int(math.floor(y0 / TILE_SIZE)) - 1
        max_ty = int(math.floor((y1 - 1) / TILE_SIZE)) + 1

        # Clamp Y to [0, n-1]; leave X unbounded (we'll wrap per-tile in _build_mosaic)
        min_ty = max(0, min_ty)
        max_ty = min(n - 1, max_ty)

        mosaic = _build_mosaic(z, min_tx, min_ty, max_tx, max_ty, cache_dir)

        # Important: the origin must align with the *unclamped/unwrapped* min indices you passed in,
        # so the crop math stays consistent across frames.
        mosaic_origin_x = min_tx * TILE_SIZE
        mosaic_origin_y = min_ty * TILE_SIZE

        crop_x = x0 - mosaic_origin_x
        crop_y = y0 - mosaic_origin_y

        frame = mosaic.crop((crop_x, crop_y, crop_x + width, crop_y + height))
        return frame, float(x0), float(y0)

    # -------- Precompute for the end zoom-out (z_full) --------
    x_full, y_full = _lonlat_to_global_px(lon, lat, z_full)
    min_fx, max_fx = float(np.min(x_full)), float(np.max(x_full))
    min_fy, max_fy = float(np.min(y_full)), float(np.max(y_full))
    full_cx = (min_fx + max_fx) / 2.0
    full_cy = (min_fy + max_fy) / 2.0

    end_x_full, end_y_full = _lonlat_to_global_px([lon[-1]], [lat[-1]], z_full)
    end_x_full, end_y_full = float(end_x_full[0]), float(end_y_full[0])

    frames_tail = int(round(fps * (tail_seconds if add_tail else 0.0)))
    total_frames = frames_main + max(frames_tail, 1)  # keep >= 1 to allow last frame
    total_duration = total_frames / float(fps)

    # --------------- MoviePy frame fn ---------------
    def make_frame(t: float):
        i = int(np.clip(round(t * fps), 0, total_frames - 1))

        if i < frames_main:
            # ---------- MAIN: dot locked to center ----------
            # Camera center *is the dot position*
            cx = float(xg[i])
            cy = float(yg[i])

            frame_img, x0, y0 = render_centered_frame(cx, cy, z_main)

            # Draw path so far
            px = xg[:i + 1] - x0
            py = yg[:i + 1] - y0
            _draw_route_on_image(frame_img, px, py, ROUTE_COLOR, ROUTE_WIDTH)

            # Dot exactly at screen center
            _draw_dot(frame_img, half_w, half_h, r=DOT_RADIUS, color=DOT_COLOR)

        else:
            # ---------- END: ease center & zoom to full route ----------
            j = i - frames_main
            ttail = 1.0 if frames_tail <= 1 else (j / float(frames_tail - 1))
            # ease
            te = 4*ttail*ttail*ttail if ttail < 0.5 else 1 - pow(-2*ttail + 2, 3)/2

            # center: end point -> full bbox center (both at z_full)
            cx = (1 - te) * end_x_full + te * full_cx
            cy = (1 - te) * end_y_full + te * full_cy

            # zoom: z_main -> z_full (integer steps are fine)
            z_float = (1 - te) * z_main + te * z_full
            z = int(round(z_float))

            frame_img, x0, y0 = render_centered_frame(cx, cy, z)

            # draw the full route at the current zoom
            xt, yt = _lonlat_to_global_px(lon, lat, z)
            _draw_route_on_image(frame_img, xt - x0, yt - y0, ROUTE_COLOR, ROUTE_WIDTH)

            # (Optional) show the final dot during the zoom-out:
            dx, dy = _lonlat_to_global_px([lon[-1]], [lat[-1]], z)
            _draw_dot(frame_img, float(dx[0] - x0), float(dy[0] - y0), r=DOT_RADIUS, color=DOT_COLOR)

        return np.asarray(frame_img)

    clip = VideoClip(make_frame, duration=total_duration)
    clip.write_videofile(
        out_file,
        fps=fps,
        codec="libx264",
        audio=False,
        preset="medium",
        threads=os.cpu_count() or 2,
        ffmpeg_params=["-pix_fmt", "yuv420p"],
    )
    clip.close()

    return {
        "frames": total_frames,
        "fps": fps,
        "duration_s": total_duration,
    }

# =============================================================================
# Overlays pass used by /download
# =============================================================================
def burn_overlays_on_video(
    video_file: str,
    profile_json_file: str,
    out_file: str,
    show_stats: bool = True,
    show_chart: bool = True,  
):
    """
    Minimal HUD burner that draws a translucent panel with avg speed,
    total gain and distance using Pillow on each frame via MoviePy.
    """
    try:
        prof = json.loads(Path(profile_json_file).read_text(encoding="utf-8"))
    except Exception:
        prof = {}

    total_mi = float(prof.get("total_miles", 0.0) or 0.0)
    total_gain_ft = float(prof.get("total_gain_feet", 0.0) or 0.0)
    moving_time_s = float(prof.get("moving_time_s", prof.get("total_time_s", 0.0)) or 0.0)
    avg_mph = (total_mi / (moving_time_s / 3600.0)) if moving_time_s > 0 else 0.0

    if not (show_stats or show_chart):
        # simple copy-through in MoviePy
        clip = VideoFileClip(video_file)
        clip.write_videofile(
            out_file,
            codec="libx264",
            audio=False,
            preset="medium",
            threads=os.cpu_count() or 2,
            ffmpeg_params=["-pix_fmt", "yuv420p"],
        )
        clip.close()
        return

    def draw_hud(get_frame, t):
        frm = get_frame(t)
        img = Image.fromarray(frm).convert("RGB")
        d = ImageDraw.Draw(img, "RGBA")
        # panel
        panel_w, panel_h = 260, 90
        d.rectangle((12, 12, 12 + panel_w, 12 + panel_h), fill=(0, 0, 0, 140), outline=(255, 255, 255, 60))
        y = 22
        lines = [
            f"Avg speed: {avg_mph:.1f} mph",
            f"Elevation gain: {total_gain_ft:.0f} ft",
            f"Distance: {total_mi:.2f} mi",
        ] if show_stats else []
        for txt in lines:
            d.text((22, y), txt, fill=(255, 255, 255, 230))
            y += 24
        return np.asarray(img)

    base = VideoFileClip(video_file)
    hud = base.fl(draw_hud, apply_to=["mask", "video"])
    hud.write_videofile(
        out_file,
        codec="libx264",
        audio=False,
        preset="medium",
        threads=os.cpu_count() or 2,
        ffmpeg_params=["-pix_fmt", "yuv420p"],
    )
    base.close()
    hud.close()
