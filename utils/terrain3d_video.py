# utils/terrain3d_video.py
import io
import math
from pathlib import Path

import gpxpy
import gpxpy.gpx
import numpy as np
import requests
from PIL import Image

import matplotlib
matplotlib.use("Agg")
import imageio_ffmpeg
matplotlib.rcParams["animation.ffmpeg_path"] = imageio_ffmpeg.get_ffmpeg_exe()
from matplotlib import pyplot as plt
from matplotlib.animation import FFMpegWriter

TILE_SIZE = 256
DEM_TILE_URL = "https://s3.amazonaws.com/elevation-tiles-prod/terrarium/{z}/{x}/{y}.png"
HTTP_HEADERS = {"User-Agent": "GPXFlyover/0.1 (demo; contact: you@example.com)"}

class GPXParseError(Exception):
    pass

# ---------------- basic helpers ----------------

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

def _choose_zoom(lon: np.ndarray, lat: np.ndarray, max_mosaic_px: int):
    lat0 = float(np.median(lat))
    def mpp(z): return 156543.03392 * math.cos(math.radians(lat0)) / (2 ** z)
    dlon = float(np.max(lon) - np.min(lon))
    dlat = float(np.max(lat) - np.min(lat))
    span_m = max(dlon * 111320 * math.cos(math.radians(lat0)),
                 dlat * 110540)
    best = 12
    for z in range(6, 17):
        if span_m / mpp(z) <= max_mosaic_px:
            best = z
        else:
            break
    return max(6, min(best, 16))

def _tile_bbox_from_px(xg: np.ndarray, yg: np.ndarray, z: int, margin_tiles: int):
    min_tx = int(np.floor(xg.min() / TILE_SIZE)) - margin_tiles
    max_tx = int(np.floor(xg.max() / TILE_SIZE)) + margin_tiles
    min_ty = int(np.floor(yg.min() / TILE_SIZE)) - margin_tiles
    max_ty = int(np.floor(yg.max() / TILE_SIZE)) + margin_tiles
    n = (2 ** z) - 1
    min_tx = max(0, min_tx); min_ty = max(0, min_ty)
    max_tx = min(n, max_tx); max_ty = min(n, max_ty)
    return min_tx, min_ty, max_tx, max_ty

def _fetch_dem_tile(z: int, x: int, y: int, cache_dir: Path) -> np.ndarray:
    cache_path = cache_dir / f"{z}_{x}_{y}.png"
    if cache_path.exists():
        img = Image.open(cache_path).convert("RGB")
    else:
        url = DEM_TILE_URL.format(z=z, x=x, y=y)
        r = requests.get(url, headers=HTTP_HEADERS, timeout=20)
        r.raise_for_status()
        img = Image.open(io.BytesIO(r.content)).convert("RGB")
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        img.save(cache_path)
    arr = np.asarray(img, dtype=np.float32)
    R, G, B = arr[..., 0], arr[..., 1], arr[..., 2]
    elev = (R * 256.0 + G + B / 256.0) - 32768.0
    return elev  # meters

def _build_dem_mosaic(z: int, x0: int, y0: int, x1: int, y1: int, cache_dir: Path) -> np.ndarray:
    cols = x1 - x0 + 1
    rows = y1 - y0 + 1
    H = np.zeros((rows * TILE_SIZE, cols * TILE_SIZE), dtype=np.float32)
    for ty in range(y0, y1 + 1):
        for tx in range(x0, x1 + 1):
            tile = _fetch_dem_tile(z, tx, ty, cache_dir)
            r0 = (ty - y0) * TILE_SIZE
            c0 = (tx - x0) * TILE_SIZE
            H[r0:r0+TILE_SIZE, c0:c0+TILE_SIZE] = tile
    return H

def _resample_along_distance(x: np.ndarray, y: np.ndarray, n: int):
    dx = np.diff(x); dy = np.diff(y)
    d = np.hypot(dx, dy)
    cum = np.concatenate([[0.0], np.cumsum(d)])
    if cum[-1] <= 0:
        return x, y
    new_cum = np.linspace(0.0, float(cum[-1]), n)
    xn = np.interp(new_cum, cum, x)
    yn = np.interp(new_cum, cum, y)
    return xn, yn

def _bilinear(Z: np.ndarray, x: np.ndarray, y: np.ndarray):
    h, w = Z.shape
    x0 = np.clip(np.floor(x).astype(int), 0, w-2)
    y0 = np.clip(np.floor(y).astype(int), 0, h-2)
    x1 = x0 + 1
    y1 = y0 + 1
    wx = x - x0; wy = y - y0
    Ia = Z[y0, x0]; Ib = Z[y0, x1]; Ic = Z[y1, x0]; Id = Z[y1, x1]
    return (Ia * (1-wx) * (1-wy) +
            Ib * wx * (1-wy) +
            Ic * (1-wx) * wy +
            Id * wx * wy)

def _heading_deg(x, y, i, k=3):
    i0 = max(0, i-k); i1 = min(len(x)-1, i+k)
    dx = x[i1] - x[i0]; dy = y[i1] - y[i0]
    if dx == 0 and dy == 0: return 0.0
    return math.degrees(math.atan2(dy, dx))

# ---------------- main (FAST) ----------------

def generate_terrain3d_video(
    gpx_file: str,
    out_file: str,
    *,
    fps: int = 24,              # ↓ fewer frames/sec
    target_duration: int = 25,  # ↓ shorter video
    grid_max: int = 320,        # ↓ DEM grid target (max dimension)
    render_stride: int = 2,     # ↓ skip rows/cols when drawing surface
    max_mosaic_px: int = 2048,  # ↓ choose lower zoom → fewer tiles
    dem_margin_tiles: int = 0,  # ↓ less DEM padding
    follow_pad_px: int = 500,
    elevate_track_m: float = 8,
    dpi: int = 100,             # ↓ lower figure DPI
    ffmpeg_preset: str = "veryfast",
    crf: int = 23,
):
    lon, lat = _parse_gpx_lonlat(gpx_file)

    # choose a lower zoom so the DEM mosaic stays small
    z = _choose_zoom(lon, lat, max_mosaic_px=max_mosaic_px)
    xg, yg = _lonlat_to_global_px(lon, lat, z)

    # fewer frames
    frames = max(int(fps * target_duration), 60)
    xg, yg = _resample_along_distance(xg, yg, frames)

    # small DEM mosaic
    tx0, ty0, tx1, ty1 = _tile_bbox_from_px(xg, yg, z, margin_tiles=dem_margin_tiles)
    H = _build_dem_mosaic(z, tx0, ty0, tx1, ty1, cache_dir=Path("tile_cache_elev"))
    origin_x = tx0 * TILE_SIZE; origin_y = ty0 * TILE_SIZE

    px = xg - origin_x; py = yg - origin_y
    z_path = _bilinear(H, px, py) + float(elevate_track_m)

    # downsample DEM to grid_max, then stride for render
    mosaic_h, mosaic_w = H.shape
    step = max(1, int(math.ceil(max(mosaic_h, mosaic_w) / float(grid_max))))
    Hs = H[::step, ::step]
    ys = np.arange(0, mosaic_h, step, dtype=float)
    xs = np.arange(0, mosaic_w, step, dtype=float)

    if render_stride > 1:
        Hs = Hs[::render_stride, ::render_stride]
        ys = ys[::render_stride]
        xs = xs[::render_stride]

    Xs, Ys = np.meshgrid(xs, ys)

    # figure
    fig = plt.figure(figsize=(10, 6), dpi=dpi)
    ax = fig.add_subplot(111, projection="3d")
    ax.set_xlabel("X (px)"); ax.set_ylabel("Y (px)"); ax.set_zlabel("Elevation (m)")
    ax.grid(True)

    # draw coarser surface once
    surf = ax.plot_surface(Xs, Ys, Hs, cmap="terrain", linewidth=0, antialiased=False)
    path_line, = ax.plot([], [], [], color="#1976d2", linewidth=2.0)

    zmin = float(np.nanmin(H)); zmax = float(np.nanmax(H))
    ax.set_zlim(zmin, zmax + 50)

    def update(i):
        path_line.set_data(px[:i+1], py[:i+1])
        path_line.set_3d_properties(z_path[:i+1])
        cx, cy = px[i], py[i]
        ax.set_xlim(cx - follow_pad_px, cx + follow_pad_px)
        ax.set_ylim(cy - follow_pad_px, cy + follow_pad_px)
        az = -_heading_deg(px, py, i) + 90.0
        ax.view_init(elev=35, azim=az)
        return path_line, surf

    # faster encoder settings
    writer = FFMpegWriter(
        fps=fps,
        codec="libx264",
        extra_args=["-pix_fmt", "yuv420p", "-movflags", "faststart",
                    "-preset", ffmpeg_preset, "-crf", str(crf)],
        metadata=dict(artist="GPX Flyover (3D Terrain, fast)"),
    )

    out = Path(out_file); out.parent.mkdir(parents=True, exist_ok=True)
    print(f"[3D] zoom={z} tiles={(tx1-tx0+1)}x{(ty1-ty0+1)} "
          f"mosaic={mosaic_w}x{mosaic_h} grid={Hs.shape} frames={frames}")

    with writer.saving(fig, out.as_posix(), dpi=dpi):
        for i in range(frames):
            update(i)
            writer.grab_frame()
    plt.close(fig)

    return {"frames": frames, "fps": fps, "duration_s": frames/float(fps)}
