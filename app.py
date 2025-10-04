import os
import io
import csv
import uuid
import json
import math
import time
import zipfile
import gpxpy
import gpxpy.gpx
import requests
import numpy as np
import matplotlib
matplotlib.use("Agg")  # headless
import matplotlib.pyplot as plt
from pathlib import Path
from PIL import Image, ImageDraw
from tempfile import NamedTemporaryFile
from werkzeug.utils import secure_filename
from utils.map_video import generate_map_flyover_video, burn_overlays_on_video
from flask import (
    Flask, render_template, request, redirect,
    url_for, flash, send_file, abort, url_for
)

# =============================================================================
# App & Paths
# =============================================================================
app = Flask(__name__)
app.secret_key = os.environ.get("FLASK_SECRET_KEY", "dev_secret_key")

BASE_DIR    = Path(__file__).resolve().parent
UPLOAD_DIR  = BASE_DIR / "uploads"
ELEVATION_DIR = BASE_DIR / "static" / "elevation"
VIDEO_DIR   = BASE_DIR / "static" / "videos"
TRACK_DIR   = BASE_DIR / "static" / "tracks"
PROFILE_DIR = BASE_DIR / "static" / "profiles"
TILE_CACHE  = BASE_DIR / "tile_cache_fullview"   

for d in (UPLOAD_DIR, ELEVATION_DIR, VIDEO_DIR, TRACK_DIR, PROFILE_DIR, TILE_CACHE):
    d.mkdir(parents=True, exist_ok=True)

MAPTILER_KEY = os.environ.get("MAPTILER_KEY", "YOUR_MAPTILER_KEY")
ALLOWED_EXTS = {".gpx", ".fit", ".tcx", ".kml"}

# =============================================================================
# GPX profile helpers (used for HUD/chart & overlays)
# =============================================================================
SIMPLIFY_EPS_M        = 8.0
ELEV_SMOOTH_WIN       = 21
ELEV_RESAMPLE_STEP_M  = 5.0
ELEV_SMOOTH_OVER_M    = 120.0
ELEV_GAIN_THRESH_M    = 0.0
STOP_SPEED_MPS        = 0.5

def _haversine_m(lat1, lon1, lat2, lon2):
    from math import radians, sin, cos, asin, sqrt
    R = 6371000.0
    dlat = radians(lat2 - lat1)
    dlon = radians(lon2 - lon1)
    a = sin(dlat/2)**2 + cos(radians(lat1))*cos(radians(lat2))*sin(dlon/2)**2
    return 2 * R * asin(sqrt(a))

def _moving_average(vals, win):
    v = np.asarray(vals, dtype=float)
    w = max(3, int(win))
    if w % 2 == 0:
        w += 1
    k = np.ones(w) / w
    return np.convolve(v, k, mode="same")

def _to_local_xy(lon, lat):
    lat0 = float(lat[len(lat)//2])
    x = (np.asarray(lon) * math.cos(math.radians(lat0))) * 111320.0
    y = (np.asarray(lat)) * 110540.0
    return x, y

def _simplify_track(lon, lat, ele, tim, eps_m=SIMPLIFY_EPS_M):
    x, y = _to_local_xy(lon, lat)
    keep = [0]; last = 0
    for i in range(1, len(x) - 1):
        if math.hypot(x[i] - x[last], y[i] - y[last]) >= eps_m:
            keep.append(i); last = i
    if keep[-1] != len(x) - 1:
        keep.append(len(x) - 1)
    keep = np.array(keep, dtype=int)
    return lon[keep], lat[keep], (ele[keep] if ele is not None else None), (tim[keep] if tim is not None else None)

def _robust_elev_gain(dist_m, elev_m,
                      step_m=ELEV_RESAMPLE_STEP_M,
                      smooth_over_m=ELEV_SMOOTH_OVER_M,
                      thresh_m=ELEV_GAIN_THRESH_M):
    grid = np.arange(0.0, float(dist_m[-1]) + step_m, step_m)
    e_grid = np.interp(grid, dist_m, elev_m)
    win = max(3, int(round(smooth_over_m / step_m)))
    if win % 2 == 0: win += 1
    try:
        from scipy.signal import savgol_filter
        e_smooth = savgol_filter(e_grid, window_length=win, polyorder=2, mode="interp")
    except Exception:
        kernel = np.ones(win) / win
        e_smooth = np.convolve(e_grid, kernel, mode="same")
    deltas = np.diff(e_smooth)
    pos = np.where(deltas > thresh_m, deltas, 0.0)
    cum_grid = np.concatenate([[0.0], np.cumsum(pos)])
    total_gain_m = float(cum_grid[-1])
    cum_on_orig = np.interp(dist_m, grid, cum_grid)
    return total_gain_m, cum_on_orig

def _gpx_to_profile(gpx_path: Path, json_path: Path) -> dict:
    lons, lats, elevs, times = [], [], [], []
    with open(gpx_path, "r", encoding="utf-8") as f:
        gpx = gpxpy.parse(f)

    for trk in gpx.tracks:
        for seg in trk.segments:
            for p in seg.points:
                if p.longitude is None or p.latitude is None:
                    continue
                lons.append(float(p.longitude))
                lats.append(float(p.latitude))
                elevs.append(float(p.elevation) if p.elevation is not None else 0.0)
                times.append(p.time)

    if len(lons) < 2:
        raise ValueError("Not enough GPX points.")

    lons = np.array(lons); lats = np.array(lats)
    elevs = np.array(elevs); times = np.array(times, dtype=object)
    lons, lats, elevs, times = _simplify_track(lons, lats, elevs, times)

    dist_m = [0.0]
    for i in range(1, len(lons)):
        dist_m.append(dist_m[-1] + _haversine_m(lats[i-1], lons[i-1], lats[i], lons[i]))
    dist_m = np.array(dist_m)

    to_mi = 1 / 1609.344
    distance_miles = (dist_m * to_mi).tolist()

    elev_smooth_m = _moving_average(elevs, ELEV_SMOOTH_WIN)
    total_gain_m, cum_gain_on_orig_m = _robust_elev_gain(dist_m, elev_smooth_m)

    to_ft = 3.280839895
    elevation_feet   = (elev_smooth_m * to_ft).tolist()
    cum_gain_feet    = (cum_gain_on_orig_m * to_ft).tolist()
    total_gain_feet  = total_gain_m * to_ft

    has_time = all(t is not None for t in times)
    time_s = []
    moving_time_s = 0.0
    if has_time:
        t0 = times[0]
        for i in range(len(times)):
            time_s.append((times[i] - t0).total_seconds())
        for i in range(1, len(times)):
            dt = (times[i] - times[i-1]).total_seconds()
            if dt <= 0: continue
            d_m = dist_m[i] - dist_m[i-1]
            v = d_m / dt
            if v >= STOP_SPEED_MPS:
                moving_time_s += dt

    data = {
        "distance_miles": distance_miles,
        "elevation_feet": elevation_feet,
        "cum_gain_feet": cum_gain_feet,
        "total_gain_feet": total_gain_feet,
        "time_s": time_s,
        "has_time": has_time,
        "moving_time_s": moving_time_s if has_time else None,
        "total_time_s": (time_s[-1] if has_time else None),
        "total_miles": distance_miles[-1],
    }
    json_path.write_text(json.dumps(data), encoding="utf-8")
    return data


# =============================================================================
# Helpers to render a static "full-route" snapshot (OSM tiles)
# =============================================================================
TILE_SIZE     = 256
OSM_TILE_URL  = "https://tile.openstreetmap.org/{z}/{x}/{y}.png"
HTTP_HEADERS  = {"User-Agent": "Plactiv/1.0 (GPX fullview)"}

def _lonlat_to_global_px(lon, lat, z):
    lon = np.asarray(lon, dtype=float); lat = np.asarray(lat, dtype=float)
    n = 2.0 ** z
    x = (lon + 180.0) / 360.0 * n * TILE_SIZE
    lat_rad = np.radians(lat)
    y = (1.0 - np.log(np.tan(lat_rad) + 1.0/np.cos(lat_rad)) / math.pi) / 2.0 * n * TILE_SIZE
    return x, y

def _fetch_tile(z, x, y, cache_dir: Path) -> Image.Image:
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

def _build_mosaic(z, x0, y0, x1, y1, cache_dir: Path) -> Image.Image:
    cols = x1 - x0 + 1
    rows = y1 - y0 + 1
    mosaic = Image.new("RGB", (cols * TILE_SIZE, rows * TILE_SIZE))
    for ty in range(y0, y1 + 1):
        for tx in range(x0, x1 + 1):
            tile = _fetch_tile(z, tx, ty, cache_dir)
            mosaic.paste(tile, ((tx - x0) * TILE_SIZE, (ty - y0) * TILE_SIZE))
    return mosaic

def _choose_zoom_to_fit(lon, lat, target_w, target_h, pad_ratio=0.08, max_z=18):
    """Pick a web-mercator zoom so the full bounds fit target_w×target_h (with padding)."""
    lon = np.asarray(lon); lat = np.asarray(lat)
    lon_min, lon_max = float(np.min(lon)), float(np.max(lon))
    lat_min, lat_max = float(np.min(lat)), float(np.max(lat))

    # try from high zoom down until fits
    for z in range(max_z, 0, -1):
        x0, y1 = _lonlat_to_global_px(lon_min, lat_min, z)
        x1, y0 = _lonlat_to_global_px(lon_max, lat_max, z)
        w = abs(x1 - x0); h = abs(y1 - y0)
        w *= (1 + pad_ratio*2); h *= (1 + pad_ratio*2)
        if w <= target_w and h <= target_h:
            return z
    return 10  # fallback

def render_full_route_snapshot(gpx_path: Path, out_png: Path, width: int, height: int) -> bool:
    """Render a full-path PNG (width×height) and return True on success."""
    try:
        lons, lats = [], []
        with open(gpx_path, "r", encoding="utf-8") as f:
            gpx = gpxpy.parse(f)
        for trk in gpx.tracks:
            for seg in trk.segments:
                for p in seg.points:
                    if p.longitude is None or p.latitude is None:
                        continue
                    lons.append(float(p.longitude))
                    lats.append(float(p.latitude))
        if len(lons) < 2:
            return False

        z = _choose_zoom_to_fit(lons, lats, width, height, pad_ratio=0.10, max_z=18)
        xg, yg = _lonlat_to_global_px(lons, lats, z)

        # compute tile range that covers the whole track + padding
        pad_px_w = width * 0.10
        pad_px_h = height * 0.10
        min_px = float(np.min(xg) - pad_px_w); max_px = float(np.max(xg) + pad_px_w)
        min_py = float(np.min(yg) - pad_px_h); max_py = float(np.max(yg) + pad_px_h)

        min_tx = int(math.floor(min_px / TILE_SIZE)) - 1
        max_tx = int(math.floor(max_px / TILE_SIZE)) + 1
        min_ty = int(math.floor(min_py / TILE_SIZE)) - 1
        max_ty = int(math.floor(max_py / TILE_SIZE)) + 1

        mosaic = _build_mosaic(z, min_tx, min_ty, max_tx, max_ty, TILE_CACHE)

        # project into mosaic coords and draw the full path
        px = xg - min_tx * TILE_SIZE
        py = yg - min_ty * TILE_SIZE

        # scale/crop to requested aspect
        mos_w, mos_h = mosaic.size
        aspect = width / float(height)
        if mos_w / mos_h > aspect:
            # crop width
            new_w = int(round(mos_h * aspect))
            left = int(round(max(0, np.mean(px) - new_w / 2)))
            mosaic = mosaic.crop((left, 0, left + new_w, mos_h))
            px -= left
        else:
            # crop height
            new_h = int(round(mos_w / aspect))
            top = int(round(max(0, np.mean(py) - new_h / 2)))
            mosaic = mosaic.crop((0, top, mos_w, top + new_h))
            py -= top

        mosaic = mosaic.resize((width, height), Image.LANCZOS)
        # rescale coords to resized image
        scale_x = width  / float(mosaic.size[0])
        scale_y = height / float(mosaic.size[1])
        px *= scale_x
        py *= scale_y

        draw = ImageDraw.Draw(mosaic)
        coords = list(zip(px.tolist(), py.tolist()))
        if len(coords) >= 2:
            draw.line(coords, fill=(255, 64, 64), width=5)
            # start/end dots
            r = 6
            sx, sy = coords[0]; ex, ey = coords[-1]
            draw.ellipse((sx-r, sy-r, sx+r, sy+r), outline=(0,0,0), fill=(0,255,0))
            draw.ellipse((ex-r, ey-r, ex+r, ey+r), outline=(0,0,0), fill=(255,0,0))

        out_png.parent.mkdir(parents=True, exist_ok=True)
        mosaic.save(out_png.as_posix())
        return True
    except Exception as e:
        print(f"[WARN] render_full_route_snapshot failed: {e}")
        return False

# =============================================================================
# Tail builders (append to 2D MP4)
# =============================================================================
def append_fullview_tail_to_video(video_file: str, full_png: str, out_file: str,
                                  tail_seconds: float = 2.5):
    """Append the full-route PNG as a short tail to the end of `video_file`."""
    try:
        from moviepy.editor import VideoFileClip, ImageClip, concatenate_videoclips
        base = VideoFileClip(video_file)
        tail = ImageClip(full_png).set_duration(tail_seconds).fadein(0.15).fadeout(0.15)
        final = concatenate_videoclips([base, tail], method="compose")

        with NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
            tmp_name = tmp.name
        final.write_videofile(tmp_name, codec="libx264", audio=False,
                              fps=int(round(base.fps or 30)),
                              preset="medium", threads=os.cpu_count() or 2)
        final.close(); base.close()
        os.replace(tmp_name, out_file)
    except Exception as e:
        print(f"[WARN] fullview tail failed, falling back to zoom-out: {e}")
        add_finish_zoomout_tail(video_file, out_file, tail_seconds=tail_seconds, zoom_start=1.18)

def add_finish_zoomout_tail(video_file: str, out_file: str,
                            tail_seconds: float = 2.5, zoom_start: float = 1.18):
    """Fallback: freeze last frame and smoothly zoom out."""
    try:
        from moviepy.editor import VideoFileClip, ImageClip, concatenate_videoclips, vfx
        base = VideoFileClip(video_file)
        fps  = int(round(base.fps or 30))
        last_t = max(0.0, base.duration - (1.0 / max(1, fps)))
        last_frame = base.get_frame(last_t)
        tail = (ImageClip(last_frame)
                .set_duration(tail_seconds)
                .fx(vfx.resize, lambda t: zoom_start - (zoom_start - 1.0) * (t / max(1e-6, tail_seconds)))
                .fadein(0.15).fadeout(0.15))
        final = concatenate_videoclips([base, tail], method="compose")
        with NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
            tmp_name = tmp.name
        final.write_videofile(tmp_name, codec="libx264", audio=False, fps=fps,
                              preset="medium", threads=os.cpu_count() or 2)
        final.close(); base.close()
        os.replace(tmp_name, out_file)
    except Exception as e:
        print(f"[WARN] add_finish_zoomout_tail failed: {e}")

# =============================================================================
# Routes
# =============================================================================
@app.route("/", methods=["GET"])
def landing():
    """Landing page (hero + product cards + contact)."""
    return render_template("landing.html")

'''
@app.route("/elevation")
def elevation():
    return render_template("elevation.html")
'''

@app.route("/video_uploads", methods=["GET"])
def video_uploads():
    """Upload page for GPX to video."""
    return render_template("video_uploads.html")

@app.route("/planner")
def planner():
    return render_template("planner.html")

@app.route("/contact", methods=["POST"], endpoint="contact")
def handle_contact():
    """Handle landing-page contact form."""
    name = (request.form.get("name") or "").strip()
    email = (request.form.get("email") or "").strip()
    message = (request.form.get("message") or "").strip()
    if not name or not email or not message:
        flash("Please fill out all fields.")
    else:
        flash("Thanks for your message — we’ll get back to you soon!")
    return redirect(url_for("landing"))

# The form in video_uploads.html should POST to url_for('upload')
@app.route("/video/upload", methods=["POST"])
def upload():
    if "gpxfile" not in request.files:
        flash("No file part in the request.")
        return redirect(url_for("video_uploads"))

    f = request.files["gpxfile"]
    if not f or f.filename == "":
        flash("No file selected.")
        return redirect(url_for("video_uploads"))
    if not f.filename.lower().endswith(".gpx"):
        flash("Please upload a .gpx file.")
        return redirect(url_for("video_uploads"))

    mode    = request.form.get("mode", "map")      # "map" or "webgl3d"
    orient  = request.form.get("orientation", "h") # "h" or "v"

    uid = str(uuid.uuid4())
    gpx_path = UPLOAD_DIR / f"{uid}.gpx"
    f.save(gpx_path.as_posix())

    try:
        # ----- 3D (web) -----
        if mode == "webgl3d":
            lons, lats, eles = [], [], []
            with open(gpx_path, "r", encoding="utf-8") as fp:
                gpx = gpxpy.parse(fp)
            for trk in gpx.tracks:
                for seg in trk.segments:
                    for p in seg.points:
                        if p.longitude is None or p.latitude is None:
                            continue
                        lons.append(float(p.longitude))
                        lats.append(float(p.latitude))
                        eles.append(float(p.elevation) if p.elevation is not None else None)
            bounds = [min(lons), min(lats), max(lons), max(lats)]
            track_json = TRACK_DIR / f"{uid}.json"
            track_json.write_text(json.dumps({
                "type": "Feature",
                "geometry": {"type": "LineString", "coordinates": [[lo, la] for lo, la in zip(lons, lats)]},
                "properties": {"elev": eles, "bounds": bounds}
            }), encoding="utf-8")
            return redirect(url_for("video_result_3d", uid=uid))

        # ----- 2D MP4 -----
        if orient == "v":
            vw, vh = 720, 1280
            zoom_boost = 1
            follow_back = 250.0
        else:
            vw, vh = 1280, 720
            zoom_boost = 0
            follow_back = 350.0

        out_path = VIDEO_DIR / f"{uid}.mp4"
        meta = generate_map_flyover_video(
            gpx_file=gpx_path.as_posix(),
            out_file=out_path.as_posix(),
            fps=30,
            target_duration=60,          # 60-second total duration
            zoom_padding_ratio=0.15,     # padding for the “fit” zoom
            style="osm",
            width=vw, height=vh,         # orientation (horizontal/vertical)
            follow_back_meters=follow_back,
            zoom_boost=zoom_boost,
            add_tail=True,                 
            tail_seconds=2.5,            # zoom-out tail length
            start_zoom_override=15,       # force zoom level 4 at start
            end_zoom_override=4,         # force zoom level 0 at end
        )

        # Optional: profile JSON (used by HUD/chart overlays & downloads)
        profile_json = PROFILE_DIR / f"{uid}.json"
        try:
            prof = _gpx_to_profile(gpx_path, profile_json)
            prof["duration_s"] = float(meta.get("duration_s", (meta.get("frames", 0) / max(1, meta.get("fps", 30)))))
            profile_json.write_text(json.dumps(prof), encoding="utf-8")
        except Exception as e:
            print(f"[WARN] profile generation failed: {e}")
        return redirect(url_for("video_result_2d", vid=uid))

    except Exception as e:
        flash(f"Video generation failed: {e}")
        return redirect(url_for("video_uploads"))

# ----- Result pages (filenames exactly as requested) -----
@app.route("/video/result/<vid>", methods=["GET"], endpoint="video_result_2d")
def result_2d(vid):
    video_url   = url_for("static", filename=f"videos/{vid}.mp4")
    profile_url = url_for("static", filename=f"profiles/{vid}.json")
    return render_template("video_result_2d.html",
                           video_url=video_url, profile_url=profile_url, vid=vid)

@app.route("/video/3d/<uid>", methods=["GET"], endpoint="video_result_3d")
def result_3d(uid):
    track_url = url_for("static", filename=f"tracks/{uid}.json")
    return render_template("video_result_3d.html",
                           track_url=track_url, maptiler_key=MAPTILER_KEY)

# ----- Download with optional overlays -----
@app.route("/download/<vid>", methods=["GET"])
def download(vid):
    """
    Download the MP4.
      ?hud=1/0   -> burn stats HUD
      ?chart=1/0 -> burn mini elevation chart
    If both are 0, return the clean video as-is.
    """
    want_hud   = request.args.get("hud", "1") == "1"
    want_chart = request.args.get("chart", "1") == "1"

    base = VIDEO_DIR / f"{vid}.mp4"
    if not base.exists():
        abort(404)

    if not (want_hud or want_chart):
        return send_file(base.as_posix(), as_attachment=True, download_name="flyover.mp4")

    profile_json = PROFILE_DIR / f"{vid}.json"
    if not profile_json.exists():
        abort(404)

    suffix = ("_hud" if want_hud else "") + ("_chart" if want_chart else "")
    out = VIDEO_DIR / f"{vid}{suffix}.mp4"

    if (not out.exists()) or (out.stat().st_mtime < max(base.stat().st_mtime, profile_json.stat().st_mtime)):
        burn_overlays_on_video(
            video_file=str(base),
            profile_json_file=str(profile_json),
            out_file=str(out),
            show_stats=want_hud,
            show_chart=want_chart,
        )

    return send_file(out.as_posix(), as_attachment=True, download_name=f"flyover{suffix}.mp4")



# =========================
# Small helpers
# =========================
def _allowed_gpx(filename: str) -> bool:
    return Path(filename).suffix.lower() in ALLOWED_EXTS

def _moving_avg(arr: np.ndarray, w: int) -> np.ndarray:
    if w <= 1:
        return arr
    pad = w // 2
    arr_padded = np.pad(arr, (pad, pad), mode="edge")
    kernel = np.ones(w) / w
    return np.convolve(arr_padded, kernel, mode="valid")

def _haversine(lat1, lon1, lat2, lon2) -> float:
    """
    Returns distance in meters between two lat/lon points.
    """
    R = 6371000.0
    p1 = math.radians(lat1); p2 = math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlmb = math.radians(lon2 - lon1)
    a = math.sin(dphi/2)**2 + math.cos(p1) * math.cos(p2) * math.sin(dlmb/2)**2
    return 2 * R * math.asin(math.sqrt(a))

def _segment_difficulty(grade_pct: float) -> str:
    """
    Simple heuristic:
      uphill:    <3 Easy, 3-6 Moderate, 6-9 Hard, >=9 Very Hard
      downhill:  mirror with 'Down' prefix
    """
    if grade_pct >= 0:
        if grade_pct < 3:   return "Easy"
        if grade_pct < 6:   return "Moderate"
        if grade_pct < 9:   return "Hard"
        return "Very Hard"
    else:
        g = abs(grade_pct)
        if g < 3:   return "Easy Down"
        if g < 6:   return "Moderate Down"
        if g < 9:   return "Hard Down"
        return "Very Hard Down"

def _units_convert(dist_m, ele_m, units: str):
    if units == "metric":
        return dist_m / 1000.0, ele_m  # km, m
    else:
        # mi, ft
        return dist_m / 1609.344, ele_m * 3.28084

def _units_labels(units: str):
    return ("Distance (km)", "Elevation (m)") if units == "metric" else ("Distance (mi)", "Elevation (ft)")

# =========================
# Core GPX → arrays
# =========================
def extract_track_arrays(gpx_bytes: bytes, smooth_window: int, units: str):
    """
    Parse GPX and return:
      distance_arr (monotonic along track), elevation_arr (same length),
      grade_pct_arr (len-1 for segments, we’ll prepend 0 to align)
    """
    gpx = gpxpy.parse(io.BytesIO(gpx_bytes))
    pts = []
    for track in gpx.tracks:
        for seg in track.segments:
            for p in seg.points:
                if p.latitude is not None and p.longitude is not None and p.elevation is not None:
                    pts.append((p.latitude, p.longitude, float(p.elevation)))
    if len(pts) < 2:
        raise ValueError("Not enough points with elevation in GPX.")

    lats = np.array([p[0] for p in pts], dtype=float)
    lons = np.array([p[1] for p in pts], dtype=float)
    eles_m = np.array([p[2] for p in pts], dtype=float)

    # cumulative distance in meters
    seg_d = np.zeros(len(lats)-1, dtype=float)
    for i in range(len(seg_d)):
        seg_d[i] = _haversine(lats[i], lons[i], lats[i+1], lons[i+1])
    dist_m = np.concatenate([[0.0], np.cumsum(seg_d)])

    # smoothing elevations (optional)
    eles_m_s = _moving_avg(eles_m, smooth_window)

    # grade (%) per segment
    # avoid divide-by-zero on tiny horizontal distance
    grade_pct = np.zeros(len(seg_d), dtype=float)
    tiny = 1e-6
    d_ele = np.diff(eles_m_s)
    grade_pct = 100.0 * d_ele / np.maximum(seg_d, tiny)

    # convert to requested units for plotting/export
    dist_u, ele_u = _units_convert(dist_m, eles_m_s, units)
    # align grade array with point arrays by prepending 0 for the first point
    grade_u = np.concatenate([[0.0], grade_pct])

    return dist_u, ele_u, grade_u

# =========================
# Plot + CSV writers
# =========================
def write_elevation_png(out_path: Path, title: str, dist_u: np.ndarray, ele_u: np.ndarray, units: str):
    xlab, ylab = _units_labels(units)
    plt.figure(figsize=(10, 4.2), dpi=130)
    plt.plot(dist_u, ele_u, linewidth=1.7)
    plt.title(title)
    plt.xlabel(xlab); plt.ylabel(ylab)
    plt.grid(True, linestyle="--", alpha=0.35)
    plt.tight_layout()
    plt.savefig(out_path.as_posix())
    plt.close()

def write_detail_csv(out_path: Path, base_name: str, dist_u: np.ndarray, ele_u: np.ndarray, grade_u: np.ndarray, units: str):
    """
    Per-point CSV: index, distance, elevation, grade %, difficulty
    """
    xlab, ylab = _units_labels(units)
    with out_path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["file", "point_index", xlab, ylab, "Grade (%)", "Segment Difficulty"])
        for i in range(len(dist_u)):
            diff = _segment_difficulty(grade_u[i])
            w.writerow([base_name, i, f"{dist_u[i]:.5f}", f"{ele_u[i]:.3f}", f"{grade_u[i]:.2f}", diff])

def summarize_track(base_name: str, dist_u: np.ndarray, ele_u: np.ndarray, units: str) -> dict:
    gain = np.sum(np.clip(np.diff(ele_u), 0, None))
    loss = -np.sum(np.clip(np.diff(ele_u), None, 0))
    return {
        "file": base_name,
        "total_distance": float(dist_u[-1]),
        "min_elevation": float(np.min(ele_u)),
        "max_elevation": float(np.max(ele_u)),
        "elevation_gain": float(gain),
        "elevation_loss": float(loss),
        "units": "metric" if units == "metric" else "imperial"
    }

# =========================
# Flask route
# =========================
@app.route("/elevation", methods=["GET", "POST"])
def elevation():
    if request.method == "GET":
        return render_template("elevation.html")

    files = request.files.getlist("gpx_files")
    if not files:
        return render_template("elevation.html", error="Please choose at least one GPX file.")
    smooth_window = max(int(request.form.get("smooth_window", 5) or 5), 1)
    units = (request.form.get("units") or "imperial").strip().lower()
    want_combined = bool(request.form.get("make_combined_plot"))

    # per-request working dir
    ts = time.strftime("%Y%m%d-%H%M%S")
    work_dir = ELEVATION_DIR / f"elev-{ts}"
    work_dir.mkdir(parents=True, exist_ok=True)

    combined_curves = []  # (title, dist_u, ele_u)
    summary_rows = []

    # Process each GPX
    for f in files:
        if not f or not f.filename:
            continue
        if not _allowed_gpx(f.filename):
            continue

        safe_base = secure_filename(Path(f.filename).stem) or "track"
        gpx_bytes = f.read()

        try:
            dist_u, ele_u, grade_u = extract_track_arrays(gpx_bytes, smooth_window, units)
        except Exception as e:
            # record an error file so user knows which failed
            (work_dir / f"{safe_base}__ERROR.txt").write_text(str(e))
            continue

        # PNG
        png_path = work_dir / f"{safe_base}_elevation.png"
        write_elevation_png(
            png_path,
            title=f"Elevation Profile - {safe_base}",
            dist_u=dist_u,
            ele_u=ele_u,
            units=units
        )

        # CSV
        csv_path = work_dir / f"{safe_base}.csv"
        write_detail_csv(csv_path, safe_base, dist_u, ele_u, grade_u, units)

        # combined plot data & summary
        combined_curves.append((safe_base, dist_u, ele_u))
        summary_rows.append(summarize_track(safe_base, dist_u, ele_u, units))

    # combined summary CSV
    if summary_rows:
        sum_path = work_dir / "combined_summary.csv"
        keys = ["file", "total_distance", "min_elevation", "max_elevation", "elevation_gain", "elevation_loss", "units"]
        with sum_path.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            for row in summary_rows:
                w.writerow(row)

    # Optional combined plot
    if want_combined and len(combined_curves) >= 2:
        plt.figure(figsize=(10, 4.2), dpi=130)
        for (title, dist_u, ele_u) in combined_curves:
            plt.plot(dist_u, ele_u, linewidth=1.4, label=title)
        xlab, ylab = _units_labels(units)
        plt.title("Elevation Profiles (Combined)")
        plt.xlabel(xlab); plt.ylabel(ylab)
        plt.grid(True, linestyle="--", alpha=0.35)
        plt.legend()
        plt.tight_layout()
        plt.savefig((work_dir / "combined_plot.png").as_posix())
        plt.close()

    # Package everything into a ZIP to auto-download
    zip_name = f"elevation_outputs_{ts}.zip"
    zip_path = work_dir / zip_name
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as z:
        for p in work_dir.iterdir():
            if p.name == zip_name:
                continue
            z.write(p, arcname=p.name)

    # Return as file download
    return send_file(
        zip_path,
        as_attachment=True,
        download_name=zip_name,
        mimetype="application/zip"
    )



if __name__ == "__main__":
    app.run(debug=True)
