'''
import os
import uuid
import json
from pathlib import Path

from flask import Flask, render_template, request, redirect, url_for, flash

# 2D video generator
from utils.map_video import generate_map_flyover_video

app = Flask(__name__)
app.secret_key = os.environ.get("FLASK_SECRET_KEY", "dev_secret_key")

BASE_DIR   = Path(__file__).resolve().parent
UPLOAD_DIR = BASE_DIR / "uploads"
VIDEO_DIR  = BASE_DIR / "static" / "videos"
TRACK_DIR  = BASE_DIR / "static" / "tracks"   # for WebGL 3D
PROFILE_DIR= BASE_DIR / "static" / "profiles" # elevation profile for 2D page

for d in (UPLOAD_DIR, VIDEO_DIR, TRACK_DIR, PROFILE_DIR):
    d.mkdir(parents=True, exist_ok=True)

MAPTILER_KEY = os.environ.get("MAPTILER_KEY", "YOUR_MAPTILER_KEY")

VIDEO_W, VIDEO_H = 1280, 720


# ---------- Helpers ----------

def _gpx_to_track_json(gpx_path: Path, json_path: Path) -> None:
    """GPX → minimal GeoJSON-ish track for the WebGL 3D page."""
    import gpxpy
    lons, lats, eles, times = [], [], [], []
    with open(gpx_path, "r", encoding="utf-8") as f:
        gpx = gpxpy.parse(f)
    for trk in gpx.tracks:
        for seg in trk.segments:
            for p in seg.points:
                if p.longitude is None or p.latitude is None:
                    continue
                lons.append(float(p.longitude))
                lats.append(float(p.latitude))
                eles.append(float(p.elevation) if p.elevation is not None else None)
                times.append(p.time.isoformat() if p.time else None)
    if len(lons) < 2:
        raise ValueError("Not enough GPX points.")
    bounds = [min(lons), min(lats), max(lons), max(lats)]
    data = {
        "type": "Feature",
        "geometry": {"type": "LineString", "coordinates": [[lo, la] for lo, la in zip(lons, lats)]},
        "properties": {"elev": eles, "time": times, "bounds": bounds},
    }
    json_path.write_text(json.dumps(data), encoding="utf-8")


def _haversine_m(lat1, lon1, lat2, lon2):
    """Great-circle distance in meters."""
    from math import radians, sin, cos, asin, sqrt
    R = 6371000.0
    dlat = radians(lat2 - lat1)
    dlon = radians(lon2 - lon1)
    a = sin(dlat/2)**2 + cos(radians(lat1))*cos(radians(lat2))*sin(dlon/2)**2
    return 2 * R * asin(sqrt(a))

def _gpx_to_elev_profile(gpx_path: Path, json_path: Path) -> dict:
    """GPX → {distance_miles[], elevation_feet[], total_miles} for the 2D page."""
    import gpxpy
    lons, lats, eles = [], [], []
    with open(gpx_path, "r", encoding="utf-8") as f:
        gpx = gpxpy.parse(f)
    for trk in gpx.tracks:
        for seg in trk.segments:
            for p in seg.points:
                if p.longitude is None or p.latitude is None:
                    continue
                lons.append(float(p.longitude))
                lats.append(float(p.latitude))
                e = float(p.elevation) if p.elevation is not None else 0.0
                eles.append(e)

    if len(lons) < 2:
        raise ValueError("Not enough GPX points.")

    # cumulative distance (meters → miles)
    dist_m = [0.0]
    for i in range(1, len(lons)):
        dist_m.append(dist_m[-1] + _haversine_m(lats[i-1], lons[i-1], lats[i], lons[i]))
    to_mi = 1/1609.344
    to_ft = 3.280839895
    distance_miles = [d * to_mi for d in dist_m]
    elevation_feet = [e * to_ft for e in eles]

    data = {
        "distance_miles": distance_miles,
        "elevation_feet": elevation_feet,
        "total_miles": distance_miles[-1]
    }
    json_path.write_text(json.dumps(data), encoding="utf-8")
    return data


# ---------- Routes ----------

@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")


@app.route("/upload", methods=["POST"])
def upload():
    if "gpxfile" not in request.files:
        flash("No file part in the request.")
        return redirect(url_for("index"))
    f = request.files["gpxfile"]
    if not f or f.filename == "":
        flash("No file selected.")
        return redirect(url_for("index"))
    if not f.filename.lower().endswith(".gpx"):
        flash("Please upload a .gpx file.")
        return redirect(url_for("index"))

    mode = request.form.get("mode", "map")
    uid = str(uuid.uuid4())
    gpx_path = UPLOAD_DIR / f"{uid}.gpx"
    f.save(gpx_path.as_posix())

    try:
        if mode == "webgl3d":
            track_json = TRACK_DIR / f"{uid}.json"
            _gpx_to_track_json(gpx_path, track_json)
            return redirect(url_for("result_3d", uid=uid))

        # 2D video + elevation profile
        out_path = VIDEO_DIR / f"{uid}.mp4"
        meta = generate_map_flyover_video(
            gpx_file=gpx_path.as_posix(),
            out_file=out_path.as_posix(),
            fps=30,
            target_duration=60,
            zoom_padding_ratio=0.15,
            style="osm",
            width=VIDEO_W, height=VIDEO_H,
        )
        profile_json = PROFILE_DIR / f"{uid}.json"
        prof = _gpx_to_elev_profile(gpx_path, profile_json)
        # include duration in the profile file so the client can sync precisely
        prof["duration_s"] = float(meta.get("duration_s", (meta.get("frames", 0)/max(1, meta.get("fps", 30)))))
        profile_json.write_text(json.dumps(prof), encoding="utf-8")
        return redirect(url_for("result", vid=uid))

    except Exception as e:
        flash(f"Video generation failed: {e}")
        return redirect(url_for("index"))


@app.route("/result/<vid>", methods=["GET"])
def result(vid):
    video_url   = url_for("static", filename=f"videos/{vid}.mp4")
    profile_url = url_for("static", filename=f"profiles/{vid}.json")
    return render_template("result.html", video_url=video_url, profile_url=profile_url)


@app.route("/map3d/<uid>", methods=["GET"])
def result_3d(uid):
    track_url = url_for("static", filename=f"tracks/{uid}.json")
    return render_template("result_3d.html", track_url=track_url, maptiler_key=MAPTILER_KEY)


if __name__ == "__main__":
    app.run(debug=True)
'''

'''
# app.py
import os
import uuid
import json
import math
from pathlib import Path

from flask import (
    Flask, render_template, request, redirect,
    url_for, flash, send_file, abort
)

import numpy as np
import gpxpy

# Video + HUD burn utilities
from utils.map_video import generate_map_flyover_video, burn_hud_on_video


# -----------------------------------------------------------------------------
# App & paths
# -----------------------------------------------------------------------------
app = Flask(__name__)
app.secret_key = os.environ.get("FLASK_SECRET_KEY", "dev_secret_key")

BASE_DIR    = Path(__file__).resolve().parent
UPLOAD_DIR  = BASE_DIR / "uploads"
VIDEO_DIR   = BASE_DIR / "static" / "videos"
TRACK_DIR   = BASE_DIR / "static" / "tracks"    # for the 3D web map
PROFILE_DIR = BASE_DIR / "static" / "profiles"  # for HUD/chart + downloadable HUD

for d in (UPLOAD_DIR, VIDEO_DIR, TRACK_DIR, PROFILE_DIR):
    d.mkdir(parents=True, exist_ok=True)

MAPTILER_KEY = os.environ.get("MAPTILER_KEY", "YOUR_MAPTILER_KEY")
VIDEO_W, VIDEO_H = 1280, 720


# -----------------------------------------------------------------------------
# Profile computation (distance, elevation gain, moving-time)
# -----------------------------------------------------------------------------
# These defaults are tuned to match Strava-like totals on long road rides.
SIMPLIFY_EPS_M        = 8.0    # reduces GPS wiggle for distance calc
ELEV_SMOOTH_WIN       = 21     # light pre-smoothing at raw samples (odd)

# Robust distance-based gain:
ELEV_RESAMPLE_STEP_M  = 5.0    # resample elevation every ~5 m
ELEV_SMOOTH_OVER_M    = 120.0  # smooth over ~120 m of travel (window length derived)
ELEV_GAIN_THRESH_M    = 0.0    # ***count all positive steps*** after smoothing

STOP_SPEED_MPS        = 0.5    # for moving-time (avg speed)


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
    """Rough meters projection for simplification."""
    lat0 = float(lat[len(lat)//2])
    x = (np.asarray(lon) * math.cos(math.radians(lat0))) * 111320.0
    y = (np.asarray(lat)) * 110540.0
    return x, y


def _simplify_track(lon, lat, ele, tim, eps_m=SIMPLIFY_EPS_M):
    """Keep first/last; only keep next if distance from last kept > eps_m."""
    x, y = _to_local_xy(lon, lat)
    keep = [0]
    last = 0
    for i in range(1, len(x) - 1):
        if math.hypot(x[i] - x[last], y[i] - y[last]) >= eps_m:
            keep.append(i)
            last = i
    if keep[-1] != len(x) - 1:
        keep.append(len(x) - 1)
    keep = np.array(keep, dtype=int)
    return lon[keep], lat[keep], (ele[keep] if ele is not None else None), (tim[keep] if tim is not None else None)


def _robust_elev_gain(dist_m, elev_m,
                      step_m=ELEV_RESAMPLE_STEP_M,
                      smooth_over_m=ELEV_SMOOTH_OVER_M,
                      thresh_m=ELEV_GAIN_THRESH_M):
    """
    Robust total elevation gain:
      1) resample elevation onto a fixed distance grid,
      2) smooth over a distance window (Savitzky–Golay if available),
      3) sum *all* positive steps (thresh_m=0 by default).
    Returns (total_gain_m, cumulative_gain_on_original_samples_m).
    """
    # 1) distance grid + resample
    grid = np.arange(0.0, float(dist_m[-1]) + step_m, step_m)
    e_grid = np.interp(grid, dist_m, elev_m)

    # 2) smooth over ~smooth_over_m of travel
    win = max(3, int(round(smooth_over_m / step_m)))
    if win % 2 == 0:
        win += 1
    try:
        from scipy.signal import savgol_filter
        e_smooth = savgol_filter(e_grid, window_length=win, polyorder=2, mode="interp")
    except Exception:
        kernel = np.ones(win) / win
        e_smooth = np.convolve(e_grid, kernel, mode="same")

    # 3) positive increments (no dead-band by default)
    deltas = np.diff(e_smooth)
    pos = np.where(deltas > thresh_m, deltas, 0.0)

    cum_grid = np.concatenate([[0.0], np.cumsum(pos)])
    total_gain_m = float(cum_grid[-1])

    # map cumulative gain back to original samples (for overlays if needed)
    cum_on_orig = np.interp(dist_m, grid, cum_grid)

    return total_gain_m, cum_on_orig


def _gpx_to_profile(gpx_path: Path, json_path: Path) -> dict:
    """
    Build the profile JSON used by the HUD/chart and downloadable HUD:
      - distance_miles[] (monotonic)
      - elevation_feet[] (smoothed)
      - cum_gain_feet[]  (cumulative along route; from robust method)
      - total_gain_feet  (robust total gain for summary)
      - time_s[] (if timestamps exist)
      - has_time, moving_time_s, total_time_s
      - total_miles
    """
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

    # 1) simplify to reduce GPS wiggle (affects distance only)
    lons, lats, elevs, times = _simplify_track(lons, lats, elevs, times)

    # 2) cumulative distance (meters → miles)
    dist_m = [0.0]
    for i in range(1, len(lons)):
        dist_m.append(dist_m[-1] + _haversine_m(lats[i-1], lons[i-1], lats[i], lons[i]))
    dist_m = np.array(dist_m)
    to_mi = 1 / 1609.344
    distance_miles = (dist_m * to_mi).tolist()

    # 3) light pre-smoothing + robust distance-based gain
    elev_smooth_m = _moving_average(elevs, ELEV_SMOOTH_WIN)
    total_gain_m, cum_gain_on_orig_m = _robust_elev_gain(dist_m, elev_smooth_m)

    to_ft = 3.280839895
    elevation_feet   = (elev_smooth_m * to_ft).tolist()
    cum_gain_feet    = (cum_gain_on_orig_m * to_ft).tolist()
    total_gain_feet  = total_gain_m * to_ft

    # 4) time arrays + moving-time seconds (for avg speed like Strava)
    has_time = all(t is not None for t in times)
    time_s = []
    moving_time_s = 0.0
    if has_time:
        t0 = times[0]
        for i in range(len(times)):
            time_s.append((times[i] - t0).total_seconds())
        for i in range(1, len(times)):
            dt = (times[i] - times[i-1]).total_seconds()
            if dt <= 0:
                continue
            d_m = dist_m[i] - dist_m[i-1]
            v = d_m / dt
            if v >= STOP_SPEED_MPS:
                moving_time_s += dt

    data = {
        "distance_miles": distance_miles,
        "elevation_feet": elevation_feet,
        "cum_gain_feet": cum_gain_feet,
        "total_gain_feet": total_gain_feet,   # <-- summary uses this
        "time_s": time_s,
        "has_time": has_time,
        "moving_time_s": moving_time_s if has_time else None,
        "total_time_s": (time_s[-1] if has_time else None),
        "total_miles": distance_miles[-1],
    }
    json_path.write_text(json.dumps(data), encoding="utf-8")
    return data


# -----------------------------------------------------------------------------
# Routes
# -----------------------------------------------------------------------------
@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")


@app.route("/upload", methods=["POST"])
def upload():
    if "gpxfile" not in request.files:
        flash("No file part in the request.")
        return redirect(url_for("index"))

    f = request.files["gpxfile"]
    if not f or f.filename == "":
        flash("No file selected.")
        return redirect(url_for("index"))
    if not f.filename.lower().endswith(".gpx"):
        flash("Please upload a .gpx file.")
        return redirect(url_for("index"))

    mode = request.form.get("mode", "map")  # "map" (2D video) or "webgl3d"
    uid = str(uuid.uuid4())
    gpx_path = UPLOAD_DIR / f"{uid}.gpx"
    f.save(gpx_path.as_posix())

    try:
        if mode == "webgl3d":
            # Build minimal track JSON for the 3D web map (coords + elev + bounds)
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
            return redirect(url_for("result_3d", uid=uid))

        # ---- 2D chase-cam video + profile used by HUD and downloadable HUD ----
        out_path = VIDEO_DIR / f"{uid}.mp4"
        meta = generate_map_flyover_video(
            gpx_file=gpx_path.as_posix(),
            out_file=out_path.as_posix(),
            fps=30,
            target_duration=60,
            zoom_padding_ratio=0.15,
            style="osm",
            width=VIDEO_W, height=VIDEO_H,
            follow_back_meters=350.0,  # camera follows behind the dot
        )

        # Profile JSON (now with robust total gain)
        profile_json = PROFILE_DIR / f"{uid}.json"
        prof = _gpx_to_profile(gpx_path, profile_json)
        prof["duration_s"] = float(meta.get("duration_s", (meta.get("frames", 0)/max(1, meta.get("fps", 30)))))
        profile_json.write_text(json.dumps(prof), encoding="utf-8")

        return redirect(url_for("result", vid=uid))

    except Exception as e:
        flash(f"Video generation failed: {e}")
        return redirect(url_for("index"))


@app.route("/result/<vid>", methods=["GET"])
def result(vid):
    video_url   = url_for("static", filename=f"videos/{vid}.mp4")
    profile_url = url_for("static", filename=f"profiles/{vid}.json")
    return render_template("result.html", video_url=video_url, profile_url=profile_url, vid=vid)


@app.route("/download/<vid>", methods=["GET"])
def download(vid):
    """
    Download the MP4. If ?hud=1, we burn the LIVE HUD into a copy before serving.
    If ?hud=0, we return the clean video as-is.
    """
    hud = request.args.get("hud", "1") == "1"
    base = VIDEO_DIR / f"{vid}.mp4"
    if not base.exists():
        abort(404)

    if not hud:
        return send_file(base.as_posix(), as_attachment=True, download_name="flyover.mp4")

    profile_json = PROFILE_DIR / f"{vid}.json"
    if not profile_json.exists():
        abort(404)

    out = VIDEO_DIR / f"{vid}_hud.mp4"
    if (not out.exists()) or (out.stat().st_mtime < max(base.stat().st_mtime, profile_json.stat().st_mtime)):
        burn_hud_on_video(str(base), str(profile_json), str(out))

    return send_file(out.as_posix(), as_attachment=True, download_name="flyover_with_stats.mp4")


@app.route("/map3d/<uid>", methods=["GET"])
def result_3d(uid):
    track_url = url_for("static", filename=f"tracks/{uid}.json")
    return render_template("result_3d.html", track_url=track_url, maptiler_key=MAPTILER_KEY)


if __name__ == "__main__":
    app.run(debug=True)
'''



























'''
import os
import uuid
import json
import math
from pathlib import Path

from flask import (
    Flask, render_template, request, redirect,
    url_for, flash, send_file, abort
)

import numpy as np
import gpxpy

# Video + overlay utilities
from utils.map_video import generate_map_flyover_video, burn_overlays_on_video


# -----------------------------------------------------------------------------
# App & paths
# -----------------------------------------------------------------------------
app = Flask(__name__)
app.secret_key = os.environ.get("FLASK_SECRET_KEY", "dev_secret_key")

BASE_DIR    = Path(__file__).resolve().parent
UPLOAD_DIR  = BASE_DIR / "uploads"
VIDEO_DIR   = BASE_DIR / "static" / "videos"
TRACK_DIR   = BASE_DIR / "static" / "tracks"    # for the 3D web map
PROFILE_DIR = BASE_DIR / "static" / "profiles"  # for HUD/chart + downloadable HUD

for d in (UPLOAD_DIR, VIDEO_DIR, TRACK_DIR, PROFILE_DIR):
    d.mkdir(parents=True, exist_ok=True)

MAPTILER_KEY = os.environ.get("MAPTILER_KEY", "YOUR_MAPTILER_KEY")  # used by result_3d.html


# -----------------------------------------------------------------------------
# Profile computation (distance, elevation gain, moving-time)
# -----------------------------------------------------------------------------
# Tunables to match Strava-like totals on long road rides.
SIMPLIFY_EPS_M        = 8.0     # meters: path simplification (distance calc)
ELEV_SMOOTH_WIN       = 21      # odd number: light sample-wise smoothing

# Robust distance-based gain:
ELEV_RESAMPLE_STEP_M  = 5.0     # resample elevation every ~5 m of travel
ELEV_SMOOTH_OVER_M    = 120.0   # smooth elevation over ~120 m of travel
ELEV_GAIN_THRESH_M    = 0.0     # count *all* positive steps after smoothing

STOP_SPEED_MPS        = 0.5     # under this -> stopped (moving-time calc)


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
    """Rough meters projection for simplification."""
    lat0 = float(lat[len(lat)//2])
    x = (np.asarray(lon) * math.cos(math.radians(lat0))) * 111320.0
    y = (np.asarray(lat)) * 110540.0
    return x, y


def _simplify_track(lon, lat, ele, tim, eps_m=SIMPLIFY_EPS_M):
    """Keep first/last; only keep next if distance from last kept > eps_m."""
    x, y = _to_local_xy(lon, lat)
    keep = [0]
    last = 0
    for i in range(1, len(x) - 1):
        if math.hypot(x[i] - x[last], y[i] - y[last]) >= eps_m:
            keep.append(i)
            last = i
    if keep[-1] != len(x) - 1:
        keep.append(len(x) - 1)
    keep = np.array(keep, dtype=int)
    return lon[keep], lat[keep], (ele[keep] if ele is not None else None), (tim[keep] if tim is not None else None)


def _robust_elev_gain(dist_m, elev_m,
                      step_m=ELEV_RESAMPLE_STEP_M,
                      smooth_over_m=ELEV_SMOOTH_OVER_M,
                      thresh_m=ELEV_GAIN_THRESH_M):
    """
    Total elevation gain:
      1) resample elevation onto a fixed distance grid (every step_m),
      2) smooth over a distance window (~smooth_over_m),
      3) sum positive steps (thresh_m=0 -> all positive increments).
    Returns (total_gain_m, cumulative_gain_on_original_samples_m).
    """
    # 1) grid + resample
    grid = np.arange(0.0, float(dist_m[-1]) + step_m, step_m)
    e_grid = np.interp(grid, dist_m, elev_m)

    # 2) smooth
    win = max(3, int(round(smooth_over_m / step_m)))
    if win % 2 == 0:
        win += 1
    try:
        from scipy.signal import savgol_filter
        e_smooth = savgol_filter(e_grid, window_length=win, polyorder=2, mode="interp")
    except Exception:
        kernel = np.ones(win) / win
        e_smooth = np.convolve(e_grid, kernel, mode="same")

    # 3) positive increments
    deltas = np.diff(e_smooth)
    pos = np.where(deltas > thresh_m, deltas, 0.0)

    cum_grid = np.concatenate([[0.0], np.cumsum(pos)])
    total_gain_m = float(cum_grid[-1])

    # map cumulative back to original samples (for HUD if needed)
    cum_on_orig = np.interp(dist_m, grid, cum_grid)

    return total_gain_m, cum_on_orig


def _gpx_to_profile(gpx_path: Path, json_path: Path) -> dict:
    """
    Build the profile JSON used by the HUD/chart and downloadable overlays.
    """
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

    # Simplify (affects distance only)
    lons, lats, elevs, times = _simplify_track(lons, lats, elevs, times)

    # Distance
    dist_m = [0.0]
    for i in range(1, len(lons)):
        dist_m.append(dist_m[-1] + _haversine_m(lats[i-1], lons[i-1], lats[i], lons[i]))
    dist_m = np.array(dist_m)
    to_mi = 1 / 1609.344
    distance_miles = (dist_m * to_mi).tolist()

    # Elevation: light pre-smoothing + robust gain
    elev_smooth_m = _moving_average(elevs, ELEV_SMOOTH_WIN)
    total_gain_m, cum_gain_on_orig_m = _robust_elev_gain(dist_m, elev_smooth_m)

    to_ft = 3.280839895
    elevation_feet   = (elev_smooth_m * to_ft).tolist()
    cum_gain_feet    = (cum_gain_on_orig_m * to_ft).tolist()
    total_gain_feet  = total_gain_m * to_ft

    # Time arrays + moving time
    has_time = all(t is not None for t in times)
    time_s = []
    moving_time_s = 0.0
    if has_time:
        t0 = times[0]
        for i in range(len(times)):
            time_s.append((times[i] - t0).total_seconds())
        for i in range(1, len(times)):
            dt = (times[i] - times[i-1]).total_seconds()
            if dt <= 0:
                continue
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


# -----------------------------------------------------------------------------
# Routes
# -----------------------------------------------------------------------------
@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")


@app.route("/upload", methods=["POST"])
def upload():
    if "gpxfile" not in request.files:
        flash("No file part in the request.")
        return redirect(url_for("index"))

    f = request.files["gpxfile"]
    if not f or f.filename == "":
        flash("No file selected.")
        return redirect(url_for("index"))
    if not f.filename.lower().endswith(".gpx"):
        flash("Please upload a .gpx file.")
        return redirect(url_for("index"))

    mode    = request.form.get("mode", "map")        # "map" or "webgl3d"
    orient  = request.form.get("orientation", "h")   # "h" or "v"

    uid = str(uuid.uuid4())
    gpx_path = UPLOAD_DIR / f"{uid}.gpx"
    f.save(gpx_path.as_posix())

    try:
        if mode == "webgl3d":
            # Build minimal track JSON for the 3D web map (coords + elev + bounds)
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
            return redirect(url_for("result_3d", uid=uid))

        # ---- 2D video generation (horizontal or vertical) ----
        if orient == "v":
            vw, vh = 720, 1280   # vertical 9:16 (Instagram-friendly)
            zoom_boost = 1
            follow_back = 250.0
        else:
            vw, vh = 1280, 720   # horizontal 16:9
            zoom_boost = 0
            follow_back = 350.0

        out_path = VIDEO_DIR / f"{uid}.mp4"
        meta = generate_map_flyover_video(
            gpx_file=gpx_path.as_posix(),
            out_file=out_path.as_posix(),
            fps=30,
            target_duration=60,
            zoom_padding_ratio=0.15,
            style="osm",
            width=vw, height=vh,
            follow_back_meters=follow_back,  # chase-cam follows behind the dot
            zoom_boost=zoom_boost,
        )

        # Profile JSON (for live HUD/chart + downloadable overlays)
        profile_json = PROFILE_DIR / f"{uid}.json"
        prof = _gpx_to_profile(gpx_path, profile_json)
        prof["duration_s"] = float(meta.get("duration_s", (meta.get("frames", 0)/max(1, meta.get("fps", 30)))))
        profile_json.write_text(json.dumps(prof), encoding="utf-8")

        return redirect(url_for("result", vid=uid))

    except Exception as e:
        flash(f"Video generation failed: {e}")
        return redirect(url_for("index"))


@app.route("/result/<vid>", methods=["GET"])
def result(vid):
    video_url   = url_for("static", filename=f"videos/{vid}.mp4")
    profile_url = url_for("static", filename=f"profiles/{vid}.json")
    return render_template("result.html", video_url=video_url, profile_url=profile_url, vid=vid)


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

    # cache output per overlay combination
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


@app.route("/map3d/<uid>", methods=["GET"])
def result_3d(uid):
    track_url = url_for("static", filename=f"tracks/{uid}.json")
    return render_template("result_3d.html", track_url=track_url, maptiler_key=MAPTILER_KEY)


if __name__ == "__main__":
    app.run(debug=True)
'''















import os
import uuid
import json
import math
from pathlib import Path

from flask import (
    Flask, render_template, request, redirect,
    url_for, flash, send_file, abort
)

import numpy as np
import gpxpy

# Video + overlay utilities
from utils.map_video import generate_map_flyover_video, burn_overlays_on_video


# -----------------------------------------------------------------------------
# App & paths
# -----------------------------------------------------------------------------
app = Flask(__name__)
app.secret_key = os.environ.get("FLASK_SECRET_KEY", "dev_secret_key")

BASE_DIR    = Path(__file__).resolve().parent
UPLOAD_DIR  = BASE_DIR / "uploads"
VIDEO_DIR   = BASE_DIR / "static" / "videos"
TRACK_DIR   = BASE_DIR / "static" / "tracks"
PROFILE_DIR = BASE_DIR / "static" / "profiles"

for d in (UPLOAD_DIR, VIDEO_DIR, TRACK_DIR, PROFILE_DIR):
    d.mkdir(parents=True, exist_ok=True)

MAPTILER_KEY = os.environ.get("MAPTILER_KEY", "YOUR_MAPTILER_KEY")


# -----------------------------------------------------------------------------
# Profile computation (distance, elevation gain, moving-time)
# -----------------------------------------------------------------------------
SIMPLIFY_EPS_M        = 8.0     # meters: path simplification (distance calc)
ELEV_SMOOTH_WIN       = 21      # odd number: light sample-wise smoothing

# Robust distance-based gain:
ELEV_RESAMPLE_STEP_M  = 5.0     # resample elevation every ~5 m of travel
ELEV_SMOOTH_OVER_M    = 120.0   # smooth elevation over ~120 m of travel
ELEV_GAIN_THRESH_M    = 0.0     # count *all* positive steps after smoothing

STOP_SPEED_MPS        = 0.5     # under this -> stopped (moving-time calc)


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
    """Rough meters projection for simplification."""
    lat0 = float(lat[len(lat)//2])
    x = (np.asarray(lon) * math.cos(math.radians(lat0))) * 111320.0
    y = (np.asarray(lat)) * 110540.0
    return x, y


def _simplify_track(lon, lat, ele, tim, eps_m=SIMPLIFY_EPS_M):
    """Keep first/last; only keep next if distance from last kept > eps_m."""
    x, y = _to_local_xy(lon, lat)
    keep = [0]
    last = 0
    for i in range(1, len(x) - 1):
        if math.hypot(x[i] - x[last], y[i] - y[last]) >= eps_m:
            keep.append(i)
            last = i
    if keep[-1] != len(x) - 1:
        keep.append(len(x) - 1)
    keep = np.array(keep, dtype=int)
    return lon[keep], lat[keep], (ele[keep] if ele is not None else None), (tim[keep] if tim is not None else None)


def _robust_elev_gain(dist_m, elev_m,
                      step_m=ELEV_RESAMPLE_STEP_M,
                      smooth_over_m=ELEV_SMOOTH_OVER_M,
                      thresh_m=ELEV_GAIN_THRESH_M):
    """
    Total elevation gain:
      1) resample elevation onto a fixed distance grid (every step_m),
      2) smooth over a distance window (~smooth_over_m),
      3) sum positive steps (thresh_m=0 -> all positive increments).
    Returns (total_gain_m, cumulative_gain_on_original_samples_m).
    """
    # 1) grid + resample
    grid = np.arange(0.0, float(dist_m[-1]) + step_m, step_m)
    e_grid = np.interp(grid, dist_m, elev_m)

    # 2) smooth
    win = max(3, int(round(smooth_over_m / step_m)))
    if win % 2 == 0:
        win += 1
    try:
        from scipy.signal import savgol_filter
        e_smooth = savgol_filter(e_grid, window_length=win, polyorder=2, mode="interp")
    except Exception:
        kernel = np.ones(win) / win
        e_smooth = np.convolve(e_grid, kernel, mode="same")

    # 3) positive increments
    deltas = np.diff(e_smooth)
    pos = np.where(deltas > thresh_m, deltas, 0.0)

    cum_grid = np.concatenate([[0.0], np.cumsum(pos)])
    total_gain_m = float(cum_grid[-1])

    # map cumulative back to original samples (for HUD if needed)
    cum_on_orig = np.interp(dist_m, grid, cum_grid)

    return total_gain_m, cum_on_orig


def _gpx_to_profile(gpx_path: Path, json_path: Path) -> dict:
    """
    Build the profile JSON used by the HUD/chart and downloadable overlays.
    """
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

    # Simplify (affects distance only)
    lons, lats, elevs, times = _simplify_track(lons, lats, elevs, times)

    # Distance
    dist_m = [0.0]
    for i in range(1, len(lons)):
        dist_m.append(dist_m[-1] + _haversine_m(lats[i-1], lons[i-1], lats[i], lons[i]))
    dist_m = np.array(dist_m)
    to_mi = 1 / 1609.344
    distance_miles = (dist_m * to_mi).tolist()

    # Elevation: light pre-smoothing + robust gain
    elev_smooth_m = _moving_average(elevs, ELEV_SMOOTH_WIN)
    total_gain_m, cum_gain_on_orig_m = _robust_elev_gain(dist_m, elev_smooth_m)

    to_ft = 3.280839895
    elevation_feet   = (elev_smooth_m * to_ft).tolist()
    cum_gain_feet    = (cum_gain_on_orig_m * to_ft).tolist()
    total_gain_feet  = total_gain_m * to_ft

    # Time arrays + moving time
    has_time = all(t is not None for t in times)
    time_s = []
    moving_time_s = 0.0
    if has_time:
        t0 = times[0]
        for i in range(len(times)):
            time_s.append((times[i] - t0).total_seconds())
        for i in range(1, len(times)):
            dt = (times[i] - times[i-1]).total_seconds()
            if dt <= 0:
                continue
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


# -----------------------------------------------------------------------------
# Routes
# -----------------------------------------------------------------------------
@app.route("/", methods=["GET"])
def landing():
    """Landing page with hero + cards"""
    return render_template("landing.html")


@app.route("/video", methods=["GET"])
def video():
    """Upload page for GPX to video"""
    return render_template("video.html")


@app.route("/contact", methods=["POST"])
def contact():
    """
    Handle landing-page contact form.
    For now we just flash a confirmation; you can plug in email or DB later.
    """
    name = (request.form.get("name") or "").strip()
    email = (request.form.get("email") or "").strip()
    message = (request.form.get("message") or "").strip()

    if not name or not email or not message:
        flash("Please fill out all fields.")
    else:
        flash("Thanks for your message — we’ll get back to you soon!")

    return redirect(url_for("landing"))


@app.route("/upload", methods=["POST"])
def upload():
    if "gpxfile" not in request.files:
        flash("No file part in the request.")
        return redirect(url_for("video"))

    f = request.files["gpxfile"]
    if not f or f.filename == "":
        flash("No file selected.")
        return redirect(url_for("video"))
    if not f.filename.lower().endswith(".gpx"):
        flash("Please upload a .gpx file.")
        return redirect(url_for("video"))

    mode    = request.form.get("mode", "map")        # "map" or "webgl3d"
    orient  = request.form.get("orientation", "h")   # "h" or "v"

    uid = str(uuid.uuid4())
    gpx_path = UPLOAD_DIR / f"{uid}.gpx"
    f.save(gpx_path.as_posix())

    try:
        # --- WebGL 3D map path ---
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
            return redirect(url_for("result_3d", uid=uid))

        # --- 2D video path ---
        if orient == "v":
            vw, vh = 720, 1280   # vertical 9:16
            zoom_boost = 1
            follow_back = 250.0
        else:
            vw, vh = 1280, 720   # horizontal 16:9
            zoom_boost = 0
            follow_back = 350.0

        out_path = VIDEO_DIR / f"{uid}.mp4"
        meta = generate_map_flyover_video(
            gpx_file=gpx_path.as_posix(),
            out_file=out_path.as_posix(),
            fps=30,
            target_duration=60,
            zoom_padding_ratio=0.15,
            style="osm",
            width=vw, height=vh,
            follow_back_meters=follow_back,
            zoom_boost=zoom_boost,
        )

        # Profile JSON (for HUD/chart + downloadable overlays)
        profile_json = PROFILE_DIR / f"{uid}.json"
        prof = _gpx_to_profile(gpx_path, profile_json)
        prof["duration_s"] = float(meta.get("duration_s", (meta.get("frames", 0)/max(1, meta.get("fps", 30)))))
        profile_json.write_text(json.dumps(prof), encoding="utf-8")

        return redirect(url_for("result", vid=uid))

    except Exception as e:
        flash(f"Video generation failed: {e}")
        return redirect(url_for("video"))


@app.route("/result/<vid>", methods=["GET"])
def result(vid):
    video_url   = url_for("static", filename=f"videos/{vid}.mp4")
    profile_url = url_for("static", filename=f"profiles/{vid}.json")
    return render_template("result.html", video_url=video_url, profile_url=profile_url, vid=vid)


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

    # cache output per overlay combination
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


@app.route("/map3d/<uid>", methods=["GET"])
def result_3d(uid):
    track_url = url_for("static", filename=f"tracks/{uid}.json")
    return render_template("result_3d.html", track_url=track_url, maptiler_key=MAPTILER_KEY)


if __name__ == "__main__":
    app.run(debug=True)
