/*
// static/js/map3d.js
(function () {
  const buttons = document.querySelectorAll(".speed-btn");
  let speed = 1; // 1x, 2x, 3x, 5x, 10x

  // read key injected by template
  const HAS_KEY = typeof MAPTILER_KEY === "string" && MAPTILER_KEY && MAPTILER_KEY !== "YOUR_MAPTILER_KEY";

  // hook into the existing speed buttons
  buttons.forEach(btn => {
    btn.addEventListener("click", () => {
      speed = parseFloat(btn.getAttribute("data-rate"));
    });
  });

  // helper: style/terrain sources depending on key
  function getStyleUrl() {
    return HAS_KEY
      ? `https://api.maptiler.com/maps/outdoor/style.json?key=${MAPTILER_KEY}`
      : `https://demotiles.maplibre.org/style.json`; // open raster style, no key
  }
  function addTerrain(map) {
    if (HAS_KEY) {
      map.addSource("terrain", {
        type: "raster-dem",
        url: `https://api.maptiler.com/tiles/terrain-rgb/tiles.json?key=${MAPTILER_KEY}`,
        tileSize: 256
      });
    } else {
      // open Terrarium DEM tiles (no key)
      map.addSource("terrain", {
        type: "raster-dem",
        tiles: ["https://s3.amazonaws.com/elevation-tiles-prod/terrarium/{z}/{x}/{y}.png"],
        tileSize: 256,
        encoding: "terrarium"
      });
    }
    map.setTerrain({ source: "terrain", exaggeration: 1.5 });
  }

  // Load the track JSON and build the map
  fetch(TRACK_URL).then(r => {
    if (!r.ok) throw new Error(`Track fetch failed: ${r.status}`);
    return r.json();
  }).then(track => {
    const coords = track?.geometry?.coordinates || [];
    const bounds = track?.properties?.bounds;
    if (coords.length < 2) throw new Error("Track has too few points.");

    const map = new maplibregl.Map({
      container: "map",
      style: getStyleUrl(),
      center: coords[Math.floor(coords.length/2)],
      zoom: 11,
      pitch: 60,
      bearing: 0,
      antialias: true,
    });
    map.addControl(new maplibregl.NavigationControl({ visualizePitch: true }), "top-left");

    map.on("load", () => {
      // terrain
      addTerrain(map);

      // sky (optional)
      map.addLayer({
        id: "sky",
        type: "sky",
        paint: { "sky-type": "atmosphere", "sky-atmosphere-sun-intensity": 12 }
      });

      // fit to route
      if (Array.isArray(bounds) && bounds.length === 4) {
        map.fitBounds([[bounds[0], bounds[1]], [bounds[2], bounds[3]]], { padding: 60, pitch: 60, bearing: 0, duration: 0 });
      }

      // full route
      map.addSource("route", { type: "geojson",
        data: { type: "Feature", geometry: { type: "LineString", coordinates: coords } }
      });
      map.addLayer({
        id: "route-line",
        type: "line",
        source: "route",
        paint: { "line-color": "#1976d2", "line-width": 4 }
      });

      // progressive route
      map.addSource("route-progress", { type: "geojson",
        data: { type: "Feature", geometry: { type: "LineString", coordinates: [coords[0]] } }
      });
      map.addLayer({
        id: "route-progress-line",
        type: "line",
        source: "route-progress",
        paint: { "line-color": "#ff7f0e", "line-width": 5 }
      });

      // marker
      const marker = new maplibregl.Marker({ color: "#ff6d00" }).setLngLat(coords[0]).addTo(map);

      // animation
      const baseDurationMs = 60000;        // 60s at 1×
      const intervalMs = 33;               // ~30 fps
      const stepsAt1x = baseDurationMs / intervalMs;
      const advancePerTick = (coords.length - 1) / stepsAt1x;

      let idxF = 0;
      function animate() {
        idxF += advancePerTick * speed;
        if (idxF >= coords.length - 1) idxF = coords.length - 1;

        const idx = Math.floor(idxF);
        const p = coords[idx];
        const pNext = coords[Math.min(idx + 1, coords.length - 1)];

        marker.setLngLat(p);
        map.getSource("route-progress").setData({
          type: "Feature",
          geometry: { type: "LineString", coordinates: coords.slice(0, idx + 1) }
        });

        // follow camera
        const bearing = Math.atan2(pNext[0] - p[0], pNext[1] - p[1]) * 180 / Math.PI;
        map.easeTo({
          center: p,
          bearing: 180 - bearing,
          pitch: 60,
          duration: intervalMs,
          easing: t => t
        });

        if (idxF < coords.length - 1) setTimeout(animate, intervalMs);
      }
      animate();
    });

    map.on("error", e => console.warn("Map error:", e));
  }).catch(err => {
    console.error(err);
    alert("Failed to load 3D map. Open the browser console for details.");
  });
})();
*/

// static/js/map3d.js
// 3D chase-cam with grade-colored route. Works with/without MAPTILER_KEY.

(function () {
  const buttons = document.querySelectorAll(".speed-btn");
  let speed = 1; // 1x, 2x, 3x, 5x, 10x
  buttons.forEach(btn => btn.addEventListener("click", () => {
    speed = parseFloat(btn.getAttribute("data-rate"));
  }));

  const HAS_KEY = typeof MAPTILER_KEY === "string" && MAPTILER_KEY && MAPTILER_KEY !== "YOUR_MAPTILER_KEY";

  function getStyleUrl() {
    return HAS_KEY
      ? `https://api.maptiler.com/maps/outdoor/style.json?key=${MAPTILER_KEY}`
      : `https://demotiles.maplibre.org/style.json`; // open raster style
  }
  function addTerrain(map, exaggeration = 1.6) {
    if (HAS_KEY) {
      map.addSource("terrain", {
        type: "raster-dem",
        url: `https://api.maptiler.com/tiles/terrain-rgb/tiles.json?key=${MAPTILER_KEY}`,
        tileSize: 256
      });
    } else {
      map.addSource("terrain", {
        type: "raster-dem",
        tiles: ["https://s3.amazonaws.com/elevation-tiles-prod/terrarium/{z}/{x}/{y}.png"],
        tileSize: 256,
        encoding: "terrarium"
      });
    }
    map.setTerrain({ source: "terrain", exaggeration });
  }

  // --- Geo helpers ---
  const R = 6371000; // Earth radius (m)
  function toRad(d) { return d * Math.PI / 180; }
  function toDeg(r) { return r * 180 / Math.PI; }
  function haversineMeters(a, b) {
    const [lon1, lat1] = a, [lon2, lat2] = b;
    const dLat = toRad(lat2 - lat1), dLon = toRad(lon2 - lon1);
    const s = Math.sin(dLat/2)**2 + Math.cos(toRad(lat1))*Math.cos(toRad(lat2))*Math.sin(dLon/2)**2;
    return 2 * R * Math.asin(Math.sqrt(s));
  }
  function bearingDeg(a, b) {
    const [lon1, lat1] = a, [lon2, lat2] = b;
    const y = Math.sin(toRad(lon2-lon1)) * Math.cos(toRad(lat2));
    const x = Math.cos(toRad(lat1))*Math.sin(toRad(lat2)) -
              Math.sin(toRad(lat1))*Math.cos(toRad(lat2))*Math.cos(toRad(lon2-lon1));
    return (toDeg(Math.atan2(y, x)) + 360) % 360;
  }
  // Destination point given start, bearing (deg), and distance (m)
  function destPoint([lon, lat], bearingDeg, distM) {
    const br = toRad(bearingDeg);
    const ang = distM / R;
    const φ1 = toRad(lat), λ1 = toRad(lon);
    const sinφ2 = Math.sin(φ1)*Math.cos(ang) + Math.cos(φ1)*Math.sin(ang)*Math.cos(br);
    const φ2 = Math.asin(sinφ2);
    const y = Math.sin(br)*Math.sin(ang)*Math.cos(φ1);
    const x = Math.cos(ang) - Math.sin(φ1)*sinφ2;
    const λ2 = λ1 + Math.atan2(y, x);
    return [((toDeg(λ2)+540)%360)-180, toDeg(φ2)];
  }

  // Build small-segment FeatureCollection with grade% per segment
  function segmentsWithGrade(coords, elev) {
    const feats = [];
    for (let i = 0; i < coords.length - 1; i++) {
      const a = coords[i], b = coords[i+1];
      const d = Math.max(1, haversineMeters(a, b)); // avoid /0
      let g = 0;
      if (Array.isArray(elev) && elev[i] != null && elev[i+1] != null) {
        g = 100 * (elev[i+1] - elev[i]) / d; // % grade
      }
      feats.push({
        type: "Feature",
        properties: { grade: g },
        geometry: { type: "LineString", coordinates: [a, b] }
      });
    }
    return { type: "FeatureCollection", features: feats };
  }

  // --- Load the track and render ---
  fetch(TRACK_URL).then(r => {
    if (!r.ok) throw new Error(`Track fetch failed: ${r.status}`);
    return r.json();
  }).then(track => {
    const coords = track?.geometry?.coordinates || [];
    const elev   = track?.properties?.elev || [];
    const bounds = track?.properties?.bounds;

    if (coords.length < 2) throw new Error("Track has too few points.");

    const map = new maplibregl.Map({
      container: "map",
      style: getStyleUrl(),
      center: coords[Math.floor(coords.length/2)],
      zoom: 11.5,
      pitch: 68,     // pronounced 3D
      bearing: 0,
      antialias: true,
    });
    map.addControl(new maplibregl.NavigationControl({ visualizePitch: true }), "top-left");

    map.on("load", () => {
      addTerrain(map, 1.6);

      map.addLayer({
        id: "sky",
        type: "sky",
        paint: { "sky-type": "atmosphere", "sky-atmosphere-sun-intensity": 12 }
      });

      if (Array.isArray(bounds) && bounds.length === 4) {
        map.fitBounds([[bounds[0], bounds[1]], [bounds[2], bounds[3]]], { padding: 60, pitch: 68, duration: 0 });
      }

      // --- Grade-colored baseline ---
      const gradeSegments = segmentsWithGrade(coords, elev);
      map.addSource("route-grade", { type: "geojson", data: gradeSegments });
      map.addLayer({
        id: "route-grade",
        type: "line",
        source: "route-grade",
        paint: {
          "line-width": 6,
          "line-color": [
            "interpolate", ["linear"], ["get", "grade"],
            -15, "#2ecc71",  // steep down
             -3, "#27ae60",  // down
              0, "#1976d2",  // flat
              4, "#f39c12",  // moderate up
              8, "#e67e22",
             12, "#e74c3c"   // steep up
          ]
        }
      });

      // --- Progressive overlay line & marker ---
      map.addSource("route-progress", {
        type: "geojson",
        data: { type: "Feature", geometry: { type: "LineString", coordinates: [coords[0]] } }
      });
      map.addLayer({
        id: "route-progress-line",
        type: "line",
        source: "route-progress",
        paint: { "line-color": "#ffffff", "line-width": 3 }
      });

      const marker = new maplibregl.Marker({ color: "#ff6d00" }).setLngLat(coords[0]).addTo(map);

      // --- Animation (CHASE CAM) ---
      const baseDurationMs = 60000; // 60s at 1x
      const intervalMs = 33;        // ~30 fps
      const stepsAt1x = baseDurationMs / intervalMs;
      const advancePerTick = (coords.length - 1) / stepsAt1x;

      // How far behind the dot to keep the camera (meters) and slight screen offset
      const FOLLOW_BACK_METERS = 350;
      const SCREEN_OFFSET = [0, 140]; // [x,y] px; positive y pushes dot higher in frame

      let idxF = 0;
      function animate() {
        idxF += advancePerTick * speed;
        if (idxF >= coords.length - 1) idxF = coords.length - 1;

        const i = Math.floor(idxF);
        const p = coords[i];
        const pNext = coords[Math.min(i + 1, coords.length - 1)];

        // update marker & progress
        marker.setLngLat(p);
        map.getSource("route-progress").setData({
          type: "Feature",
          geometry: { type: "LineString", coordinates: coords.slice(0, i + 1) }
        });

        // camera bearing and "behind" center point
        const brg = bearingDeg(p, pNext);
        const centerBehind = destPoint(p, brg - 180, FOLLOW_BACK_METERS);

        map.easeTo({
          center: centerBehind,
          bearing: brg,
          pitch: 68,
          offset: SCREEN_OFFSET,
          duration: intervalMs,
          easing: t => t
        });

        if (idxF < coords.length - 1) setTimeout(animate, intervalMs);
      }
      animate();
    });

    map.on("error", e => console.warn("Map error:", e));
  }).catch(err => {
    console.error(err);
    alert("Failed to load 3D map. Open the console for details.");
  });
})();
