# plactiv 

## GPX Flyover (2D Map + 3D Terrain)

Upload a `.gpx`, get a video:
- **2D Map**: route over OpenStreetMap tiles (camera follows the path).
- **3D Terrain**: shaded surface from public Terrarium DEM tiles with a flyover camera.

## Quick start
```bash
python3 -m venv .venv
source .venv/bin/activate           # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python app.py
# visit http://127.0.0.1:5000
