# Manganese Prospectivity Mapping
## 1-3. Problem, objective
**Problem statement:** Using AI/ML and Space Technology to Identify Manganese Reserves and Overcome Production Shortfalls.
**Objective:** learn from real GSI manganese occurrences + Sentinel-2 + SRTM and produce a *predicted* LOW/MODERATE/HIGH prospectivity map. It does NOT confirm reserves; that requires field work and official geological assessment. Prospectivity maps help prioritise exploration zones, shrink search areas and plan field surveys, which can reduce exploration time/cost and so help address shortfalls. The ML model itself does not increase mining output.
## 4. Architecture
See `docs/PROJECT_ARCHITECTURE.md`. Frontend (HTML/CSS/JS + Leaflet) -> Flask REST API -> `ml/` pipeline -> Google Earth Engine.
(Simplified layout: routes live in `backend/app.py`; models are built in `ml/train.py`.)
## 5-6. Datasets
1. **GSI manganese occurrences (real, NOT bundled):** see `data/raw/README.md`; save as `data/raw/manganese_deposits.csv`.
2. **Sentinel-2:** `COPERNICUS/S2_SR_HARMONIZED` via Earth Engine (B2,B3,B4,B8,B11,B12 + NDVI, SWIR ratio, iron oxide, ferrous ratio; SCL cloud mask, <30% cloud, median composite).
3. **DEM:** `USGS/SRTMGL1_003` -> elevation, slope, aspect.
4. Optional geology (Bhukosh/NGDR): not integrated.
## 7. Earth Engine setup
Sign up at https://earthengine.google.com, create/choose a Google Cloud project with the Earth Engine API enabled and registered for Earth Engine use, then run `earthengine authenticate` once (credentials stay in your user profile, never in the project).
## 8-10. Install and configure (Windows 11)
```
setup.bat          :: creates venv, installs requirements, creates .env
```
Edit `.env`: `EE_PROJECT=your-project-id` (optionally `STUDY_STATE=Odisha`, `S2_START`, `S2_END`).
Linux/Mac: `python3 -m venv venv && source venv/bin/activate && pip install -r requirements.txt && cp .env.example .env && python backend/app.py`
## 11-14. Run
`run.bat` (or `python backend/app.py`), open http://127.0.0.1:5000. In the dashboard: **Load occurrences -> Build feature dataset -> Train -> Run prediction**. Training and map generation are done from the dashboard buttons (REST endpoints `/api/build-features`, `/api/train`, `/api/predict`).
## 15. Outputs
`data/processed/features.csv`, `study_area.json` (validation summary), `metrics.json`; `models/*.joblib`; `data/outputs/prediction.csv`, `prospectivity.geojson`.
## 16. Troubleshooting
- "Please download the official GSI..." -> CSV missing/misnamed.
- "Earth Engine project not configured" -> set `EE_PROJECT` in `.env`.
- "authentication failed" -> run `earthengine authenticate`; confirm the project is registered for EE.
- "Satellite data unavailable" -> widen `S2_START`/`S2_END`.
- "Invalid study area" / "insufficient points" -> need >=20 valid occurrences; clear `STUDY_STATE`.
- Missing module -> activate venv, `pip install -r requirements.txt`.
## 17. Limitations
Few occurrence points; background points are not confirmed barren; 3:1 sampling makes probabilities relative; grid resolution is coarse (capped cells, ~2 km default); no geological layers; Earth Engine requests can be slow for big areas. Results are prospectivity only.
## 18. Future scope
GSI lithology/fault layers, Sentinel-1/ASTER, finer grid via GeoTIFF export, hyperparameter tuning, uncertainty maps.
