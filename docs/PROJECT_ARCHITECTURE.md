# Architecture
Browser (Leaflet dashboard, frontend/) --REST--> Flask (backend/app.py) --> ml/ modules --> Google Earth Engine

Pipeline: GSI CSV -> clean/validate (ml/preprocessing.py) -> study-area bbox from occurrences -> background points (random, >=10 km from any occurrence; "no known occurrence", not confirmed barren) -> Sentinel-2 + SRTM features via Earth Engine (ml/feature_extraction.py) -> features.csv -> train RF/SVM/XGBoost (ml/train.py) -> spatial block GroupKFold -> metrics from out-of-fold predictions (ml/evaluate.py) -> grid prediction (ml/predict.py) -> LOW (<0.40) / MODERATE (0.40-0.70) / HIGH (>=0.70) -> GeoJSON/CSV -> map.

Spatial validation: samples are grouped into 0.25° x 0.25° blocks; each block is entirely in train or test, avoiding spatial leakage.
Probabilities are relative (background:positive ratio is 3:1 and background is not true absence); treat classes as ranking for exploration prioritisation only.
Optional geological layers (GSI Bhukosh/NGDR) are not integrated; the main system uses occurrences + Sentinel-2 + DEM.
