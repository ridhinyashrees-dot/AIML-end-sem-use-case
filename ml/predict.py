import json, joblib
import numpy as np, pandas as pd
from backend.config import PROC, OUT, MODELS, FEATURES, LOW_T, HIGH_T, UserError
from ml.feature_extraction import make_grid, extract

def classify(p): return np.where(p < LOW_T, "LOW", np.where(p < HIGH_T, "MODERATE", "HIGH"))

def run_prediction(name):
    mp = MODELS / f"{name}.joblib"
    if not mp.exists(): raise UserError(f"Model '{name}' not trained yet. Train it first.")
    info = json.loads((PROC / "study_area.json").read_text())
    model = joblib.load(mp)
    grid, step = make_grid(info["bbox"])
    g = extract(grid, info["bbox"])
    if g.empty: raise UserError("Satellite data unavailable for the prediction grid.")
    g["probability"] = model.predict_proba(g[FEATURES])[:, 1]
    g["class"] = classify(g.probability.values)
    g.round(5).to_csv(OUT / "prediction.csv", index=False)
    h = step / 2
    feats = [{"type": "Feature", "properties": {"probability": round(float(r.probability), 3), "class": r["class"]},
              "geometry": {"type": "Polygon", "coordinates": [[[r.lon - h, r.lat - h], [r.lon + h, r.lat - h], [r.lon + h, r.lat + h], [r.lon - h, r.lat + h], [r.lon - h, r.lat - h]]]}}
             for _, r in g.iterrows()]
    (OUT / "prospectivity.geojson").write_text(json.dumps({"type": "FeatureCollection", "name": "Predicted Manganese Prospectivity", "features": feats}))
    return {"cells": len(g), "cell_size_deg": step, "counts": g["class"].value_counts().to_dict()}
