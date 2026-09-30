import re
import numpy as np, pandas as pd
from sklearn.neighbors import BallTree
from backend.config import RAW_CSV, UserError

MISSING_MSG = "Please download the official GSI manganese occurrence dataset and place it in data/raw/ as manganese_deposits.csv (see data/raw/README.md)."

def _num(v):
    """Decimal degrees, or DMS strings like 21°45'30\"N -> decimal."""
    if pd.isna(v): return np.nan
    try: return float(v)
    except (ValueError, TypeError): pass
    p = [float(x) for x in re.findall(r"\d+\.?\d*", str(v))]
    if not p: return np.nan
    return p[0] + (p[1] / 60 if len(p) > 1 else 0) + (p[2] / 3600 if len(p) > 2 else 0)

def _col(df, keys):
    for c in df.columns:
        if any(k in c.lower() for k in keys): return c

def load_occurrences():
    """Load real GSI occurrences, clean coordinates, drop invalid + duplicates. Returns (df, report)."""
    if not RAW_CSV.exists(): raise UserError(MISSING_MSG)
    try: raw = pd.read_csv(RAW_CSV, encoding="utf-8-sig")
    except Exception:
        raw = pd.read_csv(RAW_CSV, encoding="latin-1")
    lat_c = _col(raw, ["latdd"]) or _col(raw, ["lat"])
    lon_c = _col(raw, ["londd"]) or _col(raw, ["lon", "long"])
    if not lat_c or not lon_c:
        raise UserError(f"Could not find latitude/longitude columns. Found: {list(raw.columns)}")
    df = pd.DataFrame({"lat": raw[lat_c].map(_num), "lon": raw[lon_c].map(_num)})
    for name, keys in [("locality", ["locality"]), ("state", ["state"]), ("host_rock", ["host"]),
                       ("formation", ["formation"]), ("metallogenesis", ["metallo"])]:
        c = _col(raw, keys); df[name] = raw[c] if c else ""
    rep = {"rows_raw": len(df), "missing_coordinates": int(df[["lat", "lon"]].isna().any(axis=1).sum())}
    df = df.dropna(subset=["lat", "lon"])
    ok = df.lat.between(6, 37.5) & df.lon.between(68, 97.5)   # India bounding box
    rep["out_of_range_coordinates"] = int((~ok).sum())
    df = df[ok]
    n = len(df)
    df = df.drop_duplicates(subset=["lat", "lon"]).copy()
    df["_k"] = df.lat.round(3).astype(str) + df.lon.round(3).astype(str)
    df = df.drop_duplicates("_k").drop(columns="_k").reset_index(drop=True)
    rep["duplicates_removed"] = n - len(df)
    rep["rows_clean"] = len(df)
    if len(df) < 20:
        raise UserError(f"Only {len(df)} valid occurrence points; at least 20 are needed (try clearing STUDY_STATE or check the CSV).")
    return df, rep

def study_area(df, state=""):
    """Bounding box (W,S,E,N) from the actual occurrence distribution (+0.25 deg buffer)."""
    d = df[df.state.astype(str).str.lower().str.contains(state.lower(), na=False)] if state else df
    if len(d) < 20: raise UserError(f"Invalid study area: only {len(d)} occurrences for state '{state}'. Need >= 20.")
    return d, [d.lon.min() - .25, d.lat.min() - .25, d.lon.max() + .25, d.lat.max() + .25]

def background_points(occ, bbox, ratio=3, min_km=10, seed=42):
    """Random pseudo-absence points inside the study box, >= min_km from any known occurrence.
    These are 'background / no-known-occurrence' points, NOT confirmed barren ground."""
    rng = np.random.default_rng(seed)
    tree = BallTree(np.radians(occ[["lat", "lon"]].values), metric="haversine")
    pts, target = [], len(occ) * ratio
    for _ in range(200):
        c = np.column_stack([rng.uniform(bbox[1], bbox[3], target), rng.uniform(bbox[0], bbox[2], target)])
        d, _i = tree.query(np.radians(c), k=1)
        pts.extend(c[d[:, 0] * 6371 >= min_km].tolist())
        if len(pts) >= target: break
    return pd.DataFrame(pts[:target], columns=["lat", "lon"])
