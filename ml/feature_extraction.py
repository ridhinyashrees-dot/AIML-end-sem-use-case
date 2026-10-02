import numpy as np, pandas as pd
from backend.config import EE_PROJECT, S2_START, S2_END, FEATURES, MAX_GRID_CELLS, UserError

def init_ee():
    try: import ee
    except ImportError: raise UserError("Missing dependency 'earthengine-api'. Run: pip install -r requirements.txt")
    if not EE_PROJECT or EE_PROJECT.startswith("your-"):
        raise UserError("Earth Engine project not configured. Set EE_PROJECT in the .env file.")
    try: ee.Initialize(project=EE_PROJECT)
    except Exception as e:
        raise UserError(f"Earth Engine authentication/initialisation failed. Run 'earthengine authenticate' and check EE_PROJECT. Details: {str(e)[:200]}")
    return ee

def feature_image(ee, bbox):
    geom = ee.Geometry.Rectangle(bbox)
    def mask(i):  # SCL: 3 shadow, 8/9 cloud, 10 cirrus, 11 snow
        s = i.select("SCL")
        return i.updateMask(s.neq(3).And(s.neq(8)).And(s.neq(9)).And(s.neq(10)).And(s.neq(11)))
    col = (ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED").filterBounds(geom).filterDate(S2_START, S2_END)
           .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", 30)))
    if col.size().getInfo() == 0:
        raise UserError("Satellite data unavailable: no Sentinel-2 scenes for this area/period. Change S2_START/S2_END or study area.")
    s2 = col.map(mask).select(["B2", "B3", "B4", "B8", "B11", "B12"]).median().divide(10000)
    idx = ee.Image.cat([
        s2.normalizedDifference(["B8", "B4"]).rename("NDVI"),
        s2.select("B11").divide(s2.select("B12")).rename("SWIR_ratio"),
        s2.select("B4").divide(s2.select("B2")).rename("iron_oxide"),
        s2.select("B11").divide(s2.select("B8")).rename("ferrous")])
    dem = ee.Terrain.products(ee.Image("USGS/SRTMGL1_003")).select(["elevation", "slope", "aspect"])
    return s2.addBands(idx).addBands(dem).select(FEATURES)

def extract(points, bbox, chunk=100):
    """Extract satellite + terrain features in small batches to avoid Earth Engine memory errors."""

    ee = init_ee()
    img = feature_image(ee, bbox)
    rows = []

    pts = points.reset_index(drop=True)

    print(f"Extracting features for {len(pts)} points...")
    print(f"Batch size: {chunk}")

    for s in range(0, len(pts), chunk):

        part = pts.iloc[s:s + chunk]

        print(
            f"Processing points {s + 1}-{min(s + chunk, len(pts))} "
            f"of {len(pts)}..."
        )

        features = []

        for i, r in part.iterrows():
            features.append(
                ee.Feature(
                    ee.Geometry.Point([
                        float(r.lon),
                        float(r.lat)
                    ]),
                    {"pid": int(i)}
                )
            )

        fc = ee.FeatureCollection(features)

        try:
            result = (
                img.sampleRegions(
                    collection=fc,
                    scale=30,
                    geometries=False,
                    tileScale=4
                )
                .getInfo()
            )

        except Exception as e:
            raise UserError(
                f"Earth Engine request failed while processing "
                f"points {s + 1}-{min(s + chunk, len(pts))}: "
                f"{str(e)[:300]}"
            )

        rows.extend(
            f["properties"]
            for f in result["features"]
        )

    if not rows:
        raise UserError(
            "Satellite data unavailable for the sampled points."
        )

    f = pd.DataFrame(rows)

    if f.empty:
        raise UserError(
            "Satellite data unavailable for the sampled points."
        )

    out = (
        pts
        .join(
            f.set_index("pid"),
            how="inner"
        )
        .dropna(subset=FEATURES)
    )

    print(
        f"Successfully extracted features for "
        f"{len(out)} / {len(pts)} points."
    )

    return out
def make_grid(bbox, step=0.02):
    while ((bbox[2] - bbox[0]) / step) * ((bbox[3] - bbox[1]) / step) > MAX_GRID_CELLS: step *= 1.25
    lons = np.arange(bbox[0] + step / 2, bbox[2], step); lats = np.arange(bbox[1] + step / 2, bbox[3], step)
    g = pd.DataFrame([(a, o) for a in lats for o in lons], columns=["lat", "lon"])
    return g, step
