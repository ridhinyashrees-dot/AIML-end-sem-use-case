import re

import numpy as np
import pandas as pd

from sklearn.neighbors import BallTree

from backend.config import RAW_CSV, UserError


MISSING_MSG = (
    "Please download the official GSI manganese occurrence dataset and place it "
    "in data/raw/ as manganese_deposits.csv (see data/raw/README.md)."
)


def _num(v):
    """Convert decimal/DMS coordinate values to decimal degrees."""

    if pd.isna(v):
        return np.nan

    try:
        return float(v)
    except (ValueError, TypeError):
        pass

    text = str(v)

    # Extract numbers from strings such as:
    # 21°45'30"N
    # 21 45 30
    nums = re.findall(r"\d+(?:\.\d+)?", text)

    if not nums:
        return np.nan

    p = [float(x) for x in nums]

    value = p[0]

    if len(p) > 1:
        value += p[1] / 60

    if len(p) > 2:
        value += p[2] / 3600

    # Handle S/W coordinates if they occur.
    if "S" in text.upper() or "W" in text.upper():
        value = -value

    return value


def _col(df, keys):
    """Find the first column whose name contains one of the given keywords."""

    for c in df.columns:
        name = str(c).strip().lower()

        if any(k.lower() in name for k in keys):
            return c

    return None


def _normalize_state(value):
    """
    Normalize state names used in the GSI dataset so that
    the frontend can use standard names.
    """

    if pd.isna(value):
        return ""

    state = str(value).strip()

    aliases = {
        "Maharastra": "Maharashtra",
        "Maharashtra": "Maharashtra",

        "Orissa": "Odisha",
        "Odisha": "Odisha",

        "Chattisgarh": "Chhattisgarh",
        "Chhattisgarh": "Chhattisgarh",

        "Karnataka": "Karnataka",
        "Madhya Pradesh": "Madhya Pradesh",
        "Goa": "Goa",
        "Andhra Pradesh": "Andhra Pradesh",
    }

    return aliases.get(state, state)


def load_occurrences():
    """
    Load real GSI manganese occurrences.

    The raw dataset uses:
        statename → state
        latdd     → lat
        londd     → lon
        hostrock  → host_rock

    Returns:
        df, validation report
    """

    if not RAW_CSV.exists():
        raise UserError(MISSING_MSG)

    try:
        raw = pd.read_csv(
            RAW_CSV,
            encoding="utf-8-sig"
        )
    except Exception:
        raw = pd.read_csv(
            RAW_CSV,
            encoding="latin-1"
        )

    # ---------------------------------------------------------
    # FIND COORDINATE COLUMNS
    # ---------------------------------------------------------

    lat_c = (
        _col(raw, ["latdd"])
        or _col(raw, ["latitude"])
        or _col(raw, ["lat"])
    )

    lon_c = (
        _col(raw, ["londd"])
        or _col(raw, ["longitude"])
        or _col(raw, ["lon", "long"])
    )

    if not lat_c or not lon_c:
        raise UserError(
            f"Could not find latitude/longitude columns. "
            f"Found: {list(raw.columns)}"
        )

    # ---------------------------------------------------------
    # FIND STATE COLUMN
    # ---------------------------------------------------------

    state_c = (
        _col(raw, ["statename"])
        or _col(raw, ["state"])
    )

    # ---------------------------------------------------------
    # FIND OTHER INFORMATION COLUMNS
    # ---------------------------------------------------------

    locality_c = _col(raw, ["locality"])

    host_c = (
        _col(raw, ["hostrock"])
        or _col(raw, ["host"])
    )

    formation_c = _col(raw, ["formation"])

    metallogenesis_c = _col(
        raw,
        ["metallogenesis", "metallo"]
    )

    # ---------------------------------------------------------
    # CREATE CLEAN DATAFRAME
    # ---------------------------------------------------------

    df = pd.DataFrame({
        "lat": raw[lat_c].map(_num),
        "lon": raw[lon_c].map(_num),
    })

    if locality_c:
        df["locality"] = raw[locality_c].fillna("").astype(str)
    else:
        df["locality"] = ""

    if state_c:
        df["state"] = raw[state_c].map(_normalize_state)
    else:
        df["state"] = ""

    if host_c:
        df["host_rock"] = raw[host_c].fillna("").astype(str)
    else:
        df["host_rock"] = ""

    if formation_c:
        df["formation"] = raw[formation_c].fillna("").astype(str)
    else:
        df["formation"] = ""

    if metallogenesis_c:
        df["metallogenesis"] = (
            raw[metallogenesis_c]
            .fillna("")
            .astype(str)
        )
    else:
        df["metallogenesis"] = ""

    # ---------------------------------------------------------
    # VALIDATION REPORT
    # ---------------------------------------------------------

    rep = {
        "rows_raw": len(df),
        "missing_coordinates": int(
            df[["lat", "lon"]]
            .isna()
            .any(axis=1)
            .sum()
        ),
        "state_column": state_c,
        "latitude_column": lat_c,
        "longitude_column": lon_c,
    }

    # ---------------------------------------------------------
    # REMOVE MISSING COORDINATES
    # ---------------------------------------------------------

    df = df.dropna(
        subset=["lat", "lon"]
    ).copy()

    # ---------------------------------------------------------
    # INDIA BOUNDING BOX
    # ---------------------------------------------------------

    ok = (
        df["lat"].between(6, 37.5)
        & df["lon"].between(68, 97.5)
    )

    rep["out_of_range_coordinates"] = int(
        (~ok).sum()
    )

    df = df[ok].copy()

    # ---------------------------------------------------------
    # REMOVE EXACT DUPLICATES
    # ---------------------------------------------------------

    n = len(df)

    df = df.drop_duplicates(
        subset=["lat", "lon"]
    ).copy()

    # ---------------------------------------------------------
    # REMOVE VERY CLOSE DUPLICATES
    # ---------------------------------------------------------

    df["_k"] = (
        df["lat"].round(3).astype(str)
        + "_"
        + df["lon"].round(3).astype(str)
    )

    df = (
        df
        .drop_duplicates("_k")
        .drop(columns="_k")
        .reset_index(drop=True)
    )

    rep["duplicates_removed"] = n - len(df)

    rep["rows_clean"] = len(df)

    # ---------------------------------------------------------
    # IMPORTANT:
    # We only require 20+ points for the FULL dataset.
    # Individual states may contain fewer points.
    # ---------------------------------------------------------

    if len(df) < 20:
        raise UserError(
            f"Only {len(df)} valid occurrence points. "
            f"At least 20 total points are needed."
        )

    # ---------------------------------------------------------
    # STATE COUNTS
    # ---------------------------------------------------------

    rep["state_counts"] = (
        df["state"]
        .value_counts()
        .to_dict()
    )

    return df, rep


def study_area(df, state=""):
    """Create study area from selected state or all available occurrences."""

    d = (
        df[
            df["state"]
            .astype(str)
            .str.strip()
            .str.lower()
            .eq(state.strip().lower())
        ]
        if state
        else df
    )

    # We need at least 3 occurrence points to create a meaningful
    # state-level study area.
    if len(d) < 3:
        if state:
            raise UserError(
                f"Not enough manganese occurrences for '{state}'. "
                f"Only {len(d)} occurrence(s) are available. "
                f"Please select another study area."
            )
        else:
            raise UserError(
                f"Only {len(d)} valid occurrence points are available. "
                f"At least 3 are required."
            )

    # Add a small geographic buffer around the occurrence points
    bbox = [
        d["lon"].min() - 0.25,
        d["lat"].min() - 0.25,
        d["lon"].max() + 0.25,
        d["lat"].max() + 0.25
    ]

    return d, bbox
def background_points(
    occ,
    bbox,
    ratio=3,
    min_km=10,
    seed=42
):
    """
    Generate random background/pseudo-absence points.

    These are NOT confirmed barren locations.
    """

    rng = np.random.default_rng(seed)

    if len(occ) == 0:
        raise UserError(
            "Cannot generate background points because "
            "there are no occurrence points."
        )

    tree = BallTree(
        np.radians(
            occ[["lat", "lon"]].values
        ),
        metric="haversine"
    )

    pts = []

    target = max(
        len(occ) * ratio,
        10
    )

    for _ in range(200):

        c = np.column_stack([
            rng.uniform(
                bbox[1],
                bbox[3],
                target
            ),

            rng.uniform(
                bbox[0],
                bbox[2],
                target
            )
        ])

        d, _i = tree.query(
            np.radians(c),
            k=1
        )

        valid = (
            d[:, 0] * 6371
            >= min_km
        )

        pts.extend(
            c[valid].tolist()
        )

        if len(pts) >= target:
            break

    if not pts:
        raise UserError(
            "Could not generate enough background points "
            "inside the selected study area. "
            "Try selecting a larger study area."
        )

    return pd.DataFrame(
        pts[:target],
        columns=["lat", "lon"]
    )