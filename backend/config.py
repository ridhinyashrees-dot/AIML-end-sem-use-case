import os
from pathlib import Path
from dotenv import load_dotenv
ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")
RAW_CSV = ROOT / "data" / "raw" / "manganese_deposits.csv"
PROC = ROOT / "data" / "processed"
OUT = ROOT / "data" / "outputs"
MODELS = ROOT / "models"
EE_PROJECT = os.getenv("EE_PROJECT", "")
S2_START = os.getenv("S2_START", "2023-01-01")
S2_END = os.getenv("S2_END", "2024-12-31")
STUDY_STATE = os.getenv("STUDY_STATE", "").strip()
MAX_GRID_CELLS = 15000
BLOCK_DEG = 0.25          # spatial CV block size (degrees)
LOW_T, HIGH_T = 0.40, 0.70  # probability thresholds: LOW<0.40<=MODERATE<0.70<=HIGH
FEATURES = ["B2", "B3", "B4", "B8", "B11", "B12", "NDVI", "SWIR_ratio", "iron_oxide",
            "ferrous", "elevation", "slope", "aspect"]

class UserError(Exception):
    """Error with a message that is safe/understandable to show in the dashboard."""
