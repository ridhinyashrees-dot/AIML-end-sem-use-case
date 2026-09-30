import pandas as pd
from pathlib import Path

# --------------------------------------------------
# 1. Load dataset
# --------------------------------------------------

file_path = Path("data/raw/manganese_deposits.csv")

df = pd.read_csv(file_path)

print("=" * 60)
print("MANGANESE DATASET VALIDATION")
print("=" * 60)

# --------------------------------------------------
# 2. Basic information
# --------------------------------------------------

print("\n1. ROW COUNT")
print("-" * 60)
print("Total rows:", len(df))

print("\n2. COLUMN NAMES")
print("-" * 60)
print(list(df.columns))

# --------------------------------------------------
# 3. Required columns
# --------------------------------------------------

required_columns = [
    "locality",
    "statename",
    "toposheet",
    "latdd",
    "londd",
    "hostrock",
    "formation",
    "metallogenesis",
    "morphogenesis"
]

print("\n3. REQUIRED COLUMNS")
print("-" * 60)

for col in required_columns:
    if col in df.columns:
        print(f"✓ {col}")
    else:
        print(f"✗ MISSING: {col}")

# --------------------------------------------------
# 4. Missing coordinates
# --------------------------------------------------

print("\n4. MISSING COORDINATES")
print("-" * 60)

missing_lat = df["latdd"].isna().sum()
missing_lon = df["londd"].isna().sum()

print("Missing latitude :", missing_lat)
print("Missing longitude:", missing_lon)

missing_both = df[["latdd", "londd"]].isna().any(axis=1).sum()

print("Rows missing either coordinate:", missing_both)

# --------------------------------------------------
# 5. Invalid coordinate values
# --------------------------------------------------

print("\n5. COORDINATE RANGE")
print("-" * 60)

print("Latitude minimum :", df["latdd"].min())
print("Latitude maximum :", df["latdd"].max())

print("Longitude minimum:", df["londd"].min())
print("Longitude maximum:", df["londd"].max())

# India approximate bounds
valid_coordinates = (
    df["latdd"].between(6, 37.5)
    & df["londd"].between(68, 97.5)
)

print("\nCoordinates inside India bounding box:",
      valid_coordinates.sum())

print("Coordinates outside India bounding box:",
      (~valid_coordinates).sum())

# --------------------------------------------------
# 6. Duplicate rows
# --------------------------------------------------

print("\n6. DUPLICATES")
print("-" * 60)

full_duplicates = df.duplicated().sum()

coordinate_duplicates = df.duplicated(
    subset=["latdd", "londd"]
).sum()

print("Exact duplicate rows:", full_duplicates)
print("Duplicate coordinate rows:", coordinate_duplicates)

# --------------------------------------------------
# 7. Unique locations
# --------------------------------------------------

print("\n7. UNIQUE LOCATIONS")
print("-" * 60)

print("Unique localities:", df["locality"].nunique())
print("Unique coordinates:",
      df[["latdd", "londd"]].drop_duplicates().shape[0])

# --------------------------------------------------
# 8. States
# --------------------------------------------------

print("\n8. STATES")
print("-" * 60)

states = df["statename"].value_counts()

print(states.to_string())

print("\nTotal unique state names:", df["statename"].nunique())

# --------------------------------------------------
# 9. Commodity
# --------------------------------------------------

print("\n9. COMMODITY")
print("-" * 60)

print(df["commodity"].value_counts(dropna=False).to_string())

# --------------------------------------------------
# 10. Missing values
# --------------------------------------------------

print("\n10. MISSING VALUES")
print("-" * 60)

missing_values = df.isna().sum()

print(
    missing_values[
        missing_values > 0
    ].to_string()
    if (missing_values > 0).any()
    else "No missing values found."
)

# --------------------------------------------------
# 11. Dataset summary
# --------------------------------------------------

print("\n" + "=" * 60)
print("FINAL VALIDATION SUMMARY")
print("=" * 60)

print("Total rows                 :", len(df))
print("Exact duplicate rows      :", full_duplicates)
print("Duplicate coordinates     :", coordinate_duplicates)
print("Missing latitude          :", missing_lat)
print("Missing longitude         :", missing_lon)
print("Outside India coordinates:", (~valid_coordinates).sum())
print("Unique states             :", df["statename"].nunique())
print("Unique localities         :", df["locality"].nunique())

print("=" * 60)