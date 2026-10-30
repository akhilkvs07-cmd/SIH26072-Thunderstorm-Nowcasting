"""Dataset loading and validation utilities."""
from __future__ import annotations
import pandas as pd

REQUIRED_COLUMNS = [
    "timestamp", "latitude", "longitude", "temperature_c", "humidity_pct",
    "pressure_hpa", "wind_speed_ms", "precipitation_mm", "cape_jkg",
    "lightning_count", "storm_event",
]
FEATURE_COLUMNS = [
    "temperature_c", "humidity_pct", "pressure_hpa", "wind_speed_ms",
    "precipitation_mm", "cape_jkg", "lightning_count",
]

def validate_dataset(frame: pd.DataFrame) -> pd.DataFrame:
    missing = [c for c in REQUIRED_COLUMNS if c not in frame.columns]
    if missing:
        raise ValueError("Missing required columns: " + ", ".join(missing))
    data = frame.copy()
    data["timestamp"] = pd.to_datetime(data["timestamp"], errors="coerce", utc=True)
    if data["timestamp"].isna().any():
        raise ValueError("timestamp contains invalid or missing values")
    for col in REQUIRED_COLUMNS[1:]:
        data[col] = pd.to_numeric(data[col], errors="coerce")
    if data[REQUIRED_COLUMNS[1:]].isna().any().any():
        raise ValueError("Numeric columns contain missing or non-numeric values")
    if not data["storm_event"].isin([0, 1]).all():
        raise ValueError("storm_event must contain only 0 or 1")
    if not data["latitude"].between(-90, 90).all() or not data["longitude"].between(-180, 180).all():
        raise ValueError("Latitude/longitude outside valid geographic ranges")
    return data.sort_values("timestamp").reset_index(drop=True)

def load_dataset(uploaded_file=None, path: str = "data/demo_observations.csv") -> tuple[pd.DataFrame, str]:
    if uploaded_file is not None:
        return validate_dataset(pd.read_csv(uploaded_file)), "Uploaded observations"
    try:
        return validate_dataset(pd.read_csv(path)), "Synthetic demonstration dataset"
    except FileNotFoundError:
        raise FileNotFoundError("Demo dataset not found. Run: python scripts/generate_demo_data.py")
