"""Dataset loading and validation utilities."""
from __future__ import annotations

from pathlib import Path
import subprocess
import sys

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

    dataset_path = Path(path)
    if not dataset_path.exists():
        project_root = Path(__file__).resolve().parents[1]
        generator = project_root / "scripts" / "generate_demo_data.py"
        if not generator.exists():
            raise FileNotFoundError(
                "Demo dataset and its generator are both missing from the deployment."
            )
        try:
            subprocess.run(
                [sys.executable, str(generator)],
                cwd=project_root,
                check=True,
                capture_output=True,
                text=True,
            )
        except subprocess.CalledProcessError as exc:
            details = (exc.stderr or exc.stdout or "No details returned").strip()
            raise RuntimeError(f"Could not generate demo dataset: {details}") from exc

    if not dataset_path.exists():
        raise FileNotFoundError(f"Demo dataset was not created at {dataset_path}")

    return validate_dataset(pd.read_csv(dataset_path)), "Synthetic demonstration dataset"
