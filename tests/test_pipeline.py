import pandas as pd
from src.data import validate_dataset
from src.evaluation import metric_summary

def test_validation_and_metrics():
    frame = pd.DataFrame({
        "timestamp": ["2025-01-01T00:00:00Z", "2025-01-01T01:00:00Z"],
        "latitude": [12.9, 13.0], "longitude": [77.5, 77.6],
        "temperature_c": [28, 29], "humidity_pct": [80, 82],
        "pressure_hpa": [1008, 1007], "wind_speed_ms": [4, 5],
        "precipitation_mm": [1, 2], "cape_jkg": [900, 1200],
        "lightning_count": [0, 2], "storm_event": [0, 1]
    })
    clean = validate_dataset(frame)
    assert len(clean) == 2
    metrics = metric_summary([0, 1], [0, 1])
    assert metrics["Precision"] == 1.0
    assert metrics["Recall"] == 1.0
