"""Generate a clearly synthetic dataset for local pipeline demonstration."""
from pathlib import Path
import numpy as np
import pandas as pd

rng = np.random.default_rng(26072)
rows = []
start = pd.Timestamp("2025-04-01T00:00:00Z")
locations = [(12.97, 77.59, "Bengaluru"), (13.08, 80.27, "Chennai"), (17.38, 78.49, "Hyderabad"), (19.07, 72.88, "Mumbai")]
for day in range(120):
    for slot in range(8):
        ts = start + pd.Timedelta(days=day, hours=slot * 3)
        for lat, lon, _name in locations:
            warm = 27 + 5 * np.sin((slot / 8) * np.pi) + rng.normal(0, 1.8)
            humidity = np.clip(rng.normal(68 + (warm - 27) * 1.5, 12), 30, 100)
            cape = max(0, rng.gamma(2.0, 250) + (humidity - 65) * 15)
            rain = max(0, rng.gamma(1.1, 1.2) - 0.8)
            wind = max(0, rng.normal(4.5 + rain * 0.5, 2))
            pressure = rng.normal(1008 - rain * 1.2, 3)
            lightning = max(0, int(rng.poisson(max(0.05, cape / 1800 + rain * 0.2))))
            logit = -5.0 + cape / 650 + (humidity - 65) / 18 + rain * 0.65 + lightning * 0.7
            probability = 1 / (1 + np.exp(-logit))
            event = int(rng.random() < probability)
            rows.append([ts, lat + rng.normal(0, .015), lon + rng.normal(0, .015), warm, humidity, pressure, wind, rain, cape, lightning, event])
columns = ["timestamp", "latitude", "longitude", "temperature_c", "humidity_pct", "pressure_hpa", "wind_speed_ms", "precipitation_mm", "cape_jkg", "lightning_count", "storm_event"]
output = Path(__file__).resolve().parents[1] / "data" / "demo_observations.csv"
output.parent.mkdir(parents=True, exist_ok=True)
pd.DataFrame(rows, columns=columns).to_csv(output, index=False)
print(f"Wrote {len(rows)} synthetic rows to {output}")
