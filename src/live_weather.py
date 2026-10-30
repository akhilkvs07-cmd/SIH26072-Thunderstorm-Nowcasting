"""Optional live atmospheric context from the Open-Meteo forecast API.

These are model-based weather fields, not direct radar or lightning observations.
They are intentionally kept separate from the labelled training/evaluation data.
"""
from __future__ import annotations

import json
from urllib.parse import urlencode
from urllib.request import Request, urlopen

LOCATIONS = {
    "Bengaluru": (12.9716, 77.5946),
    "Chennai": (13.0827, 80.2707),
    "Hyderabad": (17.3850, 78.4867),
    "Mumbai": (19.0760, 72.8777),
}


def fetch_current_conditions(timeout: int = 12) -> list[dict]:
    """Fetch current modelled conditions for the supported demonstration cities."""
    results = []
    for city, (latitude, longitude) in LOCATIONS.items():
        params = urlencode({
            "latitude": latitude,
            "longitude": longitude,
            "current": (
                "temperature_2m,relative_humidity_2m,precipitation,"
                "rain,showers,pressure_msl,wind_speed_10m,wind_gusts_10m"
            ),
            "hourly": "cape",
            "forecast_days": 1,
            "timezone": "UTC",
        })
        url = f"https://api.open-meteo.com/v1/forecast?{params}"
        request = Request(url, headers={"User-Agent": "StormWatch-SIH26072/1.0"})
        with urlopen(request, timeout=timeout) as response:
            payload = json.loads(response.read().decode("utf-8"))
        current = payload.get("current", {})
        hourly = payload.get("hourly", {})
        cape_value = None
        times = hourly.get("time", [])
        capes = hourly.get("cape", [])
        current_time = current.get("time")
        if current_time in times:
            idx = times.index(current_time)
            if idx < len(capes):
                cape_value = capes[idx]
        results.append({
            "region": city,
            "latitude": latitude,
            "longitude": longitude,
            "timestamp_utc": current.get("time"),
            "temperature_c": current.get("temperature_2m"),
            "humidity_pct": current.get("relative_humidity_2m"),
            "precipitation_mm": current.get("precipitation"),
            "rain_mm": current.get("rain"),
            "showers_mm": current.get("showers"),
            "pressure_hpa": current.get("pressure_msl"),
            "wind_speed_kmh": current.get("wind_speed_10m"),
            "wind_gusts_kmh": current.get("wind_gusts_10m"),
            "cape_jkg": cape_value,
            "source": "Open-Meteo forecast model",
        })
    return results
