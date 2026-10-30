"""Live meteorological data adapters.

Open-Meteo provides modelled atmospheric forecasts. IMD adapters expose official
nowcast/warning endpoints when the deployment has valid access. Neither feed is
silently converted into a labelled ML training target.
"""
from __future__ import annotations

import json
import os
from urllib.parse import urlencode
from urllib.request import Request, urlopen
from urllib.error import HTTPError, URLError

OPEN_METEO = "https://api.open-meteo.com/v1/forecast"
IMD_BASE = "https://api.imd.gov.in/api/v1"


def _get_json(url: str, headers: dict | None = None, timeout: int = 18):
    req = Request(url, headers={"User-Agent": "StormWatch-SIH26072/1.0", **(headers or {})})
    with urlopen(req, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def fetch_forecast(latitude: float, longitude: float, label: str = "Selected location") -> dict:
    """Fetch current and next-24-hour model forecast fields for a coordinate."""
    params = urlencode({
        "latitude": latitude,
        "longitude": longitude,
        "current": (
            "temperature_2m,relative_humidity_2m,precipitation,rain,showers,"
            "pressure_msl,wind_speed_10m,wind_gusts_10m,cloud_cover"
        ),
        "hourly": (
            "temperature_2m,relative_humidity_2m,precipitation_probability,"
            "precipitation,rain,showers,pressure_msl,wind_speed_10m,"
            "wind_gusts_10m,cloud_cover,cape,convective_inhibition,"
            "lifted_index,boundary_layer_height"
        ),
        "forecast_hours": 24,
        "timezone": "Asia/Kolkata",
    })
    payload = _get_json(f"{OPEN_METEO}?{params}")
    return {"location": label, "latitude": latitude, "longitude": longitude,
            "current": payload.get("current", {}),
            "hourly": payload.get("hourly", {}),
            "source": "Open-Meteo numerical weather prediction",
            "source_url": "https://open-meteo.com/en/docs"}


def fetch_imd_feed(endpoint: str, params: dict | None = None) -> dict:
    """Fetch an IMD public API endpoint; pass IMD_API_TOKEN if account access requires it."""
    allowed = {
        "districtnowcast", "stationnowcast", "districtwarning",
        "current_wx", "aws_data", "lightning"
    }
    if endpoint not in allowed:
        raise ValueError("Unsupported IMD endpoint")
    token = os.getenv("IMD_API_TOKEN", "").strip()
    headers = {"Accept": "application/json"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    query = urlencode(params or {})
    url = f"{IMD_BASE}/{endpoint}" + (f"?{query}" if query else "")
    try:
        data = _get_json(url, headers=headers)
    except HTTPError as exc:
        if exc.code in (401, 403):
            raise RuntimeError(
                "IMD denied access. Register at https://api.imd.gov.in/public/index.php "
                "and configure IMD_API_TOKEN in Render environment variables."
            ) from exc
        raise RuntimeError(f"IMD endpoint returned HTTP {exc.code}") from exc
    except (URLError, TimeoutError) as exc:
        raise RuntimeError(f"Could not reach IMD API: {exc}") from exc
    return {"endpoint": endpoint, "data": data,
            "source": "India Meteorological Department",
            "source_url": "https://api.imd.gov.in/public/api_reference.html"}
