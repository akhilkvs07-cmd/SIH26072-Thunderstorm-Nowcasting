# Data dictionary

| Field | Type | Unit / format | Description |
|---|---|---|---|
| timestamp | datetime | ISO-8601 | Observation timestamp; parsed as UTC |
| latitude | float | degrees | Observation latitude |
| longitude | float | degrees | Observation longitude |
| temperature_c | float | °C | Near-surface air temperature |
| humidity_pct | float | % | Relative humidity |
| pressure_hpa | float | hPa | Surface atmospheric pressure |
| wind_speed_ms | float | m/s | Wind speed |
| precipitation_mm | float | mm | Recent accumulated precipitation; define accumulation interval |
| cape_jkg | float | J/kg | Convective Available Potential Energy |
| lightning_count | integer-like | count | Lightning detections in a documented preceding window |
| storm_event | integer | 0 or 1 | Independently labelled event in the specified future horizon |

All numeric values must be present in the current prototype. Extend the validation and preprocessing logic before using datasets with missing values, multiple stations, different time zones or nonuniform sampling intervals.
