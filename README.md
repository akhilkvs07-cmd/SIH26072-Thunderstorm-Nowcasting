# SIH26072 — AI/ML-Based Nowcasting of Thunderstorms and Lightning

A reproducible prototype for short-horizon thunderstorm risk nowcasting. The application compares Logistic Regression, Random Forest and HistGradientBoosting against a transparent threshold baseline, visualizes sample risk by location, reports event-detection metrics, and can retrieve separate live atmospheric context from Open-Meteo.

> **Research/demo limitation:** The included demo dataset is synthetic and is clearly labelled as such. It is for demonstrating the pipeline and software only—not for operational weather warnings. Do not interpret demo predictions as real forecasts. Operational use requires quality-controlled historical radar, satellite, lightning and/or station observations with verified event labels.

## Features

- Streamlit dashboard with configurable demo locations and forecast horizons (30, 60 and 120 minutes).
- Reproducible synthetic sample dataset for local demonstration.
- Train/test split by time to reduce temporal leakage.
- Logistic Regression, Random Forest and HistGradientBoosting model comparison plus an interpretable threshold baseline.
- ROC curve, calibration curve, permutation-based feature importance and model leaderboard.
- On-demand Open-Meteo 24-hour numerical forecast for selected Indian cities, including CAPE, convective inhibition, lifted index, rainfall probability and wind gusts.
- IMD API adapters for district nowcasts, station nowcasts and district warnings. Requires IMD portal access/credentials if the endpoint is protected.
- Separate source-labelled panels for model forecasts, official warnings and ML outputs.
- Dark responsive command-centre dashboard with risk map, analytics, alert desk, observation explorer and CSV exports.
- Precision, recall, F1, false-alarm rate, missed-event rate and confusion matrix.
- Risk probability and risk category visualization (demonstration probabilities only until trained and validated on real labelled data).
- CSV upload path for appropriately labelled historical observations.
- Model and evaluation artifacts saved locally.

## Quick start

### 1. Requirements
- Python 3.10 or newer
- pip

### 2. Install
```bash
git clone https://github.com/akhilkvs07-cmd/SIH26072-Thunderstorm-Nowcasting.git
cd SIH26072-Thunderstorm-Nowcasting
python -m venv .venv
# Windows PowerShell:
.venv\Scripts\Activate.ps1
# macOS/Linux:
# source .venv/bin/activate
pip install -r requirements.txt
```

### 3. Generate demo data and train
```bash
python scripts/generate_demo_data.py
python scripts/train_model.py
```

### 4. Run dashboard
```bash
streamlit run app.py
```

## Dataset schema

The modelling pipeline expects a CSV with these columns. `storm_event` must represent the event observed in the future horizon defined by your dataset—not simply the current weather condition:

| Column | Meaning |
|---|---|
| timestamp | Observation time in ISO-8601 format |
| latitude, longitude | Observation location |
| temperature_c | Near-surface temperature (°C) |
| humidity_pct | Relative humidity (%) |
| pressure_hpa | Surface pressure (hPa) |
| wind_speed_ms | Wind speed (m/s) |
| precipitation_mm | Recent precipitation (mm) |
| cape_jkg | Convective Available Potential Energy (J/kg) |
| lightning_count | Lightning detections in the preceding observation window |
| storm_event | Binary target: event observed in the selected future horizon (0/1) |

The target definition and observation window must be consistent across all rows. Do not combine incompatible sensors or label definitions without documenting the conversion.

## Using real observations

Replace the demo CSV with a quality-controlled dataset that follows the schema above. The dashboard accepts a CSV upload. For a credible evaluation:
1. Record source, licensing, geographic coverage, sensor resolution and missing-data handling.
2. Define the event (e.g. lightning occurrence or thunderstorm) and forecast horizon explicitly.
3. Split chronologically, preferably holding out later dates and/or unseen locations.
4. Compare against a simple baseline and report false alarms and missed events.
5. Calibrate probabilities on a separate validation set before operational interpretation.

The live Open-Meteo panel retrieves numerical weather prediction fields for the selected location. IMD adapters query official district/station nowcasts and warnings. These products are displayed as source data and are not silently used as labels or merged into the synthetic training dataset. Configure `IMD_API_TOKEN` in the deployment environment if required by the IMD API account. Confirm the actual authentication scheme and product permissions with IMD before production use. Radar imagery and lightning products may require specific IMD/MOSDAC permissions or another licensed provider. MOSDAC satellite products require account credentials and dataset access. NASA IMERG Early Run provides near-real-time precipitation estimates with roughly four-hour latency; it is not a substitute for lightning detection. None of these restricted feeds are claimed as active until credentials and data retrieval are verified.

## Live data setup

1. Open the [IMD API portal](https://api.imd.gov.in/public/index.php), create an account and request the products needed: district/station nowcast, district warnings, AWS/ARG, radar and lightning where available.
2. Add the credential to Render as `IMD_API_TOKEN` (never commit secrets to GitHub). The adapter currently uses a Bearer token; adjust the header only if IMD issues a different authentication method.
3. Use the Live Weather tab to retrieve Open-Meteo 24-hour forecast-model fields. These are global model outputs, not ground-truth observations.
4. For model training, obtain time-aligned lightning detections and independently verified thunderstorm event labels. Store event timestamps and locations, then construct separate targets for each lead time (e.g. 30, 60 and 120 minutes).
5. Keep training, validation and test periods strictly chronological. Evaluate false alarms, missed events, probability calibration and lead-time skill on independent real events.

**Production gate:** The dashboard is not an operational warning service until IMD/MOSDAC credentials and feed access are confirmed, the lightning/observation feeds are live, the ML model is trained and validated on real labelled data, and alert delivery/monitoring are tested. Open-Meteo forecast data alone cannot establish observed lightning strikes.

## Project structure

```
app.py
src/
  __init__.py
  data.py
  features.py
  modeling.py
  evaluation.py
scripts/
  generate_demo_data.py
  train_model.py
data/
  .gitkeep
models/
  .gitkeep
reports/
  .gitkeep
tests/
  test_pipeline.py
requirements.txt
```

## Evaluation definitions

- **Precision:** TP / (TP + FP)
- **Recall:** TP / (TP + FN)
- **False-alarm rate:** FP / (FP + TN)
- **Missed-event rate:** FN / (FN + TP)
- **Lead time:** time between an issued prediction and the event; requires event timestamps and archived prediction timestamps.

Metrics are calculated on the held-out test partition. Synthetic-data scores are software checks, not evidence of real-world forecasting skill.

## Responsible use

This is an academic prototype, not a certified meteorological warning system. Use official meteorological agencies for safety decisions. A deployed warning service needs validated data feeds, monitoring, uncertainty communication, and domain-expert review.

## License

MIT License. See [LICENSE](LICENSE).
