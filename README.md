# SIH26072 — AI/ML-Based Nowcasting of Thunderstorms and Lightning

A reproducible prototype for short-horizon thunderstorm risk nowcasting. The application compares Logistic Regression, Random Forest and HistGradientBoosting against a transparent threshold baseline, visualizes sample risk by location, reports event-detection metrics, and can retrieve separate live atmospheric context from Open-Meteo.

> **Research/demo limitation:** The included demo dataset is synthetic and is clearly labelled as such. It is for demonstrating the pipeline and software only—not for operational weather warnings. Do not interpret demo predictions as real forecasts. Operational use requires quality-controlled historical radar, satellite, lightning and/or station observations with verified event labels.

## Features

- Streamlit dashboard with configurable demo locations and forecast horizons (30, 60 and 120 minutes).
- Reproducible synthetic sample dataset for local demonstration.
- Train/test split by time to reduce temporal leakage.
- Logistic Regression, Random Forest and HistGradientBoosting model comparison plus an interpretable threshold baseline.
- ROC curve, calibration curve, permutation-based feature importance and model leaderboard.
- Optional on-demand Open-Meteo current atmospheric context for Bengaluru, Chennai, Hyderabad and Mumbai. This feed is displayed separately from model training and evaluation.
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

The optional Open-Meteo panel retrieves current model-based atmospheric fields for context. It does not provide verified lightning detections or event labels, and it is deliberately not passed into the trained classifier. Radar reflectivity, satellite cloud-top products and lightning-network observations remain future integrations; they are not represented as connected in this prototype.

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
