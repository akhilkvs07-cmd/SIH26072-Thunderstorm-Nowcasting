"""Train model and save reproducible evaluation report."""
from pathlib import Path
import json
import joblib
import pandas as pd
from src.data import validate_dataset
from src.modeling import train_and_evaluate
from src.evaluation import metric_summary

ROOT = Path(__file__).resolve().parents[1]
data = validate_dataset(pd.read_csv(ROOT / "data" / "demo_observations.csv"))
model, test, y_true, probabilities, predictions, baseline = train_and_evaluate(data)
model_dir = ROOT / "models"; report_dir = ROOT / "reports"
model_dir.mkdir(exist_ok=True); report_dir.mkdir(exist_ok=True)
joblib.dump(model, model_dir / "logistic_regression.joblib")
report = {
    "dataset": "synthetic demonstration data",
    "test_rows": int(len(test)),
    "ml_model": metric_summary(y_true, predictions),
    "threshold_baseline": metric_summary(y_true, baseline),
    "warning": "Synthetic-data metrics are not evidence of real-world forecasting skill."
}
(report_dir / "evaluation.json").write_text(json.dumps(report, indent=2))
print(json.dumps(report, indent=2))
