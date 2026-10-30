"""Baseline and machine-learning models."""
from __future__ import annotations
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from .features import chronological_split, get_xy

def threshold_baseline(frame):
    # Transparent demonstration rule; thresholds must be calibrated for a real region/sensor.
    score = (
        (frame["cape_jkg"] >= 800).astype(int)
        + (frame["humidity_pct"] >= 75).astype(int)
        + (frame["precipitation_mm"] >= 1.0).astype(int)
        + (frame["lightning_count"] >= 1).astype(int)
    )
    return (score >= 2).astype(int).to_numpy()

def train_and_evaluate(frame):
    train, test = chronological_split(frame)
    X_train, y_train = get_xy(train)
    X_test, y_test = get_xy(test)
    model = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("classifier", LogisticRegression(class_weight="balanced", max_iter=1000, random_state=42)),
    ])
    model.fit(X_train, y_train)
    probabilities = model.predict_proba(X_test)[:, 1]
    predictions = (probabilities >= 0.5).astype(int)
    baseline = threshold_baseline(test)
    return model, test, y_test.to_numpy(), probabilities, predictions, baseline
