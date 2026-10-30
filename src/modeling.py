"""Chronological model comparison for thunderstorm-event classification."""
from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import brier_score_loss, roc_auc_score
from .features import chronological_split, get_xy
from .evaluation import metric_summary

def threshold_baseline(frame):
    score = ((frame["cape_jkg"] >= 800).astype(int)
        + (frame["humidity_pct"] >= 75).astype(int)
        + (frame["precipitation_mm"] >= 1.0).astype(int)
        + (frame["lightning_count"] >= 1).astype(int))
    return (score >= 2).astype(int).to_numpy()

def build_models():
    return {
        "Logistic Regression": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("classifier", LogisticRegression(class_weight="balanced", max_iter=1200, random_state=42)),
        ]),
        "Random Forest": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("classifier", RandomForestClassifier(n_estimators=180, max_depth=8, min_samples_leaf=3,
                class_weight="balanced", random_state=42, n_jobs=-1)),
        ]),
        "Hist Gradient Boosting": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("classifier", HistGradientBoostingClassifier(max_iter=120, learning_rate=.08,
                max_leaf_nodes=15, l2_regularization=1.0, random_state=42)),
        ]),
    }

def train_model_zoo(frame):
    train, test = chronological_split(frame)
    X_train, y_train = get_xy(train)
    X_test, y_test = get_xy(test)
    results = {}
    for name, model in build_models().items():
        model.fit(X_train, y_train)
        prob = model.predict_proba(X_test)[:, 1]
        pred = (prob >= .5).astype(int)
        metrics = metric_summary(y_test.to_numpy(), pred)
        metrics["ROC AUC"] = float(roc_auc_score(y_test, prob)) if y_test.nunique() == 2 else float("nan")
        metrics["Brier score"] = float(brier_score_loss(y_test, prob))
        results[name] = {"model": model, "probabilities": prob, "predictions": pred, "metrics": metrics}
    baseline_pred = threshold_baseline(test)
    baseline_metrics = metric_summary(y_test.to_numpy(), baseline_pred)
    return train, test, y_test.to_numpy(), results, baseline_metrics
