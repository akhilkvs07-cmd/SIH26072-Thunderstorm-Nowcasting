"""Feature preparation and chronological split."""
from __future__ import annotations
import pandas as pd
from .data import FEATURE_COLUMNS

def chronological_split(frame: pd.DataFrame, test_fraction: float = 0.25):
    data = frame.sort_values("timestamp").reset_index(drop=True)
    split_at = int(len(data) * (1 - test_fraction))
    if split_at <= 0 or split_at >= len(data):
        raise ValueError("Dataset is too small for the requested split")
    train, test = data.iloc[:split_at], data.iloc[split_at:]
    if train["storm_event"].nunique() < 2:
        raise ValueError("Training period must contain both event classes")
    return train, test

def get_xy(frame: pd.DataFrame):
    return frame[FEATURE_COLUMNS].astype(float), frame["storm_event"].astype(int)
