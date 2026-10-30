"""Feature preparation and chronological split."""
import pandas as pd
from .data import FEATURE_COLUMNS

def chronological_split(frame: pd.DataFrame, test_fraction: float = 0.25):
    data = frame.sort_values("timestamp").reset_index(drop=True)
    unique_times = data["timestamp"].drop_duplicates().sort_values().reset_index(drop=True)
    split_at = int(len(unique_times) * (1 - test_fraction))
    if split_at <= 0 or split_at >= len(unique_times):
        raise ValueError("Dataset is too small for the requested split")
    cutoff = unique_times.iloc[split_at]
    train = data[data["timestamp"] < cutoff]
    test = data[data["timestamp"] >= cutoff]
    if train.empty or test.empty:
        raise ValueError("Chronological split produced an empty partition")
    if train["storm_event"].nunique() < 2:
        raise ValueError("Training period must contain both event classes")
    return train.reset_index(drop=True), test.reset_index(drop=True)

def get_xy(frame: pd.DataFrame):
    return frame[FEATURE_COLUMNS].astype(float), frame["storm_event"].astype(int)
