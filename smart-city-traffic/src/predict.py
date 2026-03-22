"""
predict.py
──────────
Provides a clean prediction interface for a given date/time and junction.
"""

import os
import joblib
import numpy as np
import pandas as pd

import sys
sys.path.insert(0, os.path.dirname(__file__))
from feature_engineering import extract_features, get_feature_columns

MODEL_PATH    = "models/traffic_model.pkl"
FEAT_COL_PATH = "models/feature_cols.pkl"


def load_model():
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            "No trained model found. Run main.py or train_model.py first."
        )
    model     = joblib.load(MODEL_PATH)
    feat_cols = joblib.load(FEAT_COL_PATH)
    return model, feat_cols


def predict_single(
    datetime_str: str,
    junction: int,
    model=None,
    feat_cols=None,
) -> float:
    """
    Predict vehicle count for a single (datetime, junction) pair.

    Parameters
    ----------
    datetime_str : str
        e.g. "2024-03-15 08:00"
    junction : int
        1, 2, 3, or 4

    Returns
    -------
    float  – predicted number of vehicles
    """
    if model is None:
        model, feat_cols = load_model()

    dt = pd.to_datetime(datetime_str)
    row = pd.DataFrame({"DateTime": [dt], "Junction": [junction], "Vehicles": [0]})
    row = extract_features(row)

    # Align columns
    X = np.zeros((1, len(feat_cols)))
    for i, col in enumerate(feat_cols):
        if col in row.columns:
            X[0, i] = row[col].values[0]

    pred = model.predict(X)[0]
    return max(0.0, pred)


def predict_range(
    start: str,
    end: str,
    junction: int,
    freq: str = "H",
    model=None,
    feat_cols=None,
) -> pd.DataFrame:
    """
    Predict vehicle counts over a date range.

    Returns
    -------
    pd.DataFrame with columns: DateTime, Junction, PredictedVehicles
    """
    if model is None:
        model, feat_cols = load_model()

    dates = pd.date_range(start=start, end=end, freq=freq)
    rows  = pd.DataFrame({"DateTime": dates, "Junction": junction, "Vehicles": 0})
    rows  = extract_features(rows)

    X = np.zeros((len(rows), len(feat_cols)))
    for i, col in enumerate(feat_cols):
        if col in rows.columns:
            X[:, i] = rows[col].values

    preds = np.maximum(0, model.predict(X))
    result = pd.DataFrame({
        "DateTime":          dates,
        "Junction":          junction,
        "PredictedVehicles": preds.round().astype(int),
    })
    return result


if __name__ == "__main__":
    # Quick demo
    print("🚦 Traffic Prediction Demo\n")
    val = predict_single("2024-08-15 08:30", junction=1)
    print(f"  Junction 1 | 2024-08-15 08:30 → {val:.0f} vehicles")

    df_range = predict_range("2024-08-15 00:00", "2024-08-15 23:00", junction=2)
    print(f"\n  Junction 2 – Hourly forecast for 2024-08-15:")
    print(df_range[["DateTime", "PredictedVehicles"]].to_string(index=False))
