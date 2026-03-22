"""
data_preprocessing.py
─────────────────────
Handles loading, cleaning, and saving of the raw traffic dataset.
"""

import pandas as pd
import numpy as np
import os


def load_data(filepath: str) -> pd.DataFrame:
    """Load raw traffic CSV data."""
    if not os.path.exists(filepath):
        raise FileNotFoundError(
            f"Dataset not found at '{filepath}'.\n"
            "Please download from:\n"
            "https://drive.google.com/file/d/1y61cDyuO9Zrp1fSchWcAmCxk0B6SMx7X/\n"
            "and save as: data/raw/traffic_data.csv"
        )
    df = pd.read_csv(filepath)
    print(f"✅ Loaded data: {df.shape[0]:,} rows × {df.shape[1]} columns")
    return df


def inspect_data(df: pd.DataFrame) -> None:
    """Print a summary of the raw dataframe."""
    print("\n📋 Dataset Info:")
    print(f"  Shape     : {df.shape}")
    print(f"  Columns   : {list(df.columns)}")
    print(f"\n  Dtypes:\n{df.dtypes}")
    print(f"\n  Missing values:\n{df.isnull().sum()}")
    print(f"\n  Sample rows:\n{df.head(3).to_string()}")


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean the raw dataframe:
      - Parse DateTime column
      - Drop duplicates & rows with missing Vehicles
      - Ensure Junction is integer (1–4)
    """
    df = df.copy()

    # ── DateTime parsing ────────────────────────────────────────────────────
    dt_col = None
    for c in df.columns:
        if "date" in c.lower() or "time" in c.lower():
            dt_col = c
            break

    if dt_col is None:
        raise ValueError("No DateTime-like column found in the dataset.")

    df[dt_col] = pd.to_datetime(df[dt_col], infer_datetime_format=True)
    df.rename(columns={dt_col: "DateTime"}, inplace=True)

    # ── Sort by time ─────────────────────────────────────────────────────────
    df.sort_values("DateTime", inplace=True)
    df.reset_index(drop=True, inplace=True)

    # ── Drop duplicates ──────────────────────────────────────────────────────
    before = len(df)
    df.drop_duplicates(inplace=True)
    print(f"  Dropped {before - len(df)} duplicate rows.")

    # ── Handle missing Vehicles ──────────────────────────────────────────────
    vehicles_col = [c for c in df.columns if "vehicle" in c.lower()][0]
    df.rename(columns={vehicles_col: "Vehicles"}, inplace=True)

    missing_v = df["Vehicles"].isnull().sum()
    if missing_v > 0:
        # Impute with junction-hour median
        df["Vehicles"] = df.groupby(
            [df["DateTime"].dt.hour, "Junction"]
        )["Vehicles"].transform(lambda x: x.fillna(x.median()))
        print(f"  Imputed {missing_v} missing Vehicles values.")

    # ── Ensure Junction column ────────────────────────────────────────────────
    junction_col = [c for c in df.columns if "junction" in c.lower()]
    if junction_col:
        df.rename(columns={junction_col[0]: "Junction"}, inplace=True)
        df["Junction"] = df["Junction"].astype(int)

    print(f"✅ Cleaned data: {df.shape[0]:,} rows remaining.")
    return df


def save_processed(df: pd.DataFrame, filepath: str) -> None:
    """Save cleaned dataframe to CSV."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    df.to_csv(filepath, index=False)
    print(f"✅ Processed data saved → {filepath}")


if __name__ == "__main__":
    raw_path = "data/raw/traffic_data.csv"
    out_path = "data/processed/cleaned_data.csv"

    df_raw = load_data(raw_path)
    inspect_data(df_raw)
    df_clean = clean_data(df_raw)
    save_processed(df_clean, out_path)
