"""
feature_engineering.py
───────────────────────
Extracts time-based and calendar features from the cleaned dataset.
"""

import pandas as pd
import numpy as np


# Indian public holidays (approximate; extend as needed)
INDIAN_HOLIDAYS = {
    # New Year's Day
    "01-01",
    # Republic Day
    "01-26",
    # Holi (approx)
    "03-25",
    # Good Friday (approx)
    "04-18",
    # Ambedkar Jayanti
    "04-14",
    # Labour Day
    "05-01",
    # Independence Day
    "08-15",
    # Gandhi Jayanti
    "10-02",
    # Dussehra (approx)
    "10-12",
    # Diwali (approx)
    "11-01",
    # Christmas
    "12-25",
}


def is_holiday(date: pd.Timestamp) -> int:
    """Return 1 if the date is a known public holiday, else 0."""
    key = date.strftime("%m-%d")
    return int(key in INDIAN_HOLIDAYS)


def extract_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create the full feature matrix from the cleaned DataFrame.

    New columns added
    -----------------
    Hour, DayOfWeek, Day, Month, Year,
    IsWeekend, IsHoliday,
    Quarter, WeekOfYear,
    Sin_Hour, Cos_Hour,          ← cyclic encoding
    Sin_Month, Cos_Month,
    Sin_DayOfWeek, Cos_DayOfWeek,
    IsRushHour, IsPeakSeason
    """
    df = df.copy()
    dt = df["DateTime"]

    # ── Basic calendar features ──────────────────────────────────────────────
    df["Hour"]       = dt.dt.hour
    df["DayOfWeek"]  = dt.dt.dayofweek          # 0=Mon … 6=Sun
    df["Day"]        = dt.dt.day
    df["Month"]      = dt.dt.month
    df["Year"]       = dt.dt.year
    df["Quarter"]    = dt.dt.quarter
    df["WeekOfYear"] = dt.dt.isocalendar().week.astype(int)

    # ── Binary flags ─────────────────────────────────────────────────────────
    df["IsWeekend"]  = (df["DayOfWeek"] >= 5).astype(int)
    df["IsHoliday"]  = dt.apply(is_holiday)

    # Rush hours: 7–9 AM and 17–19 (5–7 PM)
    df["IsRushHour"] = df["Hour"].isin(range(7, 10)).astype(int) | \
                       df["Hour"].isin(range(17, 20)).astype(int)

    # Peak season: Oct–Dec (festive + year-end)
    df["IsPeakSeason"] = df["Month"].isin([10, 11, 12]).astype(int)

    # ── Cyclic encoding (avoids 23→0 gap for hours, etc.) ────────────────────
    df["Sin_Hour"]      = np.sin(2 * np.pi * df["Hour"]      / 24)
    df["Cos_Hour"]      = np.cos(2 * np.pi * df["Hour"]      / 24)
    df["Sin_Month"]     = np.sin(2 * np.pi * df["Month"]     / 12)
    df["Cos_Month"]     = np.cos(2 * np.pi * df["Month"]     / 12)
    df["Sin_DayOfWeek"] = np.sin(2 * np.pi * df["DayOfWeek"] / 7)
    df["Cos_DayOfWeek"] = np.cos(2 * np.pi * df["DayOfWeek"] / 7)

    # ── One-hot encode Junction ───────────────────────────────────────────────
    junction_dummies = pd.get_dummies(df["Junction"], prefix="Junction").astype(int)
    df = pd.concat([df, junction_dummies], axis=1)

    print(f"✅ Features extracted. Total columns: {df.shape[1]}")
    return df


def get_feature_columns() -> list:
    """Return the ordered list of feature columns used for modelling."""
    base = [
        "Hour", "DayOfWeek", "Day", "Month", "Year",
        "Quarter", "WeekOfYear",
        "IsWeekend", "IsHoliday", "IsRushHour", "IsPeakSeason",
        "Sin_Hour", "Cos_Hour",
        "Sin_Month", "Cos_Month",
        "Sin_DayOfWeek", "Cos_DayOfWeek",
    ]
    junctions = [f"Junction_{i}" for i in range(1, 5)]
    return base + junctions


if __name__ == "__main__":
    import os
    path = "data/processed/cleaned_data.csv"
    if os.path.exists(path):
        df = pd.read_csv(path, parse_dates=["DateTime"])
        df_feat = extract_features(df)
        print(df_feat[get_feature_columns()].head(3).to_string())
    else:
        print("Run data_preprocessing.py first.")
