"""
train_model.py
──────────────
Trains Linear Regression and Random Forest models on the traffic dataset
and saves the best model to disk.
"""

import os
import joblib
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

# Local imports
import sys
sys.path.insert(0, os.path.dirname(__file__))
from feature_engineering import extract_features, get_feature_columns


RANDOM_STATE = 42
TEST_SIZE     = 0.2
MODEL_PATH    = "models/traffic_model.pkl"
SCALER_PATH   = "models/scaler.pkl"


def load_processed(filepath: str) -> pd.DataFrame:
    df = pd.read_csv(filepath, parse_dates=["DateTime"])
    print(f"✅ Loaded processed data: {df.shape}")
    return df


def prepare_matrices(df: pd.DataFrame):
    """Return (X, y, feature_names)."""
    df = extract_features(df)
    feat_cols = get_feature_columns()
    # Keep only columns that exist (Junction dummies vary by dataset)
    feat_cols = [c for c in feat_cols if c in df.columns]

    X = df[feat_cols].values
    y = df["Vehicles"].values
    return X, y, feat_cols


def train_all_models(X_train, y_train):
    """Train multiple models and return a dict {name: fitted_model}."""
    models = {
        "Linear Regression": Pipeline([
            ("scaler", StandardScaler()),
            ("model", LinearRegression()),
        ]),
        "Random Forest": RandomForestRegressor(
            n_estimators=200,
            max_depth=12,
            min_samples_leaf=4,
            n_jobs=-1,
            random_state=RANDOM_STATE,
        ),
        "Gradient Boosting": GradientBoostingRegressor(
            n_estimators=200,
            max_depth=5,
            learning_rate=0.05,
            random_state=RANDOM_STATE,
        ),
    }

    fitted = {}
    for name, model in models.items():
        print(f"  🔄 Training {name} …", end="", flush=True)
        model.fit(X_train, y_train)
        fitted[name] = model
        print(" done")

    return fitted


def select_best(fitted_models: dict, X_val, y_val) -> tuple:
    """Return (best_name, best_model) based on RMSE on validation set."""
    from sklearn.metrics import mean_squared_error

    best_name, best_model, best_rmse = None, None, np.inf
    for name, model in fitted_models.items():
        preds = model.predict(X_val)
        rmse  = np.sqrt(mean_squared_error(y_val, preds))
        print(f"    {name:28s}  Val RMSE = {rmse:.4f}")
        if rmse < best_rmse:
            best_rmse  = rmse
            best_name  = name
            best_model = model

    print(f"\n  🏆 Best model: {best_name}  (RMSE = {best_rmse:.4f})")
    return best_name, best_model


def save_model(model, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    joblib.dump(model, path)
    print(f"✅ Model saved → {path}")


def run_training(processed_path: str = "data/processed/cleaned_data.csv"):
    print("\n" + "="*55)
    print("  MODEL TRAINING PIPELINE")
    print("="*55)

    df   = load_processed(processed_path)
    X, y, feat_cols = prepare_matrices(df)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=0.15, random_state=RANDOM_STATE
    )

    print(f"\n  Train : {X_train.shape[0]:,} samples")
    print(f"  Val   : {X_val.shape[0]:,} samples")
    print(f"  Test  : {X_test.shape[0]:,} samples")
    print(f"  Features: {len(feat_cols)}\n")

    fitted = train_all_models(X_train, y_train)
    best_name, best_model = select_best(fitted, X_val, y_val)

    save_model(best_model, MODEL_PATH)

    # Also persist feature column order for inference
    joblib.dump(feat_cols, "models/feature_cols.pkl")

    return best_model, X_test, y_test, feat_cols


if __name__ == "__main__":
    run_training()
