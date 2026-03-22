"""
evaluate_model.py
─────────────────
Evaluates the trained model and produces result visualisations.
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    r2_score,
)

GRAPH_DIR   = "results/graphs"
METRICS_FILE = "results/metrics.txt"
os.makedirs(GRAPH_DIR, exist_ok=True)
os.makedirs("results", exist_ok=True)


# ── Colour palette ────────────────────────────────────────────────────────────
PALETTE = ["#00B4D8", "#0077B6", "#90E0EF", "#48CAE4"]
sns.set_theme(style="darkgrid", palette=PALETTE)


def compute_metrics(y_true, y_pred, label: str = "") -> dict:
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae  = mean_absolute_error(y_true, y_pred)
    r2   = r2_score(y_true, y_pred)
    mape = np.mean(np.abs((y_true - y_pred) / (y_true + 1e-9))) * 100

    metrics = {"RMSE": rmse, "MAE": mae, "R2": r2, "MAPE (%)": mape}

    print(f"\n  📊 Metrics {label}")
    print(f"     RMSE : {rmse:.4f}")
    print(f"     MAE  : {mae:.4f}")
    print(f"     R²   : {r2:.4f}")
    print(f"     MAPE : {mape:.2f}%")
    return metrics


def save_metrics(metrics: dict, path: str = METRICS_FILE) -> None:
    with open(path, "w") as f:
        f.write("Smart City Traffic Forecasting – Model Metrics\n")
        f.write("=" * 50 + "\n")
        for k, v in metrics.items():
            f.write(f"{k:12s}: {v:.4f}\n")
    print(f"✅ Metrics saved → {path}")


# ── Plots ──────────────────────────────────────────────────────────────────────

def plot_actual_vs_predicted(y_true, y_pred, n: int = 500) -> None:
    """Line chart of actual vs predicted for the first n samples."""
    fig, ax = plt.subplots(figsize=(14, 4))
    ax.plot(y_true[:n], label="Actual",    color=PALETTE[0], lw=1.5)
    ax.plot(y_pred[:n], label="Predicted", color=PALETTE[1], lw=1.5, alpha=0.8)
    ax.set_title("Actual vs Predicted Vehicle Count", fontsize=14, fontweight="bold")
    ax.set_xlabel("Sample")
    ax.set_ylabel("Vehicles")
    ax.legend()
    fig.tight_layout()
    fig.savefig(f"{GRAPH_DIR}/actual_vs_predicted.png", dpi=150)
    plt.close(fig)
    print(f"  📈 Saved: actual_vs_predicted.png")


def plot_residuals(y_true, y_pred) -> None:
    residuals = y_true - y_pred
    fig, axes = plt.subplots(1, 2, figsize=(13, 4))

    axes[0].scatter(y_pred, residuals, alpha=0.3, s=8, color=PALETTE[0])
    axes[0].axhline(0, color="red", lw=1, ls="--")
    axes[0].set_title("Residuals vs Fitted")
    axes[0].set_xlabel("Predicted")
    axes[0].set_ylabel("Residual")

    axes[1].hist(residuals, bins=60, color=PALETTE[1], edgecolor="white")
    axes[1].set_title("Residual Distribution")
    axes[1].set_xlabel("Residual")
    axes[1].set_ylabel("Count")

    fig.suptitle("Residual Analysis", fontsize=14, fontweight="bold")
    fig.tight_layout()
    fig.savefig(f"{GRAPH_DIR}/residuals.png", dpi=150)
    plt.close(fig)
    print(f"  📈 Saved: residuals.png")


def plot_feature_importance(model, feat_cols: list) -> None:
    """Works for tree-based models that expose .feature_importances_."""
    # Unwrap Pipeline if needed
    m = model
    if hasattr(model, "named_steps"):
        m = model.named_steps.get("model", model)

    if not hasattr(m, "feature_importances_"):
        print("  ⚠️  Feature importance not available for this model type.")
        return

    importances = m.feature_importances_
    idx = np.argsort(importances)[::-1][:15]   # top 15

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(
        range(len(idx)),
        importances[idx],
        color=PALETTE[0],
        edgecolor="white",
    )
    ax.set_xticks(range(len(idx)))
    ax.set_xticklabels([feat_cols[i] for i in idx], rotation=45, ha="right", fontsize=9)
    ax.set_title("Top 15 Feature Importances", fontsize=14, fontweight="bold")
    ax.set_ylabel("Importance")
    fig.tight_layout()
    fig.savefig(f"{GRAPH_DIR}/feature_importance.png", dpi=150)
    plt.close(fig)
    print(f"  📈 Saved: feature_importance.png")


def run_evaluation(model, X_test, y_test, feat_cols):
    print("\n" + "="*55)
    print("  MODEL EVALUATION")
    print("="*55)

    y_pred   = model.predict(X_test)
    metrics  = compute_metrics(y_test, y_pred, label="(Test Set)")
    save_metrics(metrics)

    plot_actual_vs_predicted(y_test, y_pred)
    plot_residuals(y_test, y_pred)
    plot_feature_importance(model, feat_cols)

    print("\n✅ Evaluation complete. Graphs saved to results/graphs/")
    return metrics


if __name__ == "__main__":
    import sys
    sys.path.insert(0, "src")
    from train_model import run_training

    model, X_test, y_test, feat_cols = run_training()
    run_evaluation(model, X_test, y_test, feat_cols)
