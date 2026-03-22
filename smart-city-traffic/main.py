"""
main.py
───────
Orchestrates the full Smart City Traffic Forecasting pipeline:
  1. Data loading & cleaning
  2. Feature engineering
  3. Model training
  4. Evaluation & visualisations
  5. Sample predictions
"""

import os
import sys

sys.path.insert(0, "src")

from data_preprocessing import load_data, clean_data, save_processed
from train_model        import run_training
from evaluate_model     import run_evaluation
from predict            import predict_single, predict_range

RAW_PATH  = "data/raw/traffic_data.csv"
PROC_PATH = "data/processed/cleaned_data.csv"


def step_banner(n: int, title: str) -> None:
    width = 55
    print("\n" + "═" * width)
    print(f"  STEP {n}: {title}")
    print("═" * width)


def main():
    print("\n" + "█" * 55)
    print("    SMART CITY TRAFFIC FORECASTING SYSTEM")
    print("█" * 55)

    # ── Step 1 – Data Preprocessing ─────────────────────────────────────────
    step_banner(1, "Data Preprocessing")
    if not os.path.exists(PROC_PATH):
        df_raw   = load_data(RAW_PATH)
        df_clean = clean_data(df_raw)
        save_processed(df_clean, PROC_PATH)
    else:
        print(f"   Processed data already exists at {PROC_PATH}. Skipping.")

    # ── Step 2 – Model Training ──────────────────────────────────────────────
    step_banner(2, "Model Training")
    model, X_test, y_test, feat_cols = run_training(PROC_PATH)

    # ── Step 3 – Model Evaluation ────────────────────────────────────────────
    step_banner(3, "Model Evaluation")
    metrics = run_evaluation(model, X_test, y_test, feat_cols)

    # ── Step 4 – Sample Predictions ──────────────────────────────────────────
    step_banner(4, "Sample Predictions")
    demo_cases = [
        ("2024-08-15 08:00", 1, "Independence Day morning – Junction 1"),
        ("2024-01-26 10:00", 2, "Republic Day mid-morning – Junction 2"),
        ("2024-03-22 17:30", 3, "Weekday evening rush – Junction 3"),
        ("2024-12-25 14:00", 4, "Christmas afternoon – Junction 4"),
    ]

    print(f"\n  {'DateTime':<22} {'Junction':>8} {'Prediction':>12}  Scenario")
    print(f"  {'-'*22} {'-'*8} {'-'*12}  {'-'*35}")
    for dt_str, jn, label in demo_cases:
        pred = predict_single(dt_str, jn, model, feat_cols)
        print(f"  {dt_str:<22} {'J'+str(jn):>8} {pred:>10.0f}  {label}")

    # ── Done ─────────────────────────────────────────────────────────────────
    print("\n" + "█" * 55)
    print("   PIPELINE COMPLETE")
    print(f"     R²   : {metrics['R2']:.4f}")
    print(f"     RMSE : {metrics['RMSE']:.4f}")
    print(f"     MAE  : {metrics['MAE']:.4f}")
    print("█" * 55)
    print("\n   Graphs → results/graphs/")
    print("   Model  → models/traffic_model.pkl")
    print("   Dashboard → streamlit run app/streamlit_app.py\n")


if __name__ == "__main__":
    main()
