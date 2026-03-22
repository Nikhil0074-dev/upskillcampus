"""
streamlit_app.py
────────────────
Smart City Traffic Forecasting Dashboard
Run with: streamlit run app/streamlit_app.py
"""

import os
import sys
import warnings
warnings.filterwarnings("ignore")

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import joblib
from datetime import datetime, timedelta

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from feature_engineering import extract_features, get_feature_columns

# ─────────────────────────────────────────────────────────────────────────────
# Page config
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Smart City Traffic Forecasting",
    page_icon="🚦",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────────────────
# Custom CSS
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
  [data-testid="stSidebar"] { background: #0d1b2a; }
  [data-testid="stSidebar"] * { color: #e0f0ff !important; }

  .kpi-box {
      background: linear-gradient(135deg, #0077B6, #00B4D8);
      border-radius: 12px;
      padding: 18px 22px;
      text-align: center;
      color: white;
      box-shadow: 0 4px 16px rgba(0,119,182,0.35);
  }
  .kpi-label { font-size: 0.82rem; opacity: 0.85; letter-spacing: 0.05em; text-transform: uppercase; }
  .kpi-value { font-size: 2.1rem; font-weight: 700; line-height: 1.1; }
  .kpi-delta { font-size: 0.78rem; opacity: 0.75; }

  .section-header {
      font-size: 1.25rem;
      font-weight: 700;
      color: #0077B6;
      border-left: 4px solid #00B4D8;
      padding-left: 10px;
      margin: 24px 0 12px 0;
  }
  div[data-testid="stHorizontalBlock"] > div { gap: 14px !important; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# Load model & data helpers
# ─────────────────────────────────────────────────────────────────────────────
MODEL_PATH    = os.path.join(os.path.dirname(__file__), "..", "models", "traffic_model.pkl")
FEAT_COL_PATH = os.path.join(os.path.dirname(__file__), "..", "models", "feature_cols.pkl")
DATA_PATH     = os.path.join(os.path.dirname(__file__), "..", "data", "processed", "cleaned_data.csv")

@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        return None, None
    return joblib.load(MODEL_PATH), joblib.load(FEAT_COL_PATH)


@st.cache_data
def load_traffic_data():
    if not os.path.exists(DATA_PATH):
        return None
    df = pd.read_csv(DATA_PATH, parse_dates=["DateTime"])
    return df


def predict_row(dt: datetime, junction: int, model, feat_cols) -> float:
    row = pd.DataFrame({"DateTime": [pd.Timestamp(dt)], "Junction": [junction], "Vehicles": [0]})
    row = extract_features(row)
    X = np.zeros((1, len(feat_cols)))
    for i, col in enumerate(feat_cols):
        if col in row.columns:
            X[0, i] = row[col].values[0]
    return max(0.0, model.predict(X)[0])


# ─────────────────────────────────────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🏙️ Smart City Traffic")
    st.markdown(" Forecasting of Smart City Traffic Patterns")
    st.markdown("---")

    page = st.radio(
        "Navigate",
        ["📊 Overview Dashboard", "🔮 Predict Traffic", "📅 Forecast Range", "📈 Model Insights"],
        index=0,
    )

    st.markdown("---")
    st.markdown("**Junctions**")
    st.markdown("- 🟢 J1 – City Centre")
    st.markdown("- 🔵 J2 – North Ring Road")
    st.markdown("- 🟡 J3 – Industrial Zone")
    st.markdown("- 🔴 J4 – Residential Area")

model, feat_cols = load_model()
df_raw = load_traffic_data()

# ─────────────────────────────────────────────────────────────────────────────
# Header
# ─────────────────────────────────────────────────────────────────────────────
st.markdown(
    "<h1 style='color:#0077B6;margin-bottom:4px'>🚦 Smart City Traffic Forecasting</h1>"
    "<p style='color:#64748b;margin-top:0'>ML-powered traffic analytics for government infrastructure planning</p>",
    unsafe_allow_html=True,
)
st.markdown("---")

# ─────────────────────────────────────────────────────────────────────────────
# ── PAGE: Overview Dashboard ─────────────────────────────────────────────────
# ─────────────────────────────────────────────────────────────────────────────
if page == "📊 Overview Dashboard":
    if df_raw is None:
        st.warning("⚠️ Dataset not found. Place `traffic_data.csv` in `data/raw/` and run `main.py` first.")
        st.info("Download dataset from: https://drive.google.com/file/d/1y61cDyuO9Zrp1fSchWcAmCxk0B6SMx7X/")
        st.stop()

    df = extract_features(df_raw.copy())

    # ── KPI Row ────────────────────────────────────────────────────────────
    total_v   = int(df["Vehicles"].sum())
    avg_v     = df["Vehicles"].mean()
    peak_hour = int(df.groupby("Hour")["Vehicles"].mean().idxmax())
    busiest_j = int(df.groupby("Junction")["Vehicles"].mean().idxmax())

    c1, c2, c3, c4 = st.columns(4)
    for col, label, val, delta in [
        (c1, "Total Records",    f"{len(df):,}",          "Full dataset"),
        (c2, "Total Vehicles",   f"{total_v:,}",          "All junctions combined"),
        (c3, "Peak Hour",        f"{peak_hour:02d}:00",    "Busiest hour of day"),
        (c4, "Busiest Junction", f"Junction {busiest_j}", "Highest avg volume"),
    ]:
        col.markdown(
            f'<div class="kpi-box"><div class="kpi-label">{label}</div>'
            f'<div class="kpi-value">{val}</div>'
            f'<div class="kpi-delta">{delta}</div></div>',
            unsafe_allow_html=True,
        )

    st.markdown('<div class="section-header">Hourly Traffic Pattern by Junction</div>', unsafe_allow_html=True)
    hourly = df.groupby(["Hour", "Junction"])["Vehicles"].mean().reset_index()
    fig = px.line(
        hourly, x="Hour", y="Vehicles", color="Junction",
        color_discrete_sequence=["#0077B6", "#00B4D8", "#48CAE4", "#90E0EF"],
        markers=True, labels={"Junction": "Junction"},
    )
    fig.add_vrect(x0=7, x1=9.5, fillcolor="red", opacity=0.08, annotation_text="Morning Rush")
    fig.add_vrect(x0=17, x1=19.5, fillcolor="orange", opacity=0.08, annotation_text="Evening Rush")
    fig.update_layout(height=380, plot_bgcolor="#f8fafc", paper_bgcolor="#ffffff")
    st.plotly_chart(fig, use_container_width=True)

    col_l, col_r = st.columns(2)

    with col_l:
        st.markdown('<div class="section-header">Weekday vs Weekend Traffic</div>', unsafe_allow_html=True)
        df["DayType"] = df["IsWeekend"].map({0: "Weekday", 1: "Weekend"})
        wd = df.groupby(["Junction", "DayType"])["Vehicles"].mean().reset_index()
        fig2 = px.bar(
            wd, x="Junction", y="Vehicles", color="DayType", barmode="group",
            color_discrete_map={"Weekday": "#0077B6", "Weekend": "#90E0EF"},
        )
        fig2.update_layout(height=320, plot_bgcolor="#f8fafc", paper_bgcolor="#ffffff")
        st.plotly_chart(fig2, use_container_width=True)

    with col_r:
        st.markdown('<div class="section-header">Monthly Traffic Trend</div>', unsafe_allow_html=True)
        monthly = df.groupby("Month")["Vehicles"].mean().reset_index()
        months  = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]
        monthly["MonthName"] = monthly["Month"].apply(lambda x: months[x-1])
        fig3 = px.area(
            monthly, x="MonthName", y="Vehicles",
            color_discrete_sequence=["#00B4D8"],
        )
        fig3.update_layout(height=320, plot_bgcolor="#f8fafc", paper_bgcolor="#ffffff")
        st.plotly_chart(fig3, use_container_width=True)

    st.markdown('<div class="section-header">Holiday vs Normal Day Impact</div>', unsafe_allow_html=True)
    df["Category"] = "Normal Weekday"
    df.loc[df["IsWeekend"] == 1, "Category"] = "Weekend"
    df.loc[df["IsHoliday"] == 1, "Category"] = "Public Holiday"
    cat_avg = df.groupby("Category")["Vehicles"].mean().reset_index()
    fig4 = px.bar(
        cat_avg, x="Category", y="Vehicles",
        color="Category",
        color_discrete_map={
            "Normal Weekday": "#0077B6",
            "Weekend": "#48CAE4",
            "Public Holiday": "#90E0EF",
        },
        text_auto=".1f",
    )
    fig4.update_layout(height=320, showlegend=False, plot_bgcolor="#f8fafc", paper_bgcolor="#ffffff")
    st.plotly_chart(fig4, use_container_width=True)


# ─────────────────────────────────────────────────────────────────────────────
# ── PAGE: Predict Traffic ────────────────────────────────────────────────────
# ─────────────────────────────────────────────────────────────────────────────
elif page == "🔮 Predict Traffic":
    st.markdown("### 🔮 Single-Point Traffic Prediction")
    st.caption("Enter a date, time, and junction to get a predicted vehicle count.")

    if model is None:
        st.error("No trained model found. Please run `python main.py` first to train the model.")
        st.stop()

    col1, col2, col3 = st.columns(3)
    with col1:
        pred_date = st.date_input("📅 Date", value=datetime.today())
    with col2:
        pred_hour = st.slider("⏰ Hour (0–23)", 0, 23, 8)
    with col3:
        pred_jn = st.selectbox("🚦 Junction", [1, 2, 3, 4], format_func=lambda x: f"Junction {x}")

    dt_input = datetime(pred_date.year, pred_date.month, pred_date.day, pred_hour, 0)

    if st.button("🚀 Predict", type="primary"):
        pred = predict_row(dt_input, pred_jn, model, feat_cols)

        # Severity label
        if pred < 20:
            level, colour = "🟢 Low", "#22c55e"
        elif pred < 50:
            level, colour = "🟡 Moderate", "#eab308"
        elif pred < 80:
            level, colour = "🟠 High", "#f97316"
        else:
            level, colour = "🔴 Congested", "#ef4444"

        st.markdown("---")
        kc1, kc2, kc3 = st.columns(3)
        kc1.metric("Predicted Vehicles", f"{pred:.0f}")
        kc2.metric("Congestion Level", level)
        kc3.metric("Junction", f"J{pred_jn}")

        # Gauge chart
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=pred,
            domain={"x": [0, 1], "y": [0, 1]},
            title={"text": f"Junction {pred_jn} | {dt_input.strftime('%d %b %Y %H:%M')}"},
            gauge={
                "axis": {"range": [0, 120]},
                "bar":  {"color": colour},
                "steps": [
                    {"range": [0, 20],  "color": "#dcfce7"},
                    {"range": [20, 50], "color": "#fef9c3"},
                    {"range": [50, 80], "color": "#ffedd5"},
                    {"range": [80, 120],"color": "#fee2e2"},
                ],
                "threshold": {
                    "line": {"color": "black", "width": 4},
                    "thickness": 0.75,
                    "value": pred,
                },
            },
        ))
        fig.update_layout(height=350, paper_bgcolor="#ffffff")
        st.plotly_chart(fig, use_container_width=True)


# ─────────────────────────────────────────────────────────────────────────────
# ── PAGE: Forecast Range ─────────────────────────────────────────────────────
# ─────────────────────────────────────────────────────────────────────────────
elif page == "📅 Forecast Range":
    st.markdown("### 📅 Multi-Hour / Multi-Day Forecast")

    if model is None:
        st.error("No trained model found. Please run `python main.py` first.")
        st.stop()

    c1, c2, c3 = st.columns(3)
    with c1:
        start_dt = st.date_input("Start Date", value=datetime.today())
    with c2:
        end_dt   = st.date_input("End Date",   value=datetime.today() + timedelta(days=1))
    with c3:
        sel_junctions = st.multiselect("Junctions", [1, 2, 3, 4], default=[1, 2])

    if st.button("📊 Generate Forecast", type="primary"):
        if not sel_junctions:
            st.warning("Please select at least one junction.")
            st.stop()
        if end_dt <= start_dt:
            st.warning("End date must be after start date.")
            st.stop()

        dates = pd.date_range(
            start=datetime(start_dt.year, start_dt.month, start_dt.day),
            end  =datetime(end_dt.year, end_dt.month, end_dt.day, 23),
            freq ="H",
        )

        all_preds = []
        for jn in sel_junctions:
            rows = pd.DataFrame({"DateTime": dates, "Junction": jn, "Vehicles": 0})
            rows = extract_features(rows)
            X = np.zeros((len(rows), len(feat_cols)))
            for i, col in enumerate(feat_cols):
                if col in rows.columns:
                    X[:, i] = rows[col].values
            preds = np.maximum(0, model.predict(X))
            for dt_val, pred_val in zip(dates, preds):
                all_preds.append({"DateTime": dt_val, "Junction": jn, "Vehicles": round(pred_val)})

        df_pred = pd.DataFrame(all_preds)
        df_pred["Junction"] = "J" + df_pred["Junction"].astype(str)

        fig = px.line(
            df_pred, x="DateTime", y="Vehicles", color="Junction",
            color_discrete_sequence=["#0077B6", "#00B4D8", "#48CAE4", "#90E0EF"],
            title="Hourly Traffic Forecast",
        )
        fig.update_layout(height=420, plot_bgcolor="#f8fafc", paper_bgcolor="#ffffff")
        st.plotly_chart(fig, use_container_width=True)

        # Summary table
        st.markdown("**Forecast Summary**")
        summary = df_pred.groupby("Junction")["Vehicles"].agg(
            Min="min", Max="max", Mean="mean", Total="sum"
        ).round(1)
        st.dataframe(summary, use_container_width=True)

        csv = df_pred.to_csv(index=False).encode("utf-8")
        st.download_button("⬇️ Download Forecast CSV", csv, "traffic_forecast.csv", "text/csv")


# ─────────────────────────────────────────────────────────────────────────────
# ── PAGE: Model Insights ─────────────────────────────────────────────────────
# ─────────────────────────────────────────────────────────────────────────────
elif page == "📈 Model Insights":
    st.markdown("### 📈 Model Performance & Insights")

    metrics_path = os.path.join(os.path.dirname(__file__), "..", "results", "metrics.txt")
    if os.path.exists(metrics_path):
        with open(metrics_path) as f:
            content = f.read()
        lines = [l for l in content.strip().split("\n") if ":" in l]
        mc1, mc2, mc3, mc4 = st.columns(4)
        cols = [mc1, mc2, mc3, mc4]
        for i, line in enumerate(lines[:4]):
            label, val = line.split(":", 1)
            cols[i].metric(label.strip(), val.strip())
    else:
        st.info("Model metrics not yet generated. Run `python main.py` first.")

    # Show saved graphs if they exist
    graph_dir = os.path.join(os.path.dirname(__file__), "..", "results", "graphs")
    for fname, title in [
        ("actual_vs_predicted.png", "Actual vs Predicted"),
        ("feature_importance.png",  "Feature Importance"),
        ("residuals.png",           "Residual Analysis"),
    ]:
        fpath = os.path.join(graph_dir, fname)
        if os.path.exists(fpath):
            st.markdown(f'<div class="section-header">{title}</div>', unsafe_allow_html=True)
            st.image(fpath, use_column_width=True)

    st.markdown('<div class="section-header">About the Model</div>', unsafe_allow_html=True)
    st.markdown("""
    | Property | Value |
    |---|---|
    | Algorithm | Random Forest Regressor |
    | Estimators | 200 trees |
    | Max Depth | 12 |
    | Feature Count | 21 engineered features |
    | Train/Test Split | 80% / 20% |
    | Target Variable | Vehicle count per hour |

    **Key engineered features:**
    - Cyclic encoding of Hour, Month, DayOfWeek (sin/cos)
    - IsWeekend, IsHoliday, IsRushHour, IsPeakSeason flags
    - One-hot encoded Junction (1–4)
    """)
