"""Streamlit dashboard for the SIH26072 nowcasting prototype."""
from __future__ import annotations
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from src.data import FEATURE_COLUMNS, load_dataset, validate_dataset
from src.modeling import train_and_evaluate
from src.evaluation import metric_summary

st.set_page_config(page_title="StormWatch | SIH26072", page_icon="⛈️", layout="wide")
st.markdown("""
<style>
.main {background: #0b1220;}
[data-testid="stMetric"] {background: #111d31; border: 1px solid #263a57; padding: 14px; border-radius: 12px;}
.stAlert {border-radius: 10px;}
div.block-container {padding-top: 1.5rem;}
</style>
""", unsafe_allow_html=True)

st.title("⛈️ StormWatch")
st.caption("SIH26072 · AI/ML-Based Nowcasting of Thunderstorms and Lightning")
st.warning("DEMONSTRATION MODE: Included observations are synthetic. Predictions and scores are not operational weather forecasts or public safety warnings.")

with st.sidebar:
    st.header("Data source")
    uploaded = st.file_uploader("Upload labelled historical CSV", type=["csv"])
    st.caption("Required columns are documented in README.md. The CSV must include a binary storm_event label.")
    st.divider()
    st.header("Nowcast settings")
    horizon = st.select_slider("Forecast horizon", options=[30, 60, 120], value=60, format_func=lambda x: f"{x} minutes")
    threshold = st.slider("Alert probability threshold", 0.10, 0.90, 0.50, 0.05)
    st.caption("Threshold is a demo setting. Calibrate it on validation data before real use.")

try:
    data, source_label = load_dataset(uploaded)
except Exception as exc:
    st.error(str(exc))
    st.stop()

st.success(f"Loaded {len(data):,} rows · {source_label}")
if uploaded is None:
    st.info("Synthetic sample data is used so the complete workflow can be explored immediately. Replace it with verified observations for scientific evaluation.")

try:
    model, test, y_true, probabilities, ml_predictions, baseline_predictions = train_and_evaluate(data)
except Exception as exc:
    st.error(f"Unable to train model: {exc}")
    st.stop()

ml_predictions = (probabilities >= threshold).astype(int)
ml_metrics = metric_summary(y_true, ml_predictions)
base_metrics = metric_summary(y_true, baseline_predictions)

latest = data.sort_values("timestamp").tail(1).iloc[0]
latest_x = pd.DataFrame([{c: float(latest[c]) for c in FEATURE_COLUMNS}])
current_probability = float(model.predict_proba(latest_x)[0, 1])
risk = "HIGH" if current_probability >= 0.70 else "MODERATE" if current_probability >= threshold else "LOW"

m1, m2, m3, m4 = st.columns(4)
m1.metric("Latest estimated risk", risk, f"{current_probability:.0%} probability")
m2.metric("Forecast horizon", f"{horizon} min")
m3.metric("Observations", f"{len(data):,}")
m4.metric("Event prevalence", f"{data['storm_event'].mean():.1%}")

st.subheader("Nowcast overview")
left, right = st.columns([1.25, 1])
with left:
    view = data.sort_values("timestamp").tail(250)
    fig = px.scatter(view, x="timestamp", y="cape_jkg", color="storm_event",
                     size="precipitation_mm", hover_data=["latitude", "longitude", "humidity_pct", "lightning_count"],
                     color_discrete_map={0: "#60a5fa", 1: "#fb7185"},
                     labels={"storm_event": "Observed event", "cape_jkg": "CAPE (J/kg)"})
    fig.update_layout(template="plotly_dark", height=360, legend_title_text="Storm event")
    st.plotly_chart(fig, use_container_width=True)
with right:
    latest_points = data.sort_values("timestamp").groupby(data["timestamp"].dt.floor("3h")).tail(1).copy()
    latest_points["event_label"] = latest_points["storm_event"].map({0: "No event", 1: "Event"})
    map_fig = px.scatter_map(latest_points, lat="latitude", lon="longitude", color="event_label",
                             hover_name="timestamp", hover_data=["cape_jkg", "precipitation_mm", "lightning_count"],
                             zoom=3, height=360, color_discrete_map={"No event": "#60a5fa", "Event": "#fb7185"})
    map_fig.update_layout(map_style="carto-darkmatter", map_center={"lat": 16.5, "lon": 78.5},
                          margin={"r": 0, "t": 0, "l": 0, "b": 0}, template="plotly_dark")
    st.plotly_chart(map_fig, use_container_width=True)

st.subheader("Model evaluation · chronological holdout")
st.caption("The final 25% of timestamps is held out for testing. Scores depend on dataset quality and label definition.")
a, b = st.columns(2)
with a:
    st.markdown("**Machine-learning model · Logistic Regression**")
    st.dataframe(pd.DataFrame({"Metric": list(ml_metrics.keys()), "Value": list(ml_metrics.values())}), hide_index=True, use_container_width=True)
with b:
    st.markdown("**Transparent threshold baseline**")
    st.dataframe(pd.DataFrame({"Metric": list(base_metrics.keys()), "Value": list(base_metrics.values())}), hide_index=True, use_container_width=True)

cm = np.array([[ml_metrics["True negatives"], ml_metrics["False positives"]],
               [ml_metrics["False negatives"], ml_metrics["True positives"]]])
cm_fig = px.imshow(cm, text_auto=True, x=["Predicted no event", "Predicted event"],
                   y=["Observed no event", "Observed event"], color_continuous_scale="Blues",
                   aspect="auto", title="ML confusion matrix")
cm_fig.update_layout(template="plotly_dark", height=330)
st.plotly_chart(cm_fig, use_container_width=True)

st.subheader("Explore observations")
display = data.sort_values("timestamp", ascending=False).head(100)
st.dataframe(display, use_container_width=True, hide_index=True)

st.download_button("Download loaded observations (CSV)", data.to_csv(index=False).encode("utf-8"),
                   file_name="nowcasting_observations.csv", mime="text/csv")
report = pd.DataFrame([
    {"Model": "Logistic Regression", **ml_metrics},
    {"Model": "Threshold baseline", **base_metrics},
])
st.download_button("Download evaluation metrics (CSV)", report.to_csv(index=False).encode("utf-8"),
                   file_name="evaluation_metrics.csv", mime="text/csv")

with st.expander("How to interpret this prototype"):
    st.markdown("""
    - **Probability** is the classifier's estimated event likelihood from the supplied feature values; it is not a calibrated warning probability.
    - **Precision** measures how many predicted events were observed. **Recall** measures how many observed events were detected.
    - **False-alarm rate** and **missed-event rate** describe different failure modes and should both be reported.
    - The included synthetic data is generated for software demonstration only. Replace it with timestamp-aligned, quality-controlled radar, satellite, lightning and station observations before drawing meteorological conclusions.
    """)
st.caption("Academic prototype · SIH26072 · Not for operational forecasting or emergency decisions.")
