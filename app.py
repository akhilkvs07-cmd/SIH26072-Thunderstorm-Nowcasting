"""StormWatch — SIH26072 thunderstorm nowcasting demonstration dashboard."""
from __future__ import annotations
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src.data import FEATURE_COLUMNS, load_dataset, validate_dataset
from src.modeling import train_model_zoo
from src.evaluation import metric_summary
from src.live_weather import fetch_current_conditions
from src.features import haversine_km
from src.real_data import fetch_forecast, fetch_imd_feed

st.set_page_config(page_title="StormWatch | SIH26072", page_icon="⛈️", layout="wide", initial_sidebar_state="expanded")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Mono:wght@400;500&family=Manrope:wght@400;500;600;700;800&display=swap');
:root{--bg:#07111f;--panel:#101f33;--line:#203650;--muted:#91a6bd;--cyan:#42d6e8;--orange:#ffad5c;--red:#ff637d}
.stApp{background:radial-gradient(ellipse at 70% -10%,#15304b 0%,#07111f 48%);color:#eaf2fb;font-family:'Manrope',sans-serif}
[data-testid="stHeader"]{background:transparent}
.block-container{padding-top:1.35rem;padding-bottom:3rem;max-width:1500px}
section[data-testid="stSidebar"]{background:#0b192b;border-right:1px solid #203650}
section[data-testid="stSidebar"] *{color:#dce8f5}
h1,h2,h3{letter-spacing:-.04em}
h1{font-weight:800!important}
[data-testid="stMetric"]{background:linear-gradient(145deg,#122740,#0d1b2d);border:1px solid #27435e;padding:17px 18px;border-radius:15px}
[data-testid="stMetricLabel"]{color:#9eb4ca!important;font-size:.78rem}
[data-testid="stMetricValue"]{font-weight:800;color:#f2f7ff}
[data-testid="stMetricDelta"]{font-family:'DM Mono',monospace}
div[data-testid="stTabs"] button{font-weight:700}
.stButton>button,.stDownloadButton>button{border-radius:9px;border:1px solid #2a536d;background:#12324a;color:#eaf8ff;font-weight:700}
.stButton>button:hover,.stDownloadButton>button:hover{border-color:#42d6e8;color:#42d6e8}
div[data-testid="stAlert"]{border-radius:12px}
[data-testid="stDataFrame"]{border:1px solid #203650;border-radius:12px}
.smallcaps{font:600 11px 'DM Mono',monospace;letter-spacing:.16em;color:#6f9bb8;text-transform:uppercase}
.hero{padding:20px 22px;border:1px solid #23435d;border-radius:18px;background:linear-gradient(110deg,rgba(20,53,78,.85),rgba(12,28,46,.72));margin:6px 0 18px}
.hero-title{font-size:34px;font-weight:800;letter-spacing:-1.5px;color:#f3f8ff}
.hero-sub{color:#9db4ca;font-size:13px;margin-top:5px}
.pill{display:inline-block;border:1px solid #7c6537;background:#392b18;color:#ffd18a;padding:5px 9px;border-radius:20px;font:600 10px 'DM Mono',monospace;letter-spacing:.08em}
.section-note{color:#91a6bd;font-size:12px;margin-top:-8px}

.hero{padding:clamp(16px,3vw,28px);overflow:hidden}.hero-top{display:flex;align-items:center;justify-content:space-between;gap:10px;flex-wrap:wrap}.hero-title{font-size:clamp(25px,4vw,42px);line-height:1.12;letter-spacing:-1px;overflow-wrap:anywhere}.hero-sub{line-height:1.6}.kpi-grid{display:grid;grid-template-columns:repeat(5,minmax(0,1fr));gap:10px;margin:12px 0 20px}.kpi{background:linear-gradient(145deg,#122740,#0d1b2d);border:1px solid #27435e;border-radius:13px;padding:14px;min-width:0}.kpi-label{color:#9eb4ca;font-size:11px}.kpi-value{font-size:clamp(20px,2.3vw,28px);font-weight:800;color:#f2f7ff;margin:8px 0 3px;overflow-wrap:anywhere}.kpi-note{font:500 10px 'DM Mono',monospace;color:#42d6a4;line-height:1.4}@media(max-width:900px){.kpi-grid{grid-template-columns:repeat(2,minmax(0,1fr))}.kpi:last-child{grid-column:span 2}}@media(max-width:600px){.block-container{padding:1rem .75rem 2rem}.hero{padding:16px;border-radius:14px}.hero-title{font-size:clamp(25px,8vw,34px)}.hero-sub{font-size:12px}.kpi{padding:12px}.kpi-value{font-size:24px}div[data-testid='stTabs'] button{font-size:11px;padding:6px 7px}}
</style>
""", unsafe_allow_html=True)

with st.sidebar:
    st.markdown('<div class="smallcaps">SIH 2026 · Problem 26072</div>', unsafe_allow_html=True)
    st.markdown("## ⛈️ StormWatch")
    st.caption("Thunderstorm & lightning nowcasting intelligence")
    st.divider()
    st.markdown("### Data centre")
    uploaded = st.file_uploader("Upload labelled observation CSV", type=["csv"])
    st.caption("Use the schema in README.md. The demo dataset is generated automatically.")
    st.divider()
    st.markdown("### Scenario controls")
    horizon = st.select_slider("Scenario horizon", options=[30,60,120], value=60, format_func=lambda x:f"{x} min")
    selected_model = st.selectbox("Prediction model", ["Logistic Regression", "Random Forest", "Hist Gradient Boosting"])
    threshold = st.slider("Alert threshold", .10, .90, .50, .05)
    city_filter = st.selectbox("Focus region", ["All regions","Bengaluru","Chennai","Hyderabad","Mumbai"])
    st.divider()
    st.markdown("### Display")
    show_demo = st.toggle("Show demonstration data notice", value=True)
    if st.button("↻ Fetch live weather context", use_container_width=True):
        with st.spinner("Fetching current model-based weather fields…"):
            try:
                st.session_state["live_weather_context"] = fetch_current_conditions()
                st.session_state["live_weather_error"] = None
            except Exception as exc:
                st.session_state["live_weather_error"] = str(exc)
    live_city = st.selectbox("Live forecast location", ["Bengaluru","Chennai","Hyderabad","Mumbai"], key="live_city")
    if st.button("Fetch 24-hour real weather forecast", use_container_width=True):
        coords = {"Bengaluru":(12.9716,77.5946),"Chennai":(13.0827,80.2707),"Hyderabad":(17.3850,78.4867),"Mumbai":(19.0760,72.8777)}
        try:
            st.session_state["forecast_data"] = fetch_forecast(*coords[live_city], label=live_city)
            st.session_state["forecast_error"] = None
        except Exception as exc:
            st.session_state["forecast_error"] = str(exc)
    if st.button("Fetch official IMD warnings / nowcast", use_container_width=True):
        try:
            st.session_state["imd_feeds"] = {
                "District nowcast": fetch_imd_feed("districtnowcast"),
                "District warnings": fetch_imd_feed("districtwarning"),
                "Station nowcast": fetch_imd_feed("stationnowcast"),
            }
            st.session_state["imd_error"] = None
        except Exception as exc:
            st.session_state["imd_error"] = str(exc)
    st.caption("Forecast models and official IMD products are kept separate from synthetic model evaluation.")
    st.caption("Prototype build · v3.0 · multi-model evaluation")

try:
    data, source_label = load_dataset(uploaded)
    data = validate_dataset(data)
except Exception as exc:
    st.error(f"Dataset could not be loaded: {exc}")
    st.stop()

if city_filter != "All regions":
    city_coords = {"Bengaluru":(12.97,77.59),"Chennai":(13.08,80.27),"Hyderabad":(17.38,78.49),"Mumbai":(19.07,72.88)}
    lat,lon = city_coords[city_filter]
    focused = data[(data.latitude.sub(lat).abs()<1.0)&(data.longitude.sub(lon).abs()<1.0)].copy()
    if focused.empty: focused = data.copy()
else:
    focused = data.copy()

try:
    train, test, y_true, model_results, baseline_metrics = train_model_zoo(data)
    selected = model_results[selected_model]
    model = selected['model']
    probabilities = selected['probabilities']
    predicted = (probabilities >= threshold).astype(int)
    metrics = metric_summary(y_true, predicted)
    metrics['ROC AUC'] = selected['metrics']['ROC AUC']
    metrics['Brier score'] = selected['metrics']['Brier score']
except Exception as exc:
    st.error(f'Model evaluation could not run: {exc}')
    st.stop()

latest = focused.sort_values('timestamp').tail(1).iloc[0]
latest_features = pd.DataFrame([{c:float(latest[c]) for c in FEATURE_COLUMNS}])
risk_probability = float(model.predict_proba(latest_features)[0,1])
risk_label = 'SEVERE' if risk_probability >= .80 else 'HIGH' if risk_probability >= .65 else 'WATCH' if risk_probability >= threshold else 'LOW'
risk_color = {'SEVERE':'#ff637d','HIGH':'#ff8c69','WATCH':'#ffbd69','LOW':'#42d6a4'}[risk_label]
last_time = pd.to_datetime(focused.timestamp.max(),utc=True)

st.markdown(f"""
<div class="hero">
 <div class="hero-top"><div class="smallcaps">ATMOSPHERIC INTELLIGENCE PLATFORM</div><span class="pill">● DEMO ENVIRONMENT</span></div>
 <div class="hero-title">StormWatch <span style="color:#42d6e8">/</span><br>Nowcasting Centre</div>
 <div class="hero-sub">AI-assisted thunderstorm risk monitoring<br>{source_label} · Sample timestamp: {last_time.strftime('%d %b %Y, %H:%M UTC')}</div>
</div>
""", unsafe_allow_html=True)
if show_demo or uploaded is None:
    st.warning("DEMONSTRATION ONLY — the bundled observations are synthetic. Risk scores, maps and alerts are illustrative, not live meteorological forecasts or public-safety warnings.")

# Compact responsive KPI strip
st.markdown(f"""
<div class="kpi-grid">
 <div class="kpi"><div class="kpi-label">CURRENT RISK INDEX</div><div class="kpi-value">{risk_probability:.0%}</div><div class="kpi-note">{risk_label} · illustrative</div></div>
 <div class="kpi"><div class="kpi-label">SCENARIO HORIZON</div><div class="kpi-value">{horizon} min</div><div class="kpi-note">User-selected setting</div></div>
 <div class="kpi"><div class="kpi-label">OBSERVATIONS</div><div class="kpi-value">{len(focused):,}</div><div class="kpi-note">{focused.timestamp.nunique():,} time points</div></div>
 <div class="kpi"><div class="kpi-label">LIGHTNING SAMPLE COUNT</div><div class="kpi-value">{int(focused.lightning_count.sum()):,}</div><div class="kpi-note">Synthetic sample only</div></div>
 <div class="kpi"><div class="kpi-label">STORM-LABEL SHARE</div><div class="kpi-value">{focused.storm_event.mean():.1%}</div><div class="kpi-note">Historical sample share</div></div>
</div>
""", unsafe_allow_html=True)

tabs = st.tabs(["🌩️ Command centre","🗺️ Risk map","📈 Atmospheric analytics","🤖 AI model lab","🚨 Alert desk","📁 Data & reports","🌐 Live weather context","🇮🇳 Official IMD feeds"])

with tabs[0]:
    st.markdown("### Regional situation overview")
    st.markdown('<div class="section-note">Latest observation by location · illustrative risk derived from the demonstration classifier</div>',unsafe_allow_html=True)
    def nearest_region(row):
        coords = {"Bengaluru": (12.97, 77.59), "Chennai": (13.08, 80.27),
                  "Hyderabad": (17.38, 78.49), "Mumbai": (19.07, 72.88)}
        return min(coords, key=lambda name: haversine_km(row.latitude, row.longitude, coords[name][0], coords[name][1]))

    loc = focused.copy()
    loc["region"] = loc.apply(nearest_region, axis=1)
    loc = loc.sort_values("timestamp").groupby("region", as_index=False).tail(1).copy()
    loc["probability"] = model.predict_proba(loc[FEATURE_COLUMNS])[:, 1]
    loc["risk"] = pd.cut(loc.probability, [-.001, .5, .65, .8, 1.001],
                         labels=["LOW", "WATCH", "HIGH", "SEVERE"])
    cols = st.columns(4)
    for col, (_, row) in zip(cols, loc.iterrows()):
        with col:
            st.markdown(f"""<div style="background:#102239;border:1px solid #27445e;border-radius:14px;padding:15px;min-height:130px">
            <div class="smallcaps">{row['risk']} RISK</div><div style="font-size:20px;font-weight:800;margin:8px 0">{row['region']}</div>
            <div style="font:700 26px 'DM Mono';color:#42d6e8">{row['probability']:.0%}</div>
            <div style="color:#91a6bd;font-size:11px">Humidity {row.humidity_pct:.0f}% · CAPE {row.cape_jkg:.0f} J/kg</div></div>""", unsafe_allow_html=True)
    st.markdown("### Atmospheric signal timeline")
    timeline = focused.sort_values("timestamp").tail(500)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=timeline.timestamp,y=timeline.cape_jkg,name="CAPE",line=dict(color="#42d6e8",width=2),yaxis="y"))
    fig.add_trace(go.Scatter(x=timeline.timestamp,y=timeline.humidity_pct,name="Humidity",line=dict(color="#a78bfa",width=1.5),yaxis="y2"))
    fig.update_layout(template="plotly_dark",height=330,margin=dict(l=10,r=10,t=25,b=10),paper_bgcolor="rgba(0,0,0,0)",plot_bgcolor="rgba(9,23,39,.6)",legend=dict(orientation="h",y=1.12),yaxis=dict(title="CAPE (J/kg)",gridcolor="#203650"),yaxis2=dict(title="Humidity (%)",overlaying="y",side="right",showgrid=False))
    st.plotly_chart(fig,use_container_width=True)

with tabs[1]:
    st.markdown("### India storm-risk geospatial view")
    st.caption("Station-level sample observations. These are not radar echoes or live lightning-network detections.")
    mapdata = loc.copy()
    mapdata["risk_probability"] = model.predict_proba(mapdata[FEATURE_COLUMNS])[:,1]
    mapdata["risk_band"] = pd.cut(mapdata.risk_probability,[-.001,.5,.65,.8,1.001],labels=["Low","Watch","High","Severe"])
    mf=px.scatter_map(mapdata,lat="latitude",lon="longitude",color="risk_band",size="risk_probability",size_max=36,hover_name="region",hover_data=["risk_probability","humidity_pct","cape_jkg","lightning_count","precipitation_mm"],zoom=3.4,center={"lat":17.5,"lon":78.5},height=570,color_discrete_map={"Low":"#42d6a4","Watch":"#ffbd69","High":"#ff8c69","Severe":"#ff637d"})
    mf.update_layout(map_style="carto-darkmatter",paper_bgcolor="rgba(0,0,0,0)",margin=dict(l=0,r=0,t=0,b=0),legend_title="Risk band")
    st.plotly_chart(mf,use_container_width=True)
    st.info("Map markers represent the latest sample for each region. No live radar, satellite or lightning feed is connected in this prototype.")

with tabs[2]:
    st.markdown("### Atmospheric analytics")
    c1,c2=st.columns(2)
    with c1:
        f=px.line(focused.sort_values("timestamp"),x="timestamp",y="temperature_c",color_discrete_sequence=["#ffad5c"],title="Near-surface temperature")
        f.update_layout(template="plotly_dark",height=330)
        st.plotly_chart(f,use_container_width=True)
    with c2:
        f=px.area(focused.sort_values("timestamp"),x="timestamp",y="precipitation_mm",color_discrete_sequence=["#42d6e8"],title="Precipitation signal")
        f.update_layout(template="plotly_dark",height=330)
        st.plotly_chart(f,use_container_width=True)
    c3,c4=st.columns(2)
    with c3:
        f=px.scatter(focused,x="cape_jkg",y="lightning_count",color="storm_event",size="precipitation_mm",title="Convective energy vs lightning",color_discrete_map={0:"#42d6e8",1:"#ff637d"})
        f.update_layout(template="plotly_dark",height=330)
        st.plotly_chart(f,use_container_width=True)
    with c4:
        corr=focused[FEATURE_COLUMNS].corr()
        f=px.imshow(corr,text_auto=".2f",color_continuous_scale="Tealrose",title="Feature correlation matrix")
        f.update_layout(template="plotly_dark",height=330)
        st.plotly_chart(f,use_container_width=True)

with tabs[3]:
    st.markdown("### Model performance & explainability")
    st.caption("Chronological holdout evaluation · Active model selected in the sidebar; compare all candidate models below.")
    b1,b2,b3,b4=st.columns(4)
    b1.metric("Accuracy",f"{metrics['Accuracy']:.1%}")
    b2.metric("Precision",f"{metrics['Precision']:.1%}")
    b3.metric("Recall",f"{metrics['Recall']:.1%}")
    b4.metric("F1 score",f"{metrics['F1 score']:.1%}")
    c1,c2=st.columns(2)
    with c1:
        cm=np.array([[metrics["True negatives"],metrics["False positives"]],[metrics["False negatives"],metrics["True positives"]]])
        cf=px.imshow(cm,text_auto=True,x=["Predicted clear","Predicted event"],y=["Observed clear","Observed event"],color_continuous_scale="Teal",title="Confusion matrix")
        cf.update_layout(template="plotly_dark",height=350)
        st.plotly_chart(cf,use_container_width=True)
    with c2:
        classifier = model.named_steps["classifier"]
        if hasattr(classifier, "feature_importances_"):
            importance = classifier.feature_importances_
            title = "Tree-model feature importance"
        elif hasattr(classifier, "coef_"):
            importance = np.abs(classifier.coef_[0])
            title = "Absolute logistic-regression coefficients"
        else:
            from sklearn.inspection import permutation_importance
            perm = permutation_importance(model, test[FEATURE_COLUMNS], y_true, n_repeats=5, random_state=42, scoring="f1")
            importance = perm.importances_mean
            title = "Permutation feature importance (F1 decrease)"
        imp=pd.DataFrame({"Feature":FEATURE_COLUMNS,"Importance":importance}).sort_values("Importance")
        cf=px.bar(imp,x="Importance",y="Feature",orientation="h",color="Importance",color_continuous_scale="Teal",title=title)
        cf.update_layout(template="plotly_dark",height=350)
        st.plotly_chart(cf,use_container_width=True)
    comparison = [{"Model": name, **item["metrics"]} for name,item in model_results.items()]
    comparison.append({"Model":"Threshold baseline", **baseline_metrics})
    compare=pd.DataFrame(comparison).sort_values("F1 score",ascending=False)
    st.markdown("#### Probability quality")
    from sklearn.calibration import calibration_curve
    q1,q2=st.columns(2)
    with q1:
        if len(np.unique(y_true)) == 2:
            from sklearn.metrics import roc_curve
            fpr,tpr,_=roc_curve(y_true, probabilities)
            roc_fig=go.Figure()
            roc_fig.add_trace(go.Scatter(x=fpr,y=tpr,mode="lines",name="Selected model",line=dict(color="#42d6e8",width=3)))
            roc_fig.add_trace(go.Scatter(x=[0,1],y=[0,1],mode="lines",name="Random baseline",line=dict(color="#91a6bd",dash="dash")))
            roc_fig.update_layout(template="plotly_dark",title="ROC curve",xaxis_title="False positive rate",yaxis_title="True positive rate",height=320,paper_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(roc_fig,use_container_width=True)
        else:
            st.info("ROC curve requires both classes in the held-out period.")
    with q2:
        if len(np.unique(y_true)) == 2:
            fraction_positive, mean_predicted = calibration_curve(y_true, probabilities, n_bins=8, strategy="quantile")
            cal_fig=go.Figure()
            cal_fig.add_trace(go.Scatter(x=mean_predicted,y=fraction_positive,mode="lines+markers",name="Model calibration",line=dict(color="#ffbd69",width=2)))
            cal_fig.add_trace(go.Scatter(x=[0,1],y=[0,1],mode="lines",name="Ideal calibration",line=dict(color="#91a6bd",dash="dash")))
            cal_fig.update_layout(template="plotly_dark",title="Calibration curve",xaxis_title="Mean predicted probability",yaxis_title="Observed event frequency",height=320,paper_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(cal_fig,use_container_width=True)
        else:
            st.info("Calibration curve requires both classes in the held-out period.")
    st.markdown("#### Model leaderboard")
    st.dataframe(compare,use_container_width=True,hide_index=True)
    st.download_button("Download model leaderboard CSV",compare.to_csv(index=False).encode("utf-8"),"stormwatch_model_leaderboard.csv","text/csv")
    st.caption("Synthetic holdout metrics are pipeline demonstrations, not evidence of meteorological forecast skill. Horizon selection is a scenario control; separate horizon-labelled training targets are required for validated 30/60/120-minute forecasts.")

with tabs[4]:
    st.markdown("### Alert desk")
    st.caption("Illustrative alert queue generated from sample station-level probabilities. Not sent to emergency services.")
    alert_rows=loc.copy()
    alert_rows["risk_probability"]=alert_rows["probability"]
    alert_rows["severity"]=pd.cut(alert_rows.risk_probability,[-.001,.5,.65,.8,1.001],labels=["LOW","WATCH","HIGH","SEVERE"])
    alert_rows=alert_rows.sort_values("risk_probability",ascending=False)
    for _,r in alert_rows.iterrows():
        colour={"LOW":"#42d6a4","WATCH":"#ffbd69","HIGH":"#ff8c69","SEVERE":"#ff637d"}[str(r.severity)]
        st.markdown(f"""<div style="border:1px solid #29435c;border-left:5px solid {colour};background:#102239;padding:13px 16px;border-radius:10px;margin:8px 0">
        <span class="smallcaps">{r.severity} · DEMO ALERT</span><div style="font-size:17px;font-weight:800">{r.region}</div>
        <div style="color:#a9bed2;font-size:12px">Illustrative event probability: {r.risk_probability:.1%} · Humidity {r.humidity_pct:.0f}% · Lightning count {int(r.lightning_count)}</div></div>""",unsafe_allow_html=True)
    st.warning("No push notifications, SMS, official warnings or real-time monitoring are active.")

with tabs[5]:
    st.markdown("### Observation explorer")
    st.caption(f"Showing latest 250 of {len(focused):,} records in the selected region.")
    st.dataframe(focused.sort_values("timestamp",ascending=False).head(250),use_container_width=True,hide_index=True)
    csv=focused.to_csv(index=False).encode("utf-8")
    st.download_button("⬇ Download observations CSV",csv,"stormwatch_observations.csv","text/csv")
    report=pd.DataFrame([{"Model": name, **item["metrics"]} for name,item in model_results.items()] + [{"Model":"Threshold baseline", **baseline_metrics}])
    st.download_button("⬇ Download model evaluation CSV",report.to_csv(index=False).encode("utf-8"),"stormwatch_model_metrics.csv","text/csv")
    st.download_button("⬇ Download alert summary CSV",alert_rows.to_csv(index=False).encode("utf-8"),"stormwatch_alert_summary.csv","text/csv")
    st.markdown("#### Data requirements for operational validation")
    st.markdown("- Timestamp-aligned radar reflectivity and satellite cloud products\n- Quality-controlled lightning network detections\n- Surface station observations and verified event labels\n- Documented forecast horizon, spatial resolution and missing-data policy")


with tabs[6]:
    st.markdown("### Live atmospheric context")
    st.caption("Current model-based weather fields for four demonstration cities. This feed is separate from the labelled training dataset.")
    live_error = st.session_state.get("live_weather_error")
    live_rows = st.session_state.get("live_weather_context")
    if live_error:
        st.error(f"Weather feed could not be reached: {live_error}")
    if not live_rows:
        st.info("Select **Fetch live weather context** in the sidebar to retrieve current conditions. Internet access is required.")
    else:
        live_df = pd.DataFrame(live_rows)
        st.success("Live context retrieved from Open-Meteo. Values are model-based estimates, not station observations.")
        k1,k2,k3,k4 = st.columns(4)
        k1.metric("Cities", len(live_df))
        k2.metric("Mean temperature", f"{live_df.temperature_c.mean():.1f} °C")
        k3.metric("Mean humidity", f"{live_df.humidity_pct.mean():.0f}%")
        k4.metric("Mean wind", f"{live_df.wind_speed_kmh.mean():.1f} km/h")
        live_map = px.scatter_map(
            live_df, lat="latitude", lon="longitude", color="humidity_pct",
            size="precipitation_mm", size_max=28, hover_name="region",
            hover_data=["temperature_c","humidity_pct","precipitation_mm","wind_speed_kmh","cape_jkg"],
            color_continuous_scale="Teal", zoom=3.4,
            center={"lat":17.5,"lon":78.5}, height=430
        )
        live_map.update_layout(map_style="carto-darkmatter", paper_bgcolor="rgba(0,0,0,0)", margin=dict(l=0,r=0,t=0,b=0))
        st.plotly_chart(live_map, use_container_width=True)
        display_cols = ["region","timestamp_utc","temperature_c","humidity_pct","precipitation_mm","pressure_hpa","wind_speed_kmh","wind_gusts_kmh","cape_jkg","source"]
        st.dataframe(live_df[display_cols], use_container_width=True, hide_index=True)
        st.download_button("Download live context CSV", live_df.to_csv(index=False).encode("utf-8"), "stormwatch_live_weather_context.csv", "text/csv")
        st.warning("Open-Meteo current weather/CAPE fields do not provide verified lightning detections or an observed thunderstorm-event label. These values are context only and are not passed into the trained classifier.")
    st.markdown("#### Data-source roadmap")
    st.markdown("- Open-Meteo model-based atmospheric context is available on demand.\n- Add quality-controlled IMD / radar / satellite feeds after confirming access, licensing and update frequency.\n- Add lightning-network detections and event timestamps before claiming lightning nowcasting skill.")


with tabs[7]:
    st.markdown("### India Meteorological Department · official products")
    st.caption("Direct IMD API integration for district/station nowcasts and district warnings. Availability depends on IMD API account permissions.")
    imd_error = st.session_state.get("imd_error")
    if imd_error:
        st.error(imd_error)
    imd_feeds = st.session_state.get("imd_feeds")
    if not imd_feeds:
        st.info("Use **Fetch official IMD warnings / nowcast** in the sidebar. If access is denied, register on the IMD API portal and add IMD_API_TOKEN to Render environment variables.")
    else:
        for feed_name, feed in imd_feeds.items():
            st.markdown("#### " + feed_name)
            payload = feed.get("data")
            if isinstance(payload, list):
                frame = pd.json_normalize(payload)
                st.dataframe(frame, use_container_width=True, hide_index=True)
                st.download_button("Download " + feed_name + " CSV", frame.to_csv(index=False).encode("utf-8"), feed_name.lower().replace(" ","_") + ".csv", "text/csv", key="download_" + feed_name)
            else:
                st.json(payload)
            st.caption("Source: India Meteorological Department · " + feed.get("source_url",""))
        st.success("Official IMD feed data retrieved. These are published nowcasts/warnings, not StormWatch ML predictions.")
    st.markdown("#### Access and operational requirements")
    st.markdown("- Create an account and request API access at https://api.imd.gov.in/public/index.php\n- Configure the issued credential as IMD_API_TOKEN in Render → Environment\n- Confirm the authentication scheme, permitted endpoints, rate limits and reuse terms with IMD before production use\n- Radar imagery and lightning data endpoints may require specific product access; no radar or strike layer is fabricated here.")

st.markdown("---")
st.markdown('<div style="text-align:center;color:#7189a2;font-size:11px">STORMWATCH · SIH 2026 · PROBLEM STATEMENT 26072<br>Academic prototype · Synthetic data demonstration · Not for operational warnings</div>',unsafe_allow_html=True)
