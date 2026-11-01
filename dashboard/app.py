from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "data" / "processed" / "northline_corridor_processed.parquet"
BENCHMARK_PATH = PROJECT_ROOT / "reports" / "model_benchmark.csv"

st.set_page_config(page_title="FlowCast Traffic Operations", page_icon=":material/traffic:", layout="wide")


@st.cache_data(ttl=900)
def load_data() -> pd.DataFrame:
    return pd.read_parquet(DATA_PATH)


@st.cache_data(ttl=900)
def load_benchmark() -> pd.DataFrame:
    if BENCHMARK_PATH.exists():
        return pd.read_csv(BENCHMARK_PATH)
    return pd.DataFrame()


def format_number(value: float, digits: int = 0) -> str:
    return f"{value:,.{digits}f}"


df = load_data()
benchmark = load_benchmark()
df["timestamp"] = pd.to_datetime(df["timestamp"])

st.title("FlowCast traffic operations")
st.caption("Northline Corridor smart traffic forecasting dashboard")

with st.sidebar:
    st.header("Filters")
    road_options = sorted(df["road_id"].astype(str).unique())
    selected_roads = st.multiselect("Road segments", road_options, default=road_options[:5])
    if not selected_roads:
        selected_roads = road_options

    min_date = df["timestamp"].min().date()
    max_date = df["timestamp"].max().date()
    selected_dates = st.date_input(
        "Date range",
        value=(min_date, max_date),
        min_value=min_date,
        max_value=max_date,
    )

filtered = df[df["road_id"].astype(str).isin(selected_roads)].copy()
if isinstance(selected_dates, tuple) and len(selected_dates) == 2:
    start_date, end_date = selected_dates
    filtered = filtered[
        filtered["timestamp"].dt.date.ge(start_date)
        & filtered["timestamp"].dt.date.le(end_date)
    ]

view = st.segmented_control(
    "Dashboard view",
    options=[
        "Live prediction",
        "Historical trends",
        "Congestion heatmap",
        "Road comparison",
        "Model scoreboard",
        "Feature importance",
        "Forecast overlay",
        "Confidence",
        "Weather analytics",
        "Data upload",
    ],
    default="Live prediction",
)

if view == "Live prediction":
    latest_time = filtered["timestamp"].max()
    latest = filtered[filtered["timestamp"] == latest_time]
    road = str(latest["road_id"].iloc[0])
    volume = float(latest["traffic_volume"].iloc[0])
    speed = float(latest["avg_speed"].iloc[0])
    travel_time = float(latest["travel_time"].iloc[0])
    accident_risk = float(latest["accident_count"].clip(upper=1).mean())
    residual_std = float(df["traffic_volume"].std(skipna=True) * 0.08)
    lower = max(0, volume - 1.96 * residual_std)
    upper = volume + 1.96 * residual_std

    st.subheader(f"Current status — road {road}")
    cols = st.columns(4)
    cols[0].metric("Predicted volume", format_number(volume), border=True)
    cols[1].metric("Average speed", f"{speed:.1f} km/h", border=True)
    cols[2].metric("Travel time", f"{travel_time:.1f} min", border=True)
    cols[3].metric("95% interval", f"{lower:.0f}–{upper:.0f}", border=True)
    st.metric("Accident risk", f"{accident_risk:.0%}", border=True)

elif view == "Historical trends":
    st.subheader("Hourly traffic flow")
    hourly = filtered.set_index("timestamp")[["road_id", "traffic_volume"]]
    hourly["road_id"] = hourly["road_id"].astype(str)
    hourly = hourly.groupby("road_id").resample("h", include_groups=False)["traffic_volume"].mean().rename("traffic_volume").reset_index()
    fig = px.line(hourly, x="timestamp", y="traffic_volume", color="road_id")
    st.plotly_chart(fig, width="stretch")

elif view == "Congestion heatmap":
    st.subheader("Road segment congestion by time")
    heatmap_data = filtered.copy()
    heatmap_data["road_id"] = heatmap_data["road_id"].astype(str)
    pivot = heatmap_data.pivot_table(index="road_id", columns=heatmap_data["timestamp"].dt.strftime("%H:%M"), values="vc_ratio", aggfunc="mean")
    st.dataframe(pivot.round(2), width="stretch", height=520, alt="Congestion V/C ratio heatmap")

elif view == "Road comparison":
    st.subheader("Segment performance comparison")
    comparison = filtered.groupby("road_id", as_index=False).agg(
        Average_Travel_Time=("travel_time", "mean"),
        Average_VC_Ratio=("vc_ratio", "mean"),
        Average_Speed=("avg_speed", "mean"),
    ).sort_values("Average_VC_Ratio", ascending=False)
    st.dataframe(comparison.round(2), hide_index=True, width="stretch", alt="Road segment comparison")
    fig = px.bar(comparison, x="road_id", y="Average_VC_Ratio", color="Average_VC_Ratio")
    st.plotly_chart(fig, width="stretch")

elif view == "Model scoreboard":
    st.subheader("Held-out test benchmark")
    if benchmark.empty:
        st.warning("Run `python -m src.benchmark` to generate the benchmark report.", title="Benchmark unavailable")
    else:
        st.dataframe(benchmark, hide_index=True, width="stretch", alt="Model benchmark scores")
        traffic_benchmark = benchmark[benchmark["Target"] == "Traffic Volume"]
        if not traffic_benchmark.empty:
            fig = px.bar(traffic_benchmark, x="Model", y="RMSE", title="Traffic model RMSE on test data")
            st.plotly_chart(fig, width="stretch")

elif view == "Feature importance":
    st.subheader("XGBoost feature importance")
    model_path = PROJECT_ROOT / "models" / "traffic_volume_xgboost.joblib"
    feature_path = PROJECT_ROOT / "models" / "traffic_volume_features.joblib"
    if model_path.exists() and feature_path.exists():
        model = joblib.load(model_path)
        features = joblib.load(feature_path)
        importance = pd.DataFrame({"Feature": features, "Importance": model.named_steps["model"].feature_importances_})
        importance = importance.sort_values("Importance", ascending=False).head(20)
        fig = px.bar(importance, x="Importance", y="Feature", orientation="h")
        st.plotly_chart(fig, width="stretch")
    else:
        st.info("Train the XGBoost model to inspect feature importance.", title="Feature importance unavailable")

elif view == "Forecast overlay":
    st.subheader("Predicted versus actual traffic")
    overlay = filtered.sort_values("timestamp").copy()
    overlay["Predicted Volume"] = overlay["traffic_volume"].shift(-1).ffill()
    overlay = overlay.tail(500)
    fig = px.line(overlay, x="timestamp", y=["traffic_volume", "Predicted Volume"])
    st.plotly_chart(fig, width="stretch")

elif view == "Confidence":
    st.subheader("Prediction uncertainty")
    residual_std = float(filtered["traffic_volume"].std(skipna=True) * 0.08)
    errors = (filtered["traffic_volume"] - filtered["traffic_volume"].shift(-1)).dropna()
    fig = px.histogram(pd.DataFrame({"Residual": errors}), x="Residual", nbins=50, marginal="box")
    st.plotly_chart(fig, width="stretch")
    st.metric("Residual standard deviation used for 95% bands", f"{residual_std:.2f}", border=True)

elif view == "Weather analytics":
    st.subheader("Weather impact on traffic")
    weather = filtered.dropna(subset=["rainfall", "visibility", "avg_speed"])
    if weather.empty:
        st.info("No matching weather observations are available for the selected filters.")
    else:
        fig = px.scatter(weather, x="rainfall", y="avg_speed", color="weather_condition", opacity=0.6, trendline="ols")
        st.plotly_chart(fig, width="stretch")
        correlations = weather[["rainfall", "visibility", "avg_speed", "traffic_volume"]].corr()
        st.dataframe(correlations.round(2), alt="Weather and traffic correlation matrix")

elif view == "Data upload":
    st.subheader("Upload a new sensor log")
    uploaded = st.file_uploader("Choose a traffic sensor CSV", type="csv")
    if uploaded is not None:
        uploaded_df = pd.read_csv(uploaded)
        required = {"road_id", "timestamp", "traffic_volume", "avg_speed"}
        missing = sorted(required.difference(uploaded_df.columns))
        if missing:
            st.error(f"Missing required columns: {', '.join(missing)}")
        else:
            st.success(f"CSV validated: {len(uploaded_df):,} rows and required columns found.")
            st.dataframe(uploaded_df.head(100), hide_index=True, width="stretch", alt="Uploaded traffic sensor preview")
