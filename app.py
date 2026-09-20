import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import folium
from streamlit_folium import st_folium

import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings("ignore")


def health_risk_score(aqi):
    """Converts AQI to a 1-10 Health Risk Score using CPCB's official AQI categories."""
    if aqi <= 50:
        return 1, "Good", "🟢"
    elif aqi <= 100:
        return 3, "Satisfactory", "🟢"
    elif aqi <= 200:
        return 5, "Moderate", "🟡"
    elif aqi <= 300:
        return 7, "Poor", "🟠"
    elif aqi <= 400:
        return 9, "Very Poor", "🔴"
    else:
        return 10, "Severe", "🔴"


# ── Page Config ──
st.set_page_config(
    page_title="VayuDrishti — Air Vision",
    page_icon="🛰️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Custom CSS ──
st.markdown("""
<style>
    .main { background-color: #0A0E1A; }
    .stApp { background-color: #0A0E1A; }
    h1, h2, h3 { color: #00C6FF; }
    .metric-card {
        background: #162040;
        border: 1px solid #1E2D45;
        border-radius: 10px;
        padding: 15px;
        text-align: center;
    }
    .stMetric {
        background: #162040;
        border-radius: 10px;
        padding: 10px;
    }
</style>
""", unsafe_allow_html=True)

# ── Load Data (15-city real ground data: AQI + pollutants) ──
@st.cache_data
def load_combined():
    d = pd.read_csv(
        "https://raw.githubusercontent.com/MANIMARAN-V18/"
        "VayuDrishti-ISRO-BAH2026/main/data/daily_combined.csv",
        sep=None, engine="python"
    )
    d["Date"] = pd.to_datetime(d["Date"], errors="coerce")
    d = d.dropna(subset=["Date", "AQI_avg"])
    d = d.rename(columns={"City": "city", "AQI_avg": "AQI"})
    d["month"] = d["Date"].dt.strftime("%b-%Y")
    return d

df = load_combined()

# ── Load original 10-city satellite-fusion dataset (for HCHO/satellite-only pages) ──
@st.cache_data
def load_satellite_data():
    return pd.read_csv(
        "https://raw.githubusercontent.com/MANIMARAN-V18/"
        "VayuDrishti-ISRO-BAH2026/main/data/master_dataset.csv"
    )

sat_df = load_satellite_data()

# ── Sidebar ──
st.sidebar.image(
    "https://upload.wikimedia.org/wikipedia/commons/"
    "thumb/b/bd/Indian_Space_Research_Organisation_Logo.svg/"
    "1200px-Indian_Space_Research_Organisation_Logo.svg.png",
    width=100
)
st.sidebar.title("VayuDrishti")
st.sidebar.markdown("*Air Vision for Every Indian*")
st.sidebar.markdown("---")

page = st.sidebar.selectbox(
    "Navigate",
    [
        "🏠 Home",
        "🗺️ AQI Map",
        "🔥 HCHO Hotspots",
        "📊 Model Performance",
        "🏙️ City Analysis",
        "🔮 AQI Forecast",
        "🧮 Live AQI Predictor"
    ]
)

st.sidebar.markdown("---")
st.sidebar.markdown("**Project Info**")
st.sidebar.markdown("🛰️ ISRO BAH 2026")
st.sidebar.markdown("📡 Challenge 03")
st.sidebar.markdown("👥 Team ASTROTECH")
st.sidebar.markdown("🎓 Puducherry, India")

# ════════════════════════════════
# PAGE 1 — HOME
# ════════════════════════════════
if page == "🏠 Home":
    st.title("🛰️ VayuDrishti — वायु दृष्टि")
    st.markdown(
        "### *Air Vision for Every Indian*"
    )
    st.caption(
        "Ground-truth models (below, Model Performance, Forecast) cover "
        "**15 real CPCB-monitored cities**. Satellite fusion (HCHO hotspots) "
        "currently covers the original **10-city** hackathon scope."
    )
    st.markdown("---")

    # Key metrics
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            label="🏙️ Cities Covered",
            value="15",
            delta="Real CPCB ground data"
        )
    with col2:
        st.metric(
            label="📊 Real Daily Records",
            value=f"{len(df):,}",
            delta="AQI + 5 pollutants"
        )
    with col3:
        st.metric(
            label="🤖 XGBoost R²",
            value="0.924",
            delta="Same-day AQI, real data"
        )
    with col4:
        st.metric(
            label="🔮 LSTM Forecast R²",
            value="0.832",
            delta="Next-day, 15 cities"
        )

    st.markdown("---")

    # Project description
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### 🎯 What is VayuDrishti?")
        st.markdown("""
        VayuDrishti is an Air Quality prediction and forecasting system that:

        - 🛰️ Uses **ISRO INSAT-3D** + **Sentinel-5P** satellite data (10-city hotspot detection)
        - 📡 Uses **real CPCB ground station data** across 15 Indian cities
        - 🧠 Applies **XGBoost** for same-day AQI prediction from pollutants
        - 🔮 Applies an **LSTM** for next-day AQI forecasting
        - 🔥 Detects **HCHO hotspots** from biomass burning (10-city satellite scope)
        """)

    with col2:
        st.markdown("### 🚨 The Problem We Solve")
        st.markdown("""
        India faces a critical air quality gap:

        - 👥 **1.4 Billion** people in India
        - 📡 Only **~300** CPCB air sensors exist
        - ❌ Most of India has no local, forecastable AQI data
        - 🏭 HCHO from factories and biomass burning is a serious hazard

        **VayuDrishti combines real ground data with satellite fusion to close this gap.**
        """)

    st.markdown("---")
    st.markdown("### 📊 Data Overview — 15 Cities (Real CPCB Data)")

    col1, col2, col3 = st.columns(3)
    with col1:
        st.dataframe(
            df.groupby("city")["AQI"]
            .mean().round(1)
            .sort_values(ascending=False)
            .reset_index()
            .rename(columns={"AQI": "Avg AQI"}),
            use_container_width=True
        )
    with col2:
        fig = px.bar(
            df.groupby("city")["AQI"]
            .mean().sort_values(ascending=False)
            .reset_index(),
            x="city", y="AQI",
            color="AQI",
            color_continuous_scale="RdYlGn_r",
            title="Average AQI by City (real data)"
        )
        fig.update_layout(
            template="plotly_dark",
            height=300
        )
        st.plotly_chart(fig, use_container_width=True)
    with col3:
        top_city = df.groupby("city")["AQI"].mean().idxmax()
        top_val = df.groupby("city")["AQI"].mean().max()
        low_city = df.groupby("city")["AQI"].mean().idxmin()
        low_val = df.groupby("city")["AQI"].mean().min()
        st.markdown("**Key Findings:**")
        st.error(f"🔴 {top_city}: Highest avg AQI ({top_val:.0f})")
        st.success(f"🟢 {low_city}: Lowest avg AQI ({low_val:.0f})")
        st.info("📅 Jan-Jun 2024 real CPCB ground data")

# ════════════════════════════════
# PAGE 2 — AQI MAP
# ════════════════════════════════
elif page == "🗺️ AQI Map":
    st.title("🗺️ India AQI Map")
    st.markdown(
        "Interactive map showing real daily AQI and pollutant levels "
        "across 15 Indian cities (CPCB ground data, Jan-Jun 2024)"
    )
    st.markdown("---")

    city_coords = {
        "Delhi"             : [28.6139, 77.2090],
        "Mumbai"            : [19.0760, 72.8777],
        "Chennai"           : [13.0827, 80.2707],
        "Kolkata"           : [22.5726, 88.3639],
        "Bengaluru"         : [12.9716, 77.5946],
        "Hyderabad"         : [17.3850, 78.4867],
        "Puducherry"        : [11.9416, 79.8083],
        "Lucknow"           : [26.8467, 80.9462],
        "Patna"             : [25.5941, 85.1376],
        "Ahmedabad"         : [23.0225, 72.5714],
        "Guwahati"          : [26.1445, 91.7362],
        "Thiruvananthapuram": [8.5241, 76.9366],
        "Dehradun"          : [30.3165, 78.0322],
        "Jodhpur"           : [26.2389, 73.0243],
        "Raipur"            : [21.2514, 81.6296],
    }

    month_order = sorted(
        df["month"].unique(),
        key=lambda m: pd.to_datetime(m, format="%b-%Y")
    )
    selected_month = st.selectbox("Select Month", month_order)

    df_month = (
        df[df["month"] == selected_month]
        .groupby("city")[["AQI", "PM2.5", "PM10", "NO2", "SO2"]]
        .mean()
        .reset_index()
    )

    m = folium.Map(
        location=[20.5937, 78.9629],
        zoom_start=5,
        tiles="https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png",
        attr="© OpenStreetMap contributors © CARTO"
    )

    for _, row in df_month.iterrows():
        city = row["city"]
        aqi = row["AQI"]

        if city not in city_coords:
            continue

        lat, lon = city_coords[city]

        if aqi > 200:
            color = "red"
        elif aqi > 150:
            color = "orange"
        elif aqi > 100:
            color = "yellow"
        else:
            color = "green"

        folium.CircleMarker(
            location=[lat, lon],
            radius=max(aqi / 15, 5),
            color=color,
            fill=True,
            fill_color=color,
            fill_opacity=0.7,
            popup=folium.Popup(
                f"<b>{city}</b><br>"
                f"AQI: {aqi:.0f}<br>"
                f"PM2.5: {row['PM2.5']:.1f}<br>"
                f"PM10: {row['PM10']:.1f}<br>"
                f"NO2: {row['NO2']:.1f}<br>"
                f"SO2: {row['SO2']:.1f}<br>"
                f"Month: {selected_month}",
                max_width=220
            ),
            tooltip=f"{city}: AQI {aqi:.0f}"
        ).add_to(m)

    st_folium(m, width=900, height=500)

    st.markdown("### 📊 AQI & Pollutant Values")
    st.dataframe(
        df_month.sort_values("AQI", ascending=False).round(2),
        use_container_width=True
    )

# ════════════════════════════════
# PAGE 3 — HCHO HOTSPOTS (satellite-only, original 10-city scope)
# ════════════════════════════════
elif page == "🔥 HCHO Hotspots":
    st.title("🔥 HCHO Hotspot Detection")
    st.markdown(
        "Formaldehyde hotspots from biomass burning and industries — "
        "**satellite data (Sentinel-5P), covering the original 10-city scope**"
    )
    st.caption(
        "HCHO detection relies on satellite fusion, which hasn't yet been "
        "extended to the 5 newly added cities (Guwahati, Thiruvananthapuram, "
        "Dehradun, Jodhpur, Raipur)."
    )
    st.markdown("---")

    city_hcho = sat_df.groupby("city")[
        "HCHO_sat"
    ].mean().sort_values(
        ascending=False
    ).reset_index()
    city_hcho.columns = ["city", "HCHO_mean"]

    col1, col2 = st.columns(2)

    with col1:
        colors = [
            "#FF6B6B" if v > 300
            else "#FFB347" if v > 220
            else "#00E5A0"
            for v in city_hcho["HCHO_mean"]
        ]
        fig = go.Figure(go.Bar(
            x=city_hcho["HCHO_mean"],
            y=city_hcho["city"],
            orientation="h",
            marker_color=colors,
            text=city_hcho["HCHO_mean"].round(1),
            textposition="outside"
        ))
        fig.add_vline(
            x=300, line_dash="dash", line_color="red",
            annotation_text="Danger (300)"
        )
        fig.add_vline(
            x=220, line_dash="dash", line_color="orange",
            annotation_text="Moderate (220)"
        )
        fig.update_layout(
            title="HCHO Hotspot Ranking (10-city satellite scope)",
            template="plotly_dark",
            height=400
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.markdown("### 🚨 Hotspot Alerts")
        for _, row in city_hcho.iterrows():
            hcho = row["HCHO_mean"]
            city = row["city"]
            if hcho > 300:
                st.error(f"🔴 **{city}** — DANGER: HCHO = {hcho:.1f}")
            elif hcho > 220:
                st.warning(f"🟡 **{city}** — MODERATE: HCHO = {hcho:.1f}")
            else:
                st.success(f"🟢 **{city}** — SAFE: HCHO = {hcho:.1f}")

    st.markdown("---")
    st.markdown("### 📈 Monthly HCHO Trend")

    month_order_sat = [
        "Jan-2024", "Feb-2024", "Mar-2024",
        "Apr-2024", "May-2024", "Jun-2024"
    ]
    top3 = city_hcho.head(3)["city"].tolist()
    df_top3 = sat_df[sat_df["city"].isin(top3)]

    fig2 = px.line(
        df_top3,
        x="month", y="HCHO_sat",
        color="city",
        markers=True,
        title="HCHO Trend — Top 3 Hotspot Cities",
        category_orders={"month": month_order_sat}
    )
    fig2.add_hline(
        y=300, line_dash="dash", line_color="red",
        annotation_text="Danger threshold"
    )
    fig2.update_layout(template="plotly_dark", height=400)
    st.plotly_chart(fig2, use_container_width=True)

# ════════════════════════════════
# PAGE 4 — MODEL PERFORMANCE
# ════════════════════════════════
elif page == "📊 Model Performance":
    st.title("📊 Model Performance")
    st.markdown(
        "VayuDrishti model accuracy — trained and validated on **real CPCB "
        "ground data across 15 cities**"
    )
    st.markdown("---")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### 🤖 XGBoost — Same-Day AQI from Pollutants")
        st.metric("RMSE", "20.99")
        st.metric("R² Score", "0.9237", delta="+92.4% accuracy")
        st.success(
            "✅ Trained on 2,565 real daily records across 15 cities "
            "(PM2.5, PM10, NO2, SO2, CO → AQI)"
        )

        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=92.37,
            title={"text": "XGBoost Accuracy %"},
            gauge={
                "axis": {"range": [0, 100]},
                "bar": {"color": "#00C6FF"},
                "steps": [
                    {"range": [0, 50], "color": "#FF6B6B"},
                    {"range": [50, 80], "color": "#FFB347"},
                    {"range": [80, 100], "color": "#00E5A0"},
                ],
                "threshold": {
                    "line": {"color": "white", "width": 4},
                    "thickness": 0.75,
                    "value": 92.37
                }
            }
        ))
        fig.update_layout(template="plotly_dark", height=300)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.markdown("### 🔮 LSTM — Next-Day AQI Forecast")
        st.metric("RMSE", "23.51")
        st.metric("R² Score", "0.8319", delta="+83.2% accuracy")
        st.info(
            "ℹ️ Forecasts tomorrow's AQI from the past 7 days, validated "
            "with a strict time-based split (harder, more honest than "
            "same-day prediction)"
        )

        fig2 = go.Figure(go.Indicator(
            mode="gauge+number",
            value=83.19,
            title={"text": "LSTM Forecast Accuracy %"},
            gauge={
                "axis": {"range": [0, 100]},
                "bar": {"color": "#7B61FF"},
                "steps": [
                    {"range": [0, 50], "color": "#FF6B6B"},
                    {"range": [50, 80], "color": "#FFB347"},
                    {"range": [80, 100], "color": "#00E5A0"},
                ],
                "threshold": {
                    "line": {"color": "white", "width": 4},
                    "thickness": 0.75,
                    "value": 83.19
                }
            }
        ))
        fig2.update_layout(template="plotly_dark", height=300)
        st.plotly_chart(fig2, use_container_width=True)

    st.markdown("---")
    st.markdown("### 📊 Model Comparison")
    comparison = pd.DataFrame({
        "Model": ["XGBoost (pollutant-based)", "LSTM (time-series forecast)"],
        "Task": ["Same-day AQI from PM2.5/PM10/NO2/SO2/CO", "Next-day AQI from 7-day trend"],
        "RMSE": [20.99, 23.51],
        "R²": [0.9237, 0.8319],
        "Validation": ["5-fold CV + held-out test", "Time-based split (train on earlier dates)"]
    })
    st.dataframe(comparison, use_container_width=True)
    st.caption(
        "Both models are trained on real CPCB ground data across 15 Indian "
        "cities (2,565 combined daily records, Jan-Jun 2024) — not simulated data."
    )

# ════════════════════════════════
# PAGE 5 — CITY ANALYSIS
# ════════════════════════════════
elif page == "🏙️ City Analysis":
    st.title("🏙️ City-wise Analysis")
    st.markdown("Deep dive into each city — real CPCB ground data, 15 cities")
    st.markdown("---")

    selected_city = st.selectbox(
        "Select City",
        sorted(df["city"].unique())
    )

    city_data = df[df["city"] == selected_city].sort_values("Date")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Avg AQI", f"{city_data['AQI'].mean():.0f}")
    with col2:
        st.metric("Max AQI", f"{city_data['AQI'].max():.0f}")
    with col3:
        st.metric("Avg PM2.5", f"{city_data['PM2.5'].mean():.1f}")
    with col4:
        aqi_val = city_data["AQI"].mean()
        if aqi_val > 200:
            risk = "HIGH 🔴"
        elif aqi_val > 150:
            risk = "MODERATE 🟡"
        else:
            risk = "LOW 🟢"
        st.metric("Overall AQI Level", risk)

    score, category, emoji = health_risk_score(city_data["AQI"].mean())
    st.markdown("---")
    st.markdown("### ⚕️ Health Risk Score")
    hcol1, hcol2 = st.columns([1, 3])
    with hcol1:
        st.metric("Risk Score (1-10)", f"{score}/10")
    with hcol2:
        st.markdown(f"**Category: {emoji} {category}**")
        if score <= 3:
            st.success("Air quality is safe for most outdoor activities.")
        elif score <= 5:
            st.warning("Sensitive groups (children, elderly, asthma patients) should limit prolonged outdoor exertion.")
        elif score <= 7:
            st.warning("Everyone may experience mild effects; sensitive groups should reduce outdoor activity.")
        else:
            st.error("Health warning: avoid outdoor activity, especially for sensitive groups.")

    st.markdown("---")

    col1, col2 = st.columns(2)

    with col1:
        fig = px.line(
            city_data,
            x="Date", y="AQI",
            title=f"{selected_city} — Daily AQI Trend (2024)",
            color_discrete_sequence=["#00C6FF"]
        )
        fig.update_layout(template="plotly_dark", height=300)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        fig2 = px.line(
            city_data,
            x="Date", y="PM2.5",
            title=f"{selected_city} — Daily PM2.5 Trend (2024)",
            color_discrete_sequence=["#FF6B6B"]
        )
        fig2.update_layout(template="plotly_dark", height=300)
        st.plotly_chart(fig2, use_container_width=True)

    st.markdown("### 🧪 Pollutant Breakdown (Average)")
    pollutants = ["PM2.5", "PM10", "NO2", "SO2", "CO"]
    avg_vals = city_data[pollutants].mean()

    fig3 = px.bar(
        x=pollutants,
        y=avg_vals.values,
        title=f"{selected_city} — Average Pollutants",
        color=pollutants,
        color_discrete_sequence=[
            "#FF6B6B", "#FFB347", "#00C6FF", "#7B61FF", "#00E5A0"
        ]
    )
    fig3.update_layout(template="plotly_dark", height=300, showlegend=False)
    st.plotly_chart(fig3, use_container_width=True)

    st.markdown("### 📋 Raw Data (last 30 days)")
    st.dataframe(
        city_data[["Date", "AQI", "PM2.5", "PM10", "NO2", "SO2", "CO"]]
        .sort_values("Date", ascending=False)
        .head(30)
        .round(2),
        use_container_width=True
    )

# ════════════════════════════════
# PAGE 6 — AQI FORECAST
# ════════════════════════════════
elif page == "🔮 AQI Forecast":
    st.title("🔮 Next-Day AQI Forecast")
    st.markdown(
        "Predicts **tomorrow's AQI** using real CPCB historical "
        "data (Jan-Jun 2024) across **15 Indian cities**, validated "
        "with a time-based train/test split — no data leakage."
    )
    st.markdown("---")

    class LSTMForecast(nn.Module):
        def __init__(self, hidden=32):
            super().__init__()
            self.lstm = nn.LSTM(input_size=1, hidden_size=hidden,
                                 num_layers=2, batch_first=True, dropout=0.2)
            self.fc1 = nn.Linear(hidden, 16)
            self.relu = nn.ReLU()
            self.fc2 = nn.Linear(16, 1)

        def forward(self, x):
            out, _ = self.lstm(x)
            out = out[:, -1, :]
            out = self.relu(self.fc1(out))
            out = self.fc2(out)
            return out.squeeze(-1)

    @st.cache_resource
    def load_lstm():
        model = LSTMForecast()
        import urllib.request, os
        path = "lstm_model_15city.pth"
        if not os.path.exists(path):
            urllib.request.urlretrieve(
                "https://raw.githubusercontent.com/MANIMARAN-V18/"
                "VayuDrishti-ISRO-BAH2026/main/models/lstm_model_15city.pth",
                path
            )
        model.load_state_dict(torch.load(path, map_location="cpu"))
        model.eval()
        return model

    lstm_model = load_lstm()

    selected_city = st.selectbox(
        "Select City", sorted(df["city"].unique()), key="forecast_city"
    )

    city_series = (
        df[df["city"] == selected_city]
        .sort_values("Date")
        .tail(7)
    )

    if len(city_series) < 7:
        st.warning("Not enough recent days of data for this city to forecast.")
    else:
        last7 = city_series["AQI"].values

        fig = px.line(
            city_series, x="Date", y="AQI", markers=True,
            title=f"{selected_city} — Last 7 Days AQI",
            color_discrete_sequence=["#00C6FF"]
        )
        fig.update_layout(template="plotly_dark", height=300)
        st.plotly_chart(fig, use_container_width=True)

        scaler = StandardScaler()
        last7_scaled = scaler.fit_transform(last7.reshape(-1, 1)).flatten()
        x_tensor = torch.tensor(last7_scaled, dtype=torch.float32).view(1, 7, 1)
        with torch.no_grad():
            pred_scaled = lstm_model(x_tensor).item()
        predicted_aqi = scaler.inverse_transform([[pred_scaled]])[0][0]

        st.markdown("---")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Today's AQI", f"{last7[-1]:.0f}")
        with col2:
            delta = predicted_aqi - last7[-1]
            st.metric(
                "Predicted Tomorrow's AQI",
                f"{predicted_aqi:.0f}",
                delta=f"{delta:+.0f}"
            )
        with col3:
            if predicted_aqi > 200:
                st.error("🔴 Category: Poor")
            elif predicted_aqi > 150:
                st.warning("🟡 Category: Moderate-Poor")
            elif predicted_aqi > 100:
                st.warning("🟡 Category: Moderate")
            else:
                st.success("🟢 Category: Satisfactory")

        st.markdown("---")
        st.markdown("### 📊 Model Validation (on held-out real data)")
        val_comparison = pd.DataFrame({
            "Model": ["XGBoost (pollutant-based)", "LSTM (time-series forecast)"],
            "R²": [0.9237, 0.8319],
            "RMSE": [20.99, 23.51],
            "Notes": [
                "Same-day AQI from PM2.5/PM10/NO2/SO2/CO, 15 cities",
                "Next-day AQI, 7-day sequence, 15 cities"
            ]
        })
        st.dataframe(val_comparison, use_container_width=True)
        st.caption(
            "Validated using a **time-based split** (train on earlier dates, "
            "test on later unseen dates) — a harder but honest benchmark "
            "compared to random-split accuracy."
        )

# ════════════════════════════════
# PAGE 7 — LIVE AQI PREDICTOR (NEW)
# ════════════════════════════════
elif page == "🧮 Live AQI Predictor":
    st.title("🧮 Live AQI Predictor")
    st.markdown(
        "Enter pollutant levels and get a **live predicted AQI**, using the "
        "XGBoost model trained on real CPCB data across 15 cities "
        "(test R² = 0.9237)."
    )
    st.markdown("---")

    import joblib

    @st.cache_resource
    def load_xgb_pollutant_model():
        import urllib.request, os
        path = "xgb_pollutant_model_15city.pkl"
        if not os.path.exists(path):
            urllib.request.urlretrieve(
                "https://raw.githubusercontent.com/MANIMARAN-V18/"
                "VayuDrishti-ISRO-BAH2026/main/models/xgb_pollutant_model_15city.pkl",
                path
            )
        return joblib.load(path)

    xgb_model = load_xgb_pollutant_model()

    all_cities = [
        "Ahmedabad", "Bengaluru", "Chennai", "Dehradun", "Delhi",
        "Guwahati", "Hyderabad", "Jodhpur", "Kolkata", "Lucknow",
        "Mumbai", "Patna", "Puducherry", "Raipur", "Thiruvananthapuram"
    ]

    col1, col2 = st.columns(2)

    with col1:
        selected_city = st.selectbox("City", sorted(all_cities))
        selected_month = st.slider("Month", 1, 12, 6)
        pm25 = st.number_input("PM2.5 (µg/m³)", min_value=0.0, max_value=500.0, value=80.0, step=1.0)
        pm10 = st.number_input("PM10 (µg/m³)", min_value=0.0, max_value=600.0, value=120.0, step=1.0)

    with col2:
        no2 = st.number_input("NO2 (µg/m³)", min_value=0.0, max_value=300.0, value=25.0, step=1.0)
        so2 = st.number_input("SO2 (µg/m³)", min_value=0.0, max_value=300.0, value=10.0, step=1.0)
        co = st.number_input("CO (mg/m³)", min_value=0.0, max_value=50.0, value=1.0, step=0.1)

    if st.button("Predict AQI", type="primary"):
        feature_cols = [
            'PM2.5', 'PM10', 'NO2', 'SO2', 'CO', 'month',
            'city_Ahmedabad', 'city_Bengaluru', 'city_Chennai', 'city_Dehradun',
            'city_Delhi', 'city_Guwahati', 'city_Hyderabad', 'city_Jodhpur',
            'city_Kolkata', 'city_Lucknow', 'city_Mumbai', 'city_Patna',
            'city_Puducherry', 'city_Raipur', 'city_Thiruvananthapuram'
        ]

        row = {col: 0 for col in feature_cols}
        row["PM2.5"] = pm25
        row["PM10"] = pm10
        row["NO2"] = no2
        row["SO2"] = so2
        row["CO"] = co
        row["month"] = selected_month
        city_col = f"city_{selected_city}"
        if city_col in row:
            row[city_col] = 1

        X_input = pd.DataFrame([row])[feature_cols]
        predicted_aqi = xgb_model.predict(X_input)[0]

        st.markdown("---")
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Predicted AQI", f"{predicted_aqi:.0f}")
        with col2:
            if predicted_aqi > 300:
                st.error("🔴 Category: Severe")
            elif predicted_aqi > 200:
                st.error("🔴 Category: Very Poor")
            elif predicted_aqi > 150:
                st.warning("🟡 Category: Poor")
            elif predicted_aqi > 100:
                st.warning("🟡 Category: Moderate")
            elif predicted_aqi > 50:
                st.success("🟢 Category: Satisfactory")
            else:
                st.success("🟢 Category: Good")

        score, hrs_category, emoji = health_risk_score(predicted_aqi)
        st.markdown("### ⚕️ Health Risk Score")
        hcol1, hcol2 = st.columns([1, 3])
        with hcol1:
            st.metric("Risk Score (1-10)", f"{score}/10")
        with hcol2:
            st.markdown(f"**{emoji} {hrs_category}**")

        st.caption(
            "Prediction from a real-data-trained XGBoost model (2,565 real "
            "daily records, 15 cities). Not a substitute for official CPCB readings."
        )
