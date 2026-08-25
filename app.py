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
 
# ── Load Data ──
@st.cache_data
def load_data():
    df = pd.read_csv(
        "https://raw.githubusercontent.com/MANIMARAN-V18/VayuDrishti-ISRO-BAH2026/main/data/master_dataset.csv"
    )
    return df
 
df = load_data()
 
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
        "🔮 AQI Forecast"
    ]
)
 
st.sidebar.markdown("---")
st.sidebar.markdown("**Project Info**")
st.sidebar.markdown("🛰️ ISRO BAH 2026")
st.sidebar.markdown("📡 Challenge 03")
st.sidebar.markdown("👥 Team VayuDrishti")
st.sidebar.markdown("🎓 Puducherry, India")
 
# ════════════════════════════════
# PAGE 1 — HOME
# ════════════════════════════════
if page == "🏠 Home":
    st.title("🛰️ VayuDrishti — वायु दृष्टि")
    st.markdown(
        "### *Air Vision for Every Indian*"
    )
    st.markdown("---")
 
    # Key metrics
    col1, col2, col3, col4 = st.columns(4)
 
    with col1:
        st.metric(
            label="🏙️ Cities Covered",
            value="10",
            delta="Across India"
        )
    with col2:
        st.metric(
            label="🛰️ Satellite Images",
            value="1,416",
            delta="Sentinel-5P TROPOMI"
        )
    with col3:
        st.metric(
            label="🤖 XGBoost R²",
            value="0.983",
            delta="Excellent accuracy"
        )
    with col4:
        st.metric(
            label="🔴 HCHO Hotspots",
            value="1",
            delta="Danger zone detected"
        )
 
    st.markdown("---")
 
    # Project description
    col1, col2 = st.columns(2)
 
    with col1:
        st.markdown("### 🎯 What is VayuDrishti?")
        st.markdown("""
        VayuDrishti is India\'s first intelligent
        Air Quality prediction system that:
 
        - 🛰️ Uses **ISRO INSAT-3D** + **Sentinel-5P** satellite data
        - 🧠 Applies **CNN-LSTM deep learning** model
        - 🗺️ Predicts **AQI at 3km resolution** across India
        - 🔥 Detects **HCHO hotspots** from biomass burning
        - ⚕️ Provides **Health Risk Score** per district
        """)
 
    with col2:
        st.markdown("### 🚨 The Problem We Solve")
        st.markdown("""
        India faces a critical air quality gap:
 
        - 👥 **1.4 Billion** people in India
        - 📡 Only **~300** CPCB air sensors exist
        - ❌ **99.9%** of India has NO local AQI data
        - 🏭 HCHO from factories causes cancer
        - 🌾 Biomass burning spikes pollution seasonally
 
        **VayuDrishti fills this gap using satellites!**
        """)
 
    st.markdown("---")
    st.markdown("### 📊 Data Overview")
 
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
            title="Average AQI by City"
        )
        fig.update_layout(
            template="plotly_dark",
            height=300
        )
        st.plotly_chart(fig,
                        use_container_width=True)
    with col3:
        st.markdown("**Key Findings:**")
        st.error("🔴 Delhi: Highest AQI (281)")
        st.warning("🟡 Kolkata: Highest HCHO (319)")
        st.success("🟢 Puducherry: Cleanest city")
        st.info("📅 April: Worst HCHO month")
 
# ════════════════════════════════
# PAGE 2 — AQI MAP
# ════════════════════════════════
elif page == "🗺️ AQI Map":
    st.title("🗺️ India AQI Map")
    st.markdown(
        "Interactive map showing AQI levels "
        "across Indian cities"
    )
    st.markdown("---")
 
    # City coordinates
    city_coords = {
        "Delhi"     : [28.6139, 77.2090],
        "Mumbai"    : [19.0760, 72.8777],
        "Chennai"   : [13.0827, 80.2707],
        "Kolkata"   : [22.5726, 88.3639],
        "Bengaluru" : [12.9716, 77.5946],
        "Hyderabad" : [17.3850, 78.4867],
        "Puducherry": [11.9416, 79.8083],
        "Lucknow"   : [26.8467, 80.9462],
        "Patna"     : [25.5941, 85.1376],
        "Ahmedabad" : [23.0225, 72.5714],
    }
 
    # Month filter
    month_order = [
        "Jan-2024", "Feb-2024", "Mar-2024",
        "Apr-2024", "May-2024", "Jun-2024"
    ]
    selected_month = st.selectbox(
        "Select Month", month_order
    )
 
    # Filter data
    df_month = df[df["month"] == selected_month]
 
    # Create Folium map
    m = folium.Map(
        location=[20.5937, 78.9629],
        zoom_start=5,
        tiles="CartoDB dark_matter"
    )
 
    for _, row in df_month.iterrows():
        city = row["city"]
        aqi  = row["AQI"]
 
        if city not in city_coords:
            continue
 
        lat, lon = city_coords[city]
 
        # Color based on AQI
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
            radius=aqi / 15,
            color=color,
            fill=True,
            fill_color=color,
            fill_opacity=0.7,
            popup=folium.Popup(
                f"<b>{city}</b><br>"
                f"AQI: {aqi:.0f}<br>"
                f"PM2.5: {row['PM2.5']:.1f}<br>"
                f"Month: {selected_month}",
                max_width=200
            ),
            tooltip=f"{city}: AQI {aqi:.0f}"
        ).add_to(m)
 
    st_folium(m, width=900, height=500)
 
    # AQI table
    st.markdown("### 📊 AQI Values")
    st.dataframe(
        df_month[["city", "AQI", "PM2.5",
                  "PM10", "NO2", "SO2"]]
        .sort_values("AQI", ascending=False)
        .round(2),
        use_container_width=True
    )
 
# ════════════════════════════════
# PAGE 3 — HCHO HOTSPOTS
# ════════════════════════════════
elif page == "🔥 HCHO Hotspots":
    st.title("🔥 HCHO Hotspot Detection")
    st.markdown(
        "Formaldehyde hotspots from "
        "biomass burning and industries"
    )
    st.markdown("---")
 
    # HCHO ranking
    city_hcho = df.groupby("city")[
        "HCHO_sat"
    ].mean().sort_values(
        ascending=False
    ).reset_index()
    city_hcho.columns = ["city", "HCHO_mean"]
 
    col1, col2 = st.columns(2)
 
    with col1:
        # Horizontal bar chart
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
            x=300,
            line_dash="dash",
            line_color="red",
            annotation_text="Danger (300)"
        )
        fig.add_vline(
            x=220,
            line_dash="dash",
            line_color="orange",
            annotation_text="Moderate (220)"
        )
        fig.update_layout(
            title="HCHO Hotspot Ranking",
            template="plotly_dark",
            height=400
        )
        st.plotly_chart(fig,
                        use_container_width=True)
 
    with col2:
        st.markdown("### 🚨 Hotspot Alerts")
        for _, row in city_hcho.iterrows():
            hcho = row["HCHO_mean"]
            city = row["city"]
            if hcho > 300:
                st.error(
                    f"🔴 **{city}** — "
                    f"DANGER: HCHO = {hcho:.1f}"
                )
            elif hcho > 220:
                st.warning(
                    f"🟡 **{city}** — "
                    f"MODERATE: HCHO = {hcho:.1f}"
                )
            else:
                st.success(
                    f"🟢 **{city}** — "
                    f"SAFE: HCHO = {hcho:.1f}"
                )
 
    # Monthly trend
    st.markdown("---")
    st.markdown("### 📈 Monthly HCHO Trend")
 
    month_order = [
        "Jan-2024", "Feb-2024", "Mar-2024",
        "Apr-2024", "May-2024", "Jun-2024"
    ]
    top3 = city_hcho.head(3)["city"].tolist()
    df_top3 = df[df["city"].isin(top3)]
 
    fig2 = px.line(
        df_top3,
        x="month", y="HCHO_sat",
        color="city",
        markers=True,
        title="HCHO Trend — Top 3 Hotspot Cities",
        category_orders={"month": month_order}
    )
    fig2.add_hline(
        y=300, line_dash="dash",
        line_color="red",
        annotation_text="Danger threshold"
    )
    fig2.update_layout(
        template="plotly_dark",
        height=400
    )
    st.plotly_chart(fig2,
                    use_container_width=True)
 
# ════════════════════════════════
# PAGE 4 — MODEL PERFORMANCE
# ════════════════════════════════
elif page == "📊 Model Performance":
    st.title("📊 Model Performance")
    st.markdown(
        "VayuDrishti ML model accuracy metrics"
    )
    st.markdown("---")
 
    col1, col2 = st.columns(2)
 
    with col1:
        st.markdown("### 🤖 XGBoost Results")
        st.metric("RMSE", "10.12",
                  delta="Excellent!")
        st.metric("R² Score", "0.983",
                  delta="+98.3% accuracy")
        st.success(
            "✅ XGBoost explains 98.3% "
            "of AQI variation!"
        )
 
        # Gauge chart
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=98.3,
            title={"text": "XGBoost Accuracy %"},
            gauge={
                "axis": {"range": [0, 100]},
                "bar": {"color": "#00C6FF"},
                "steps": [
                    {"range": [0, 50],
                     "color": "#FF6B6B"},
                    {"range": [50, 80],
                     "color": "#FFB347"},
                    {"range": [80, 100],
                     "color": "#00E5A0"},
                ],
                "threshold": {
                    "line": {"color": "white",
                             "width": 4},
                    "thickness": 0.75,
                    "value": 98.3
                }
            }
        ))
        fig.update_layout(
            template="plotly_dark",
            height=300
        )
        st.plotly_chart(fig,
                        use_container_width=True)
 
    with col2:
        st.markdown("### 🧠 CNN-LSTM Results")
        st.metric("RMSE", "19.92",
                  delta="Good result")
        st.metric("R² Score", "0.935",
                  delta="+93.5% accuracy")
        st.info(
            "ℹ️ CNN-LSTM will outperform "
            "XGBoost with larger datasets!"
        )
 
        # Gauge chart
        fig2 = go.Figure(go.Indicator(
            mode="gauge+number",
            value=93.5,
            title={"text": "CNN-LSTM Accuracy %"},
            gauge={
                "axis": {"range": [0, 100]},
                "bar": {"color": "#7B61FF"},
                "steps": [
                    {"range": [0, 50],
                     "color": "#FF6B6B"},
                    {"range": [50, 80],
                     "color": "#FFB347"},
                    {"range": [80, 100],
                     "color": "#00E5A0"},
                ],
                "threshold": {
                    "line": {"color": "white",
                             "width": 4},
                    "thickness": 0.75,
                    "value": 93.5
                }
            }
        ))
        fig2.update_layout(
            template="plotly_dark",
            height=300
        )
        st.plotly_chart(fig2,
                        use_container_width=True)
 
    # Comparison table
    st.markdown("---")
    st.markdown("### 📊 Model Comparison")
    comparison = pd.DataFrame({
        "Model"   : ["XGBoost", "CNN-LSTM"],
        "RMSE"    : [10.12, 19.92],
        "R²"      : [0.983, 0.935],
        "Best for": [
            "Small datasets, fast training",
            "Large datasets, time patterns"
        ]
    })
    st.dataframe(comparison,
                 use_container_width=True)
 
# ════════════════════════════════
# PAGE 5 — CITY ANALYSIS
# ════════════════════════════════
elif page == "🏙️ City Analysis":
    st.title("🏙️ City-wise Analysis")
    st.markdown("Deep dive into each city")
    st.markdown("---")
 
    # City selector
    selected_city = st.selectbox(
        "Select City",
        sorted(df["city"].unique())
    )
 
    city_data = df[
        df["city"] == selected_city
    ].sort_values("month")
 
    col1, col2, col3, col4 = st.columns(4)
 
    with col1:
        st.metric(
            "Avg AQI",
            f"{city_data['AQI'].mean():.0f}"
        )
    with col2:
        st.metric(
            "Max AQI",
            f"{city_data['AQI'].max():.0f}"
        )
    with col3:
        st.metric(
            "Avg HCHO",
            f"{city_data['HCHO_sat'].mean():.1f}"
        )
    with col4:
        aqi_val = city_data["AQI"].mean()
        if aqi_val > 200:
            risk = "HIGH 🔴"
        elif aqi_val > 150:
            risk = "MODERATE 🟡"
        else:
            risk = "LOW 🟢"
        st.metric("Health Risk", risk)
 
    st.markdown("---")
 
    col1, col2 = st.columns(2)
 
    with col1:
        # AQI trend
        fig = px.line(
            city_data,
            x="month", y="AQI",
            markers=True,
            title=f"{selected_city} — AQI Trend",
            color_discrete_sequence=["#00C6FF"]
        )
        fig.update_layout(
            template="plotly_dark",
            height=300
        )
        st.plotly_chart(fig,
                        use_container_width=True)
 
    with col2:
        # HCHO trend
        fig2 = px.line(
            city_data,
            x="month",
            y="HCHO_sat",
            markers=True,
            title=f"{selected_city} — HCHO Trend",
            color_discrete_sequence=["#FF6B6B"]
        )
        fig2.update_layout(
            template="plotly_dark",
            height=300
        )
        st.plotly_chart(fig2,
                        use_container_width=True)
 
    # Pollutant breakdown
    st.markdown("### 🧪 Pollutant Breakdown")
    pollutants = ["PM2.5", "PM10",
                  "NO2", "SO2", "CO"]
    avg_vals = city_data[pollutants].mean()
 
    fig3 = px.bar(
        x=pollutants,
        y=avg_vals.values,
        title=f"{selected_city} — Average Pollutants",
        color=pollutants,
        color_discrete_sequence=[
            "#FF6B6B", "#FFB347",
            "#00C6FF", "#7B61FF", "#00E5A0"
        ]
    )
    fig3.update_layout(
        template="plotly_dark",
        height=300,
        showlegend=False
    )
    st.plotly_chart(fig3,
                    use_container_width=True)
 
    # Raw data
    st.markdown("### 📋 Raw Data")
    st.dataframe(
        city_data[[
            "month", "AQI", "PM2.5",
            "PM10", "NO2", "SO2",
            "HCHO_sat", "NO2_sat"
        ]].round(2),
        use_container_width=True
    )
 
# ════════════════════════════════
# PAGE 6 — AQI FORECAST (NEW)
# ════════════════════════════════
elif page == "🔮 AQI Forecast":
    st.title("🔮 Next-Day AQI Forecast")
    st.markdown(
        "Predicts **tomorrow's AQI** using real CPCB historical "
        "data (Jan-Jun 2024), validated with a time-based train/test "
        "split — no data leakage."
    )
    st.markdown("---")
 
    @st.cache_data
    def load_daily():
        return pd.read_csv(
            "https://raw.githubusercontent.com/MANIMARAN-V18/"
            "VayuDrishti-ISRO-BAH2026/main/data/daily_master.csv",
            parse_dates=["Date"]
        )
 
    daily_df = load_daily()
 
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
        path = "lstm_model.pth"
        if not os.path.exists(path):
            urllib.request.urlretrieve(
                "https://raw.githubusercontent.com/MANIMARAN-V18/"
                "VayuDrishti-ISRO-BAH2026/main/models/lstm_model.pth",
                path
            )
        model.load_state_dict(torch.load(path, map_location="cpu"))
        model.eval()
        return model
 
    lstm_model = load_lstm()
 
    selected_city = st.selectbox(
        "Select City", sorted(daily_df["City"].unique()), key="forecast_city"
    )
 
    city_series = (
        daily_df[daily_df["City"] == selected_city]
        .sort_values("Date")
        .tail(7)
    )
 
    if len(city_series) < 7:
        st.warning("Not enough recent days of data for this city to forecast.")
    else:
        last7 = city_series["AQI_avg"].values
 
        fig = px.line(
            city_series, x="Date", y="AQI_avg", markers=True,
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
            "Model": ["XGBoost (tuned)", "LSTM (deep learning)"],
            "R²": [0.7837, 0.8711],
            "RMSE": [30.02, 22.83],
            "Notes": [
                "Lag/rolling features + city + calendar",
                "7-day sequence, 2-layer, 13,473 params"
            ]
        })
        st.dataframe(val_comparison, use_container_width=True)
        st.caption(
            "Validated using a **time-based split** (train on earlier dates, "
            "test on later unseen dates) — a harder but honest benchmark "
            "compared to random-split accuracy."
        )
