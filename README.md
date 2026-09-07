# 🛰️ VayuDrishti — वायु दृष्टि
## Air Vision for Every Indian

[![ISRO BAH 2026](https://img.shields.io/badge/ISRO-BAH%202026-blue)]()
[![Challenge](https://img.shields.io/badge/Challenge-03-orange)]()
[![Python](https://img.shields.io/badge/Python-3.10+-green)]()
[![Streamlit](https://img.shields.io/badge/Streamlit-Live-red)]()

---

## 🎯 Problem Statement

India has **1.4 billion people** but only **~300 CPCB air quality sensors**.
That means **99.9% of India has NO local AQI data**.

VayuDrishti tackles this using ISRO satellite data, real CPCB ground data, and machine learning — including a deep learning forecasting model that predicts tomorrow's air quality from real historical trends.

---

## 🚀 What is VayuDrishti?

VayuDrishti is an AQI prediction and forecasting system that:

- 🛰️ Uses **ISRO INSAT-3D** + **Sentinel-5P TROPOMI** satellite data
- 🧠 Applies **XGBoost** and **CNN-LSTM** models for same-day AQI/PM2.5 prediction
- 🔮 Forecasts **next-day AQI** using a real-data-trained LSTM model, city by city
- 🔥 Detects **HCHO hotspots** from biomass burning
- 🗺️ Visualizes AQI trends across 10 Indian cities on an interactive dashboard

---

## 📊 Model Performance

### Same-day AQI/PM2.5 prediction (original hackathon models, trained on satellite + ground fusion data)

| Model | RMSE | R² Score |
|-------|------|----------|
| XGBoost (Baseline) | 10.12 | **0.983** |
| CNN-LSTM (Deep Learning) | 19.92 | **0.935** |

### Next-day AQI forecasting (new — trained on real CPCB historical data, time-based validation)

| Model | RMSE | R² Score | Notes |
|-------|------|----------|-------|
| XGBoost (tuned) | 30.02 | 0.7837 | Lag + rolling-average features, hyperparameter search with `TimeSeriesSplit` |
| **LSTM** | **22.83** | **0.8711** | 2-layer LSTM, 7-day input sequence per city, 13,473 parameters |

**Why the forecast R² looks lower:** these two models predict a genuinely **unseen future day**, validated on a strict time-based split (train on earlier dates, test only on later dates). This is a harder, more honest benchmark than same-day prediction and reflects realistic forecasting performance rather than an inflated same-day fit.

---

## 📡 Data Sources

| Dataset | Source | Purpose | Status |
|---------|--------|---------|--------|
| INSAT-3D AOD | MOSDAC (ISRO) | Aerosol data | Used in original models |
| Sentinel-5P NO2/HCHO | Google Earth Engine | Gas columns | Used in original models |
| MODIS Fire | NASA FIRMS | Biomass burning | Used in original models |
| Ground AQI (real, hourly) | CPCB CCR Portal | Ground truth for forecasting | **Added — 39,909 real hourly readings, 10 cities, Jan-Jun 2024** |
| Meteorology | ERA5 (Copernicus) | Wind/humidity | Used in original models |

---

## 🔥 HCHO Hotspot Results

| Rank | City | HCHO (μmol/m²) | Zone |
|------|------|----------------|------|
| 1 | Kolkata | 319.4 | 🔴 Danger |
| 2 | Patna | 280.9 | 🟡 Moderate |
| 3 | Delhi | 276.8 | 🟡 Moderate |
| 9 | Chennai | 152.2 | 🟢 Safe |
| 10 | Puducherry | 147.6 | 🟢 Safe |

---

## 🛠️ Tech Stack

| Layer | Technology |
|-------|-----------|
| Language | Python 3.10+ |
| Satellite Data | Google Earth Engine + MOSDAC |
| Ground Data | CPCB CCR Portal (real hourly station data) |
| Data Processing | Pandas, NumPy, xarray |
| ML Model | XGBoost, Scikit-learn |
| Deep Learning | PyTorch (CNN-LSTM, LSTM forecaster) |
| Explainability | SHAP |
| Visualization | Plotly, Folium, Matplotlib |
| Dashboard | Streamlit (6 pages, incl. live AQI Forecast) |
| Deployment | Streamlit Cloud |

---

## 🖥️ Dashboard Pages

1. **Home** — project overview and key metrics
2. **AQI Map** — interactive India map by month
3. **HCHO Hotspots** — danger-zone ranking and trends
4. **Model Performance** — same-day model comparison
5. **City Analysis** — per-city pollutant breakdown
6. **🔮 AQI Forecast** *(new)* — pick a city, see the last 7 real days, get tomorrow's predicted AQI from the LSTM model, with live model validation scores shown

---

## 🌱 Planned Enhancements (not yet built)

These are genuine ideas for future work — not currently implemented, listed here transparently rather than claimed as shipped:

- **Coastal Meteorology Module** — sea-breeze correction for coastal cities (Puducherry, Chennai, Mumbai)
- **30-Day HCHO Hotspot Forecast** — using multi-year MODIS fire history
- **District-level Health Risk Score** (1-10 scale)
- **Pollutant-level (PM2.5/PM10/NO2) real-time forecasting** — current forecast model uses composite AQI only, since that's what the CPCB portal exposes at scale
- **Expansion to 50+ Indian cities** and 2+ years of historical data (final year project target)

---

## 🏃 How to Run

```bash
# Clone repository
git clone https://github.com/MANIMARAN-V18/VayuDrishti-ISRO-BAH2026

# Install dependencies
pip install -r requirements.txt

# Run dashboard
streamlit run app.py
```

---

## 🎯 Alignment with National Goals

- ✅ **NCAP** — National Clean Air Programme
- ✅ **SDG 11.6** — Reduce urban environmental effects
- ✅ **ISRO Mission** — Satellite data for public benefit

---

## 👥 Team ASTROTECH

- **Manimaran V (Maran)** — Team Lead
- Anas M Y
- Iyyappan N
- Jaidev S

🎓 B.Tech Computer Science Engineering, Final Year (2027 batch)
🏫 Manakula Vinayagar Institute of Technology (MVIT), Puducherry
🛰️ ISRO Bharatiya Antariksh Hackathon 2026 — Challenge 03 (Participation Certificate: 2026H2S06BAH-P05406)

---

## 📞 Contact

**Email:** mani1801maran@gmail.com
**GitHub:** [MANIMARAN-V18](https://github.com/MANIMARAN-V18)

---

*VayuDrishti — Air Vision for Every Indian* 🇮🇳
