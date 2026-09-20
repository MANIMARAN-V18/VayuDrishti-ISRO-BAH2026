# 🛰️ VayuDrishti — वायु दृष्टि
## Air Vision for Every Indian

[![ISRO BAH 2026](https://img.shields.io/badge/ISRO-BAH%202026-blue)]()
[![Challenge](https://img.shields.io/badge/Challenge-03-orange)]()
[![Python](https://img.shields.io/badge/Python-3.10+-green)]()
[![Streamlit](https://img.shields.io/badge/Streamlit-Live-red)]()

---

## 🎯 Problem Statement

India has **1.4 billion people** but only **~300 CPCB air quality sensors**.
That means most of India has no local, forecastable AQI data.

VayuDrishti tackles this using real CPCB ground station data, ISRO/Sentinel satellite fusion, and two independently validated machine learning models — one predicting today's AQI from pollutant levels, and one forecasting tomorrow's AQI from recent trends.

---

## 🚀 What is VayuDrishti?

VayuDrishti is an AQI prediction and forecasting system that:

- 📡 Uses **real, ground-truth CPCB station data** across **15 Indian cities**
- 🧠 Applies **XGBoost** to predict same-day AQI from pollutant readings (PM2.5, PM10, NO2, SO2, CO)
- 🔮 Applies an **LSTM** to forecast **next-day AQI** from the past 7 days' trend, per city
- 🛰️ Uses **ISRO INSAT-3D** + **Sentinel-5P TROPOMI** satellite data to detect **HCHO hotspots** from biomass burning (10-city scope)
- 🗺️ Visualizes AQI and pollutant trends across all 15 cities on a live interactive dashboard

---

## 📊 Model Performance

### Same-day AQI prediction from pollutants (XGBoost) — real data, 15 cities

| Metric | Value |
|--------|-------|
| Dataset | 2,565 real daily records, 15 cities, Jan-Jun 2024 |
| Validation | 5-fold CV (R²=0.9117) + held-out test set |
| **Test R²** | **0.9237** |
| **Test RMSE** | **20.99** |
| Top features | PM2.5 (54%), PM10 (25%) — matches known atmospheric science |

### Next-day AQI forecasting (LSTM) — real data, 15 cities, time-based validation

| Metric | Value |
|--------|-------|
| Dataset | Same 2,565-record base, sequenced per city |
| Validation | Strict time-based split (train on earlier dates, test on later unseen dates) |
| **Test R²** | **0.8319** |
| **Test RMSE** | **23.51** |
| Architecture | 2-layer LSTM, 32 hidden units, 13,473 parameters, 7-day input window |

**Why the forecast R² is lower than the same-day model:** forecasting an unseen future day is a genuinely harder task than fitting to known same-day pollutant readings. Both numbers are honestly reported from real held-out data — no random-split inflation.

### Earlier hackathon-stage models (10-city, satellite-fusion features)

| Model | RMSE | R² Score |
|-------|------|----------|
| XGBoost (original) | 10.12 | 0.983 |
| CNN-LSTM (original) | 19.92 | 0.935 |

These were trained on a small simulated dataset (~60 rows) fused with satellite features, and are kept here for historical comparison. The 15-city models above are the current, real-data-validated versions used in the live dashboard.

---

## 📡 Data Sources

| Dataset | Source | Purpose | Scope |
|---------|--------|---------|-------|
| Real hourly AQI | CPCB CCR Portal | Ground-truth AQI | 15 cities |
| Real 15-min pollutant readings (PM2.5, PM10, NO2, SO2, CO) | CPCB CCR Advance Search | Pollutant-based prediction | 15 cities |
| INSAT-3D AOD | MOSDAC (ISRO) | Aerosol data | 10 cities (original scope) |
| Sentinel-5P NO2/HCHO | Google Earth Engine | HCHO hotspot detection | 10 cities (original scope) |
| MODIS Fire | NASA FIRMS | Biomass burning | 10 cities (original scope) |
| Meteorology | ERA5 (Copernicus) | Wind/humidity | 10 cities (original scope) |

**15 cities covered (ground data):** Delhi, Mumbai, Chennai, Kolkata, Bengaluru, Hyderabad, Puducherry, Lucknow, Patna, Ahmedabad, Guwahati, Thiruvananthapuram, Dehradun, Jodhpur, Raipur.

---

## 🔥 HCHO Hotspot Results (10-city satellite scope)

| Rank | City | HCHO (μmol/m²) | Zone |
|------|------|----------------|------|
| 1 | Kolkata | 319.4 | 🔴 Danger |
| 2 | Patna | 280.9 | 🟡 Moderate |
| 3 | Delhi | 276.8 | 🟡 Moderate |
| 9 | Chennai | 152.2 | 🟢 Safe |
| 10 | Puducherry | 147.6 | 🟢 Safe |

HCHO detection has not yet been extended to the 5 newly added cities — this depends on satellite fusion work, which is a planned future enhancement.

---

## 🛠️ Tech Stack

| Layer | Technology |
|-------|-----------|
| Language | Python 3.10+ |
| Ground Data | CPCB CCR Portal (real hourly AQI + 15-min pollutant readings) |
| Satellite Data | Google Earth Engine + MOSDAC |
| Data Processing | Pandas, NumPy |
| ML Model | XGBoost, Scikit-learn |
| Deep Learning | PyTorch (LSTM forecaster) |
| Visualization | Plotly, Folium |
| Dashboard | Streamlit (6 pages) |
| Deployment | Streamlit Cloud |

---

## 🖥️ Dashboard Pages

1. **Home** — 15-city overview, both models' headline metrics
2. **AQI Map** — interactive map, all 15 cities, AQI + pollutant popup
3. **HCHO Hotspots** — satellite-based danger-zone ranking (10-city scope, clearly labeled)
4. **Model Performance** — honest comparison of the XGBoost and LSTM models
5. **City Analysis** — per-city real AQI and pollutant trends, all 15 cities
6. **🔮 AQI Forecast** — pick a city, see the last 7 real days, get tomorrow's predicted AQI live from the LSTM model

---

## 🌱 Planned Enhancements (not yet built)

Listed transparently as future work, not shipped features:

- **HCHO/satellite fusion for the 5 new cities** — extending hotspot detection beyond the original 10
- **Coastal Meteorology Module** — sea-breeze correction for coastal cities (Puducherry, Chennai, Mumbai)
- **30-Day HCHO Hotspot Forecast** — using multi-year MODIS fire history
- **District-level Health Risk Score** (1-10 scale)
- **Live pollutant-input prediction tool** — a page where users enter PM2.5/PM10/NO2/SO2/CO and get a live XGBoost AQI prediction (model already trained, not yet wired into the UI)
- **Expansion to 30+ Indian cities** and 2+ years of historical data (final year project target)

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



## 📞 Contact

**Email:** mani1801maran@gmail.com
**GitHub:** [MANIMARAN-V18](https://github.com/MANIMARAN-V18)

---

*VayuDrishti — Air Vision for Every Indian* 🇮🇳
