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

VayuDrishti solves this using ISRO satellite data and CNN-LSTM deep learning.

---

## 🚀 What is VayuDrishti?

VayuDrishti is India's first intelligent AQI prediction system that:

- 🛰️ Uses **ISRO INSAT-3D** + **Sentinel-5P TROPOMI** satellite data
- 🧠 Applies **CNN-LSTM hybrid deep learning** model
- 🗺️ Predicts **AQI at 3km resolution** across all of India
- 🔥 Detects **HCHO hotspots** from biomass burning
- 🌊 Includes **Coastal Meteorology Module** for coastal cities
- ⚕️ Provides **Health Risk Score** per district

---

## 🌟 4 Unique Innovations

### 1. Hyper-Local 3km AQI Grid
Village-level coverage for all 3.2 million sq km of India.
First time every Indian village gets daily AQI data.

### 2. 30-Day HCHO Hotspot Prediction
Using 5 years of MODIS fire patterns to predict biomass
burning hotspots 30 days in advance.

### 3. Coastal Meteorology Module
Sea breeze correction for coastal cities like Puducherry,
Chennai, Mumbai. No existing global model has this.

### 4. District Health Risk Score
Simple 1-10 score per district — making satellite science
accessible to every Indian citizen.

---

## 📊 Model Performance

| Model | RMSE | R² Score |
|-------|------|----------|
| XGBoost (Baseline) | 10.12 | **0.983** |
| CNN-LSTM (Deep Learning) | 19.92 | **0.935** |

---

## 🛠️ Tech Stack

| Layer | Technology |
|-------|-----------|
| Language | Python 3.10+ |
| Satellite Data | Google Earth Engine + MOSDAC |
| Data Processing | Pandas, NumPy, xarray |
| ML Model | XGBoost, Scikit-learn |
| Deep Learning | PyTorch (CNN-LSTM) |
| Explainability | SHAP |
| Visualization | Plotly, Folium, Matplotlib |
| Dashboard | Streamlit |
| Deployment | Streamlit Cloud |

---

## 📡 Data Sources

| Dataset | Source | Purpose |
|---------|--------|---------|
| INSAT-3D AOD | MOSDAC (ISRO) | Aerosol data |
| Sentinel-5P NO2/HCHO | Google Earth Engine | Gas columns |
| MODIS Fire | NASA FIRMS | Biomass burning |
| Ground AQI | CPCB | Ground truth |
| Meteorology | ERA5 (Copernicus) | Wind/humidity |

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

## 🏃 How to Run

```bash
# Clone repository
git clone https://github.com/YOUR_USERNAME/VayuDrishti-ISRO-BAH2026

# Install dependencies
pip install -r requirements.txt

# Run dashboard
streamlit run app.py
```

---

## 🎯 Alignment with National Goals

- ✅ **NCAP** — National Clean Air Programme
- ✅ **SDG 11.6** — Reduce urban environmental effects
- ✅ **Chintan Shivir 2.0** — Urban climate targets
- ✅ **ISRO Mission** — Satellite data for public benefit

---

## 👥 Team VayuDrishti

- 🎓 B.Tech Computer Science Engineering
- 📍 Puducherry, India
- 🏫 Manakula Vinayagar Institute of Technology
- 🛰️ ISRO Bharatiya Antariksh Hackathon 2026

---

## 📞 Contact

**Email:** mani1801maran@gmail.com

---

*VayuDrishti — Air Vision for Every Indian* 🇮🇳
