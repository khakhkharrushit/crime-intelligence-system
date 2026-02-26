# Crime Intelligence & Forecasting System

An end-to-end data science project that analyzes crime patterns, detects high-risk zones, and forecasts future crime trends using time-series modeling.

---

## 📌 Problem Statement

Urban crime analysis requires:

- Identification of high-risk geographic zones
- Detection of temporal trends
- Reliable forecasting for resource allocation

This project builds a crime intelligence system using clustering and time-series forecasting techniques.

---

## 📊 Dataset

Chicago Crime Dataset  
~1.45 million records  
Features include:

- Date
- Crime Type
- Latitude
- Longitude
- Location Details

---

## 🗺 Week 1–2: Spatial Crime Intelligence

### Techniques Used:
- KMeans Clustering
- Outlier Removal
- Risk Level Assignment
- Hotspot Summary Generation

### Output:
- High / Medium / Low risk zones
- Cluster centroid coordinates
- Exported intelligence dataset
- Saved clustering model

---

## 📈 Week 3: Time-Series Forecasting

### Steps Performed:
- Daily aggregation of crime counts
- Moving average smoothing
- Stationarity testing (ADF Test)
- ARIMA modeling
- SARIMA modeling
- Model comparison using RMSE

### Model Comparison:

| Model | RMSE |
|--------|--------|
| ARIMA(1,1,1) | 106.97 |
| SARIMA | 136.19 |

### Final Selected Model:
ARIMA(1,1,1) based on better generalization performance.

---

## 🛠 Tech Stack

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- Statsmodels
- Joblib

---

## 📂 Project Structure
crimeintelligencesystem/
│
├── data/
│ ├── raw/
│ └── processed/
│
├── models/
│ └── kmeans_hotspot_model.pkl
│
├── notebooks/
│ ├── 01_data_exploration.ipynb
│ └── 02_time_series_forecasting.ipynb
│
└── README.md


---

## 🚀 Future Improvements

- Machine Learning regression models (Random Forest, XGBoost)
- Crime type forecasting
- Dashboard using Streamlit
- Deployment as API

---

## 🎯 Project Highlights

- End-to-end ML pipeline
- Spatial + Temporal intelligence
- Model evaluation & generalization analysis
- Production-style structure