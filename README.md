# 🚔 Crime Intelligence & Forecasting System

An end-to-end data science system that analyzes crime patterns, detects high-risk zones, and forecasts future crime trends using both statistical and machine learning models.

---

## 📌 Problem Statement

Urban crime analysis requires:

- Identification of high-risk geographic zones  
- Detection of temporal crime trends  
- Reliable forecasting for resource allocation  

This project builds a complete crime intelligence pipeline combining spatial clustering, statistical forecasting, and machine learning regression.

---

## 📊 Dataset

Chicago Crime Dataset  
~1.45 million records  

Key Features:

- Date  
- Primary Crime Type  
- Latitude & Longitude  
- Location Description  

Two structured datasets were created:

- `crime_intelligence_dataset.csv` → Spatial intelligence  
- `daily_crime_series.csv` → Time-series forecasting  

---

## 🗺 Spatial Crime Intelligence (Week 1–2)

### Techniques Used
- KMeans Clustering
- Outlier Removal
- Risk Level Assignment
- Cluster Centroid Extraction
- Intelligence Dataset Export

### Output
- High / Medium / Low risk zones
- Cluster summary table
- Exported structured intelligence dataset

Model Saved:
models/kmeans_hotspot_model.pkl


---

## 📈 Statistical Time-Series Forecasting (Week 3)

### Steps Performed
- Daily aggregation of crime counts
- Moving averages (7-day, 30-day)
- Stationarity testing (ADF Test)
- Differencing for stationarity
- ARIMA modeling
- SARIMA modeling
- Model comparison using RMSE

### Statistical Model Comparison

| Model | RMSE |
|--------|--------|
| ARIMA(1,1,1) | 106.97 |
| SARIMA | 136.19 |

ARIMA was selected as the better statistical baseline model.

---

## 🤖 Machine Learning Forecasting (Week 4)

### Feature Engineering
- Lag features (1, 7, 30 days)
- Rolling mean features
- Month encoding
- Weekday encoding

### Models Trained
- Random Forest Regressor
- XGBoost Regressor

### Machine Learning Model Comparison

| Model | RMSE |
|--------|--------|
| Random Forest | **48.31** |
| XGBoost | 49.13 |

---

## ✅ Final Production Model

**Random Forest Regressor**

- Lowest RMSE  
- Reduced error by more than 50% compared to ARIMA  
- Strong generalization performance  
- Stable residual behavior  

Production model saved as:

---

## 📈 Statistical Time-Series Forecasting (Week 3)

### Steps Performed
- Daily aggregation of crime counts
- Moving averages (7-day, 30-day)
- Stationarity testing (ADF Test)
- Differencing for stationarity
- ARIMA modeling
- SARIMA modeling
- Model comparison using RMSE

### Statistical Model Comparison

| Model | RMSE |
|--------|--------|
| ARIMA(1,1,1) | 106.97 |
| SARIMA | 136.19 |

ARIMA was selected as the better statistical baseline model.

---

## 🤖 Machine Learning Forecasting (Week 4)

### Feature Engineering
- Lag features (1, 7, 30 days)
- Rolling mean features
- Month encoding
- Weekday encoding

### Models Trained
- Random Forest Regressor
- XGBoost Regressor

### Machine Learning Model Comparison

| Model | RMSE |
|--------|--------|
| Random Forest | **48.31** |
| XGBoost | 49.13 |

---

## ✅ Final Production Model

**Random Forest Regressor**

- Lowest RMSE  
- Reduced error by more than 50% compared to ARIMA  
- Strong generalization performance  
- Stable residual behavior  

Production model saved as:
models/final_forecast_model.pkl


---

## 🖥 Interactive Dashboard (Week 5)

Built using **Streamlit**, featuring:

- Multi-day crime forecasting
- Historical vs forecast visualization
- Risk zone filtering
- Interactive crime hotspot map
- Model performance comparison panel

Run locally:
streamlit run app/app.py


---

## 🛠 Tech Stack

- Python
- Pandas
- NumPy
- Scikit-learn
- XGBoost
- Statsmodels
- Matplotlib / Seaborn
- Streamlit
- Joblib

---

## 📂 Project Structure

---

## 🛠 Tech Stack

- Python
- Pandas
- NumPy
- Scikit-learn
- XGBoost
- Statsmodels
- Matplotlib / Seaborn
- Streamlit
- Joblib

---

## 📂 Project Structure
crimeintelligencesystem/
│
├── data/
│ ├── raw/
│ └── processed/
│ ├── crime_intelligence_dataset.csv
│ └── daily_crime_series.csv
│
├── models/
│ ├── kmeans_hotspot_model.pkl
│ └── final_forecast_model.pkl
│
├── notebooks/
│ ├── 01_data_exploration.ipynb
│ ├── 02_time_series_forecasting.ipynb
│ └── 03_machine_learning_models.ipynb
│
├── app/
│ └── app.py
│
├── requirements.txt
└── README.md


---

## 🎯 Project Highlights

- End-to-end ML pipeline  
- Spatial + Temporal crime intelligence  
- Statistical vs Machine Learning comparison  
- Feature engineering for time-series regression  
- Production model freezing  
- Interactive deployment using Streamlit  
- Clean modular architecture  

---

## 🚀 Future Improvements

- Crime-type specific forecasting  
- District-level modeling  
- API deployment  
- Hyperparameter optimization  
- Real-time data streaming integration  

## 🚀 Live Demo

🔗 https://crime-intelligence-system-aa3zrcujca2iricnltfoty.streamlit.app