# Crime Intelligence & Forecasting System

An end-to-end data science project that analyzes crime patterns, detects high-risk zones, and forecasts future crime trends using statistical and machine learning models.

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

---

## 🗺 Week 1–2: Spatial Crime Intelligence

### Techniques Used:
- KMeans Clustering
- Outlier Removal
- Risk Level Assignment
- Cluster Centroid Extraction
- Intelligence Dataset Export

### Output:
- High / Medium / Low risk zones
- Cluster summary table
- Exported structured intelligence dataset
- Saved clustering model (`kmeans_hotspot_model.pkl`)

---

## 📈 Week 3: Statistical Time-Series Forecasting

### Steps Performed:
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

## 🤖 Week 4: Machine Learning Forecasting

### Feature Engineering
- Lag features (1, 7, 30 days)
- Rolling mean features
- Month and weekday encoding

### Models Trained
- Random Forest Regressor
- XGBoost Regressor

### Machine Learning Model Comparison

| Model | RMSE |
|--------|--------|
| Random Forest | 48.31 |
| XGBoost | 49.13 |

### Final Selected Model

**Random Forest Regressor**

- Lowest RMSE  
- Strong generalization  
- Residuals randomly distributed  
- No significant overfitting  

Machine learning significantly outperformed statistical models.

---

## 📊 Final Model Performance Summary

| Category | Best Model | RMSE |
|------------|-------------|--------|
| Statistical | ARIMA | 106.97 |
| Machine Learning | Random Forest | **48.31** |

---

## 🛠 Tech Stack

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- XGBoost
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
│ ├── 02_time_series_forecasting.ipynb
│ └── 03_machine_learning_models.ipynb
│
├── requirements.txt
└── README.md

---

## 🚀 Future Improvements

- Real-time crime forecasting
- Crime-type specific prediction
- Streamlit interactive dashboard
- Model deployment as API
- Hyperparameter tuning & cross-validation

---

## 🎯 Project Highlights

- End-to-end ML pipeline  
- Spatial + Temporal intelligence  
- Statistical + ML model comparison  
- Feature engineering for time-series regression  
- Model generalization & residual diagnostics  
- Production-style structured repository  