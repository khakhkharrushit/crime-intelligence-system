import streamlit as st
import joblib
import pandas as pd
import numpy as np
import os

# -------------------------------
# Safe Path Handling
# -------------------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

MODEL_PATH = os.path.join(BASE_DIR, "models", "final_forecast_model.pkl")

INTEL_PATH = os.path.join(BASE_DIR, "data", "processed", "crime_intelligence_dataset.csv")
TS_PATH = os.path.join(BASE_DIR, "data", "processed", "daily_crime_series.csv")

model = joblib.load(MODEL_PATH)

df_intel = pd.read_csv(INTEL_PATH)
df_ts = pd.read_csv(TS_PATH)
# -------------------------------
# Page Configuration
# -------------------------------
st.set_page_config(
    page_title="Crime Intelligence Dashboard",
    layout="wide"
)

st.title("🚔 Crime Intelligence & Forecasting Dashboard")
st.markdown("Spatial Risk Detection + Machine Learning Forecasting System")

st.divider()

# -------------------------------
# Tabs
# -------------------------------
tab1, tab2, tab3 = st.tabs([
    "📈 Forecast",
    "🗺 Risk Zones",
    "📊 Model Insights"
])

# ==========================================================
# TAB 1 — ADVANCED FORECAST
# ==========================================================
with tab1:

    st.subheader("📊 Advanced Crime Forecasting")

    # Use clean daily time-series dataset
    df_ts["Date"] = pd.to_datetime(df_ts["Date"])
    daily_data = df_ts.sort_values("Date")

    st.markdown("### 📈 Historical Crime Trend (Last 90 Days)")
    st.line_chart(
        daily_data.set_index("Date")["Crime_Count"].tail(90)
    )

    st.divider()

    st.markdown("### 🔮 Multi-Day Forecast")

    forecast_days = st.slider("Select Forecast Horizon (Days)", 1, 14, 7)

    # Automatically calculate latest features from real data
    latest = daily_data.tail(30)

    lag_1 = latest["Crime_Count"].iloc[-1]
    lag_7 = latest["Crime_Count"].iloc[-7]
    lag_30 = latest["Crime_Count"].iloc[0]

    month = latest["Date"].iloc[-1].month
    weekday = latest["Date"].iloc[-1].weekday()

    predictions = []

    for i in range(forecast_days):

        rolling_mean_7 = np.mean([lag_1, lag_7])
        rolling_mean_30 = np.mean([lag_1, lag_30])

        input_data = pd.DataFrame([[
            lag_1,
            lag_7,
            lag_30,
            rolling_mean_7,
            rolling_mean_30,
            month,
            weekday
        ]], columns=[
            "lag_1",
            "lag_7",
            "lag_30",
            "rolling_mean_7",
            "rolling_mean_30",
            "month",
            "weekday"
        ])

        prediction = model.predict(input_data)[0]
        predictions.append(int(prediction))

        # Update lags for next iteration
        lag_30 = lag_7
        lag_7 = lag_1
        lag_1 = prediction
        weekday = (weekday + 1) % 7

    forecast_df = pd.DataFrame({
        "Day": range(1, forecast_days + 1),
        "Predicted Crime Count": predictions
    })

    st.dataframe(forecast_df)

    st.markdown("### 📊 Actual vs Forecast (Extended View)")

    # Last 30 actual days
    last_actual = daily_data.tail(30).copy()

    # Create future dates
    last_date = last_actual["Date"].iloc[-1]
    future_dates = pd.date_range(
        start=last_date + pd.Timedelta(days=1),
        periods=forecast_days
    )

    forecast_extended = pd.DataFrame({
        "Date": future_dates,
        "Crime_Count": predictions
    })

    # Combine actual + forecast
    combined = pd.concat([
        last_actual[["Date", "Crime_Count"]],
        forecast_extended
    ])

    combined = combined.set_index("Date")

    st.line_chart(combined)

    st.caption("Blue line shows last 30 actual days followed by forecasted values.")

    st.info("Multi-step forecasting generated using iterative Random Forest predictions.")

# ==========================================================
# TAB 2 — RISK ZONES
# ==========================================================
with tab2:

    st.subheader("🗺 Crime Risk Zones Map")

    selected_risk = st.selectbox(
        "Filter by Risk Level",
        ["All", "High Risk", "Medium Risk", "Low Risk"]
    )

    if selected_risk != "All":
        df_filtered = df_intel[df_intel["Risk_Level"] == selected_risk]
    else:
        df_filtered = df_intel

    # Performance sampling
    df_filtered = df_filtered.sample(min(5000, len(df_filtered)))

    st.map(df_filtered.rename(columns={
        "Latitude": "lat",
        "Longitude": "lon"
    }))

    st.divider()

    st.subheader("📊 Risk Summary")

    risk_counts = df_intel["Risk_Level"].value_counts()

    col1, col2, col3 = st.columns(3)

    col1.metric("High Risk Zones", risk_counts.get("High Risk", 0))
    col2.metric("Medium Risk Zones", risk_counts.get("Medium Risk", 0))
    col3.metric("Low Risk Zones", risk_counts.get("Low Risk", 0))

# ==========================================================
# TAB 3 — MODEL INSIGHTS
# ==========================================================
with tab3:

    st.subheader("📊 Model Performance Comparison")

    comparison_df = pd.DataFrame({
        "Model": ["ARIMA", "SARIMA", "Random Forest", "XGBoost"],
        "RMSE": [106.97, 136.19, 48.31, 49.13]
    })

    st.dataframe(comparison_df)

    st.divider()

    st.subheader("🧠 Key Insights")

    st.write("""
    - Random Forest significantly outperformed statistical models.
    - 7-day rolling mean was the strongest predictive feature.
    - Model shows strong generalization with stable residuals.
    - Machine Learning reduced error by more than 50% compared to ARIMA.
    """)

    st.divider()

    st.subheader("🛠 Model Details")

    st.write("""
    Final Model: Random Forest Regressor  
    RMSE: 48.31  
    Features Used:
    - Lag values (1, 7, 30 days)
    - Rolling means
    - Month encoding
    - Weekday encoding
    """)