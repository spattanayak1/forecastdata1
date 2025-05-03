import warnings
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.arima.model import ARIMA
from prophet import Prophet
from datetime import timedelta
import numpy as np

warnings.filterwarnings("ignore")

# Set page config before anything else
st.set_page_config(layout="wide", page_title="📈 Sales Forecast App")

# Hide GitHub icon and Streamlit UI elements
hide_streamlit_style = """
    <style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    </style>
"""
st.markdown(hide_streamlit_style, unsafe_allow_html=True)
st.sidebar.title("🔍 Navigation")
# Sidebar navigation
menu = st.sidebar.radio("📋 Menu", ["📅 Forecast"])

if menu == "📅 Forecast":
    # Title
    st.title("📊 Sales Forecast App")

    # Upload section
    uploaded_file = st.file_uploader("📁 Upload your time-series CSV (max 100 rows)", type=["csv"])

    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)

            if df.shape[0] > 100:
                st.warning("❗ Please upload a CSV with 100 rows or fewer.")
            else:
                st.success("✅ File uploaded successfully!")

                # Ensure it has two columns
                if df.shape[1] != 2:
                    st.error("CSV must have exactly two columns: 'Week' and 'Sales'")
                else:
                    df.columns = ['Week', 'Sales']
                    df['Week'] = pd.to_numeric(df['Week'], errors='coerce')
                    df['Sales'] = pd.to_numeric(df['Sales'], errors='coerce')
                    df.dropna(inplace=True)
                    df = df.sort_values('Week')
                    df.reset_index(drop=True, inplace=True)

                    # ---------------------- Analysis Section ----------------------
                    st.header("📈 Exploratory Analysis")

                    st.subheader("🔹 Raw Data")
                    st.dataframe(df)

                    st.subheader("🔹 Summary Statistics")
                    st.write(df['Sales'].describe())

                    st.subheader("🔹 Mean & Median")
                    st.write(f"Mean Sales: {df['Sales'].mean():.2f}")
                    st.write(f"Median Sales: {df['Sales'].median():.2f}")

                    st.subheader("🔹 Normality (Histogram)")
                    fig, ax = plt.subplots()
                    sns.histplot(df['Sales'], kde=True, ax=ax)
                    st.pyplot(fig)

                    st.subheader("🔹 Stationarity (ADF Test)")
                    result = adfuller(df['Sales'])
                    st.write(f"ADF Statistic: {result[0]:.2f}")
                    st.write(f"p-value: {result[1]:.4f}")
                    if result[1] < 0.05:
                        st.success("✅ The series is stationary.")
                    else:
                        st.warning("⚠️ The series is not stationary.")

                    st.subheader("🔹 Sales Over Time")
                    fig, ax = plt.subplots()
                    sns.lineplot(x=df['Week'], y=df['Sales'], ax=ax)
                    ax.set_title("Sales by Week")
                    st.pyplot(fig)

                    # ---------------------- Forecast Section ----------------------
                    st.header("📅 Forecasting (Next 3 Weeks)")

                    # Prepare future dataframe
                    future_weeks = list(range(df['Week'].max()+1, df['Week'].max()+4))

                    # ETS Forecast
                    ets_model = ExponentialSmoothing(df['Sales'], trend='add', seasonal=None).fit()
                    ets_forecast = ets_model.forecast(3)

                    # ARIMA Forecast
                    arima_model = ARIMA(df['Sales'], order=(1,1,1)).fit()
                    arima_forecast = arima_model.forecast(3)

                    # Prophet Forecast
                    prophet_df = df.copy()
                    prophet_df['ds'] = pd.date_range(start='2024-01-01', periods=len(df), freq='W')
                    prophet_df.rename(columns={'Sales': 'y'}, inplace=True)
                    prophet_model = Prophet()
                    prophet_model.fit(prophet_df[['ds', 'y']])
                    future = prophet_model.make_future_dataframe(periods=3, freq='W')
                    forecast = prophet_model.predict(future)
                    prophet_forecast = forecast[['ds', 'yhat']].tail(3).reset_index(drop=True)

                    # Combine forecasts into a table
                    forecast_df = pd.DataFrame({
                        'Week': future_weeks,
                        'ETS': ets_forecast.values,
                        'ARIMA': arima_forecast.values,
                        'Prophet': prophet_forecast['yhat'].values
                    })

                    st.subheader("🔹 Individual Forecasts")
                    st.write("📌 ETS Forecast:", ets_forecast.values)
                    st.write("📌 ARIMA Forecast:", arima_forecast.values)
                    st.write("📌 Prophet Forecast:", prophet_forecast['yhat'].values)

                    st.subheader("📊 Combined Forecast Table")
                    st.dataframe(forecast_df)

                    # Plot comparison chart
                    st.subheader("📈 Forecast vs Original (Line Chart)")
                    fig, ax = plt.subplots(figsize=(10, 5))
                    ax.plot(df['Week'], df['Sales'], label='Actual Sales')
                    ax.plot(forecast_df['Week'], forecast_df['ETS'], label='ETS Forecast', linestyle='--')
                    ax.plot(forecast_df['Week'], forecast_df['ARIMA'], label='ARIMA Forecast', linestyle='--')
                    ax.plot(forecast_df['Week'], forecast_df['Prophet'], label='Prophet Forecast', linestyle='--')
                    ax.set_xlabel('Week')
                    ax.set_ylabel('Sales')
                    ax.set_title('Actual vs Forecast (Next 3 Weeks)')
                    ax.legend()
                    st.pyplot(fig)

        except Exception as e:
            st.error(f"Error processing file: {e}")
    else:
        st.info("👈 Upload a CSV file to begin.")
