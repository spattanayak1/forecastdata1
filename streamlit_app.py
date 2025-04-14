import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")

from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.arima.model import ARIMA
from prophet import Prophet
from datetime import timedelta

# Set page config at the very top!
st.set_page_config(layout="wide", page_title="📈 Sales Forecast App")

# Hide Streamlit default header/footer
hide_st_style = """
    <style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    .css-1rs6os.edgvbvh3 {visibility: hidden;}  /* GitHub corner */
    </style>
"""
st.markdown(hide_st_style, unsafe_allow_html=True)

# Sidebar menu
menu = st.sidebar.radio("Navigation", ["📁 Upload & Forecast", "📊 Analysis", "📜 About"])

# Global state
if 'data' not in st.session_state:
    st.session_state.data = None

# Upload & Forecast
if menu == "📁 Upload & Forecast":
    st.title("📁 Upload Data & Forecast")

    uploaded_file = st.file_uploader("Upload CSV file (max 100 rows)", type=["csv"])
    
    if uploaded_file:
        try:
            df = pd.read_csv(uploaded_file)
            if df.shape[0] > 100:
                st.warning("Please upload a file with max 100 rows.")
            else:
                st.session_state.data = df.copy()
                st.success("File uploaded successfully.")
                
                if 'Week' not in df.columns or 'Sales' not in df.columns:
                    st.error("CSV must have columns: 'Week', 'Sales'")
                else:
                    df = df[["Week", "Sales"]].dropna()
                    df['Week'] = pd.to_datetime(df['Week'], errors='coerce', format="%Y-%m-%d")
                    if df['Week'].isna().sum() > 0:
                        df['Week'] = pd.date_range(start='2023-01-01', periods=len(df), freq='W')
                    
                    df = df.sort_values("Week")

                    st.subheader("📉 Original Data")
                    st.line_chart(df.set_index('Week'))

                    ### ARIMA
                    st.subheader("🔢 ARIMA Forecast")
                    arima_model = ARIMA(df['Sales'], order=(1,1,1)).fit()
                    arima_forecast = arima_model.forecast(steps=3)
                    
                    ### ETS
                    st.subheader("🌀 Exponential Smoothing (ETS)")
                    ets_model = ExponentialSmoothing(df['Sales'], seasonal='add', seasonal_periods=4).fit()
                    ets_forecast = ets_model.forecast(steps=3)

                    ### Prophet
                    st.subheader("🔮 Prophet Forecast")
                    prophet_df = df.rename(columns={"Week": "ds", "Sales": "y"})
                    m = Prophet()
                    m.fit(prophet_df)
                    future = m.make_future_dataframe(periods=3, freq='W')
                    forecast = m.predict(future)

                    ### Final Forecast Comparison
                    st.subheader("📊 Forecast Comparison")

                    last_date = df['Week'].max()
                    future_dates = [last_date + timedelta(weeks=i+1) for i in range(3)]

                    result_df = pd.DataFrame({
                        "Week": future_dates,
                        "ARIMA": arima_forecast.values,
                        "ETS": ets_forecast.values,
                        "Prophet": forecast.tail(3)['yhat'].values
                    })

                    full_plot = df.copy()
                    full_plot = full_plot.append(result_df.rename(columns={"ARIMA": "Sales"}), ignore_index=True)

                    st.line_chart(full_plot.set_index("Week"))

                    st.write("📋 Forecast Table")
                    st.dataframe(result_df.set_index("Week").style.format("{:.2f}"))

        except Exception as e:
            st.error(f"Error processing file: {e}")

# Analysis Page
elif menu == "📊 Analysis":
    st.title("📊 Data Analysis")

    if st.session_state.data is None:
        st.warning("Please upload data first from 'Upload & Forecast'.")
    else:
        df = st.session_state.data

        st.write("### Preview of Uploaded Data")
        st.dataframe(df.head())

        st.write("### Summary Statistics")
        st.dataframe(df.describe())

        st.write("### Distribution")
        fig, ax = plt.subplots()
        sns.histplot(df['Sales'], kde=True, ax=ax)
        st.pyplot(fig)

        st.write("### Boxplot")
        fig, ax = plt.subplots()
        sns.boxplot(x=df['Sales'], ax=ax)
        st.pyplot(fig)

        from scipy.stats import shapiro
        stat, p = shapiro(df['Sales'])
        st.write(f"Shapiro-Wilk Test for Normality: p-value = {p:.4f}")
        if p > 0.05:
            st.success("✅ Data appears to be normally distributed.")
        else:
            st.warning("⚠️ Data may not be normally distributed.")

        from statsmodels.tsa.seasonal import seasonal_decompose
        try:
            df_sorted = df.sort_values("Week")
            df_sorted.set_index('Week', inplace=True)
            result = seasonal_decompose(df_sorted['Sales'], model='additive', period=4)
            st.write("### Seasonality Decomposition")
            fig = result.plot()
            st.pyplot(fig)
        except Exception as e:
            st.error(f"Error in seasonality decomposition: {e}")

# About Page
elif menu == "📜 About":
    st.title("📜 About This App")
    st.markdown("""
    This is a simple time series forecasting app built with **Streamlit**.  
    Upload your own CSV file with `Week` and `Sales`, and get forecasts using:
    - ARIMA
    - ETS
    - Prophet

    Use the menu on the left to navigate between uploading, analysis, and forecasting.
    """)

