import streamlit as st
import pandas as pd
import numpy as np
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from scipy.stats import shapiro
import matplotlib.pyplot as plt

# Set page config
st.set_page_config(page_title="Sales Forecast App", layout="wide")
# Ensure session state initialized
if "data" not in st.session_state:
    st.session_state["data"] = None

# Title
st.title("📊 Sales Analysis & Forecasting App")

# Sidebar navigation
menu = st.sidebar.radio("Navigation", ["Upload Data", "Analysis", "Forecast"])

# Initialize session state for data
if "data" not in st.session_state:
    st.session_state.data = None

# Upload Menu
if menu == "Upload Data":
    st.header("📁 Upload Your CSV File")
    st.markdown("Please upload a CSV file with **Week** and **Sales** columns. Max 100 rows.")

    uploaded_file = st.file_uploader("Upload CSV", type=['csv'])
if uploaded_file is not None:
    try:
        # Try reading using utf-8, fallback to other encodings
        try:
            df = pd.read_csv(uploaded_file, encoding='utf-8')
        except UnicodeDecodeError:
            try:
                df = pd.read_csv(uploaded_file, encoding='ISO-8859-1')  # common fallback
            except:
                df = pd.read_csv(uploaded_file, encoding='utf-16')  # another fallback

        if len(df) > 100:
            st.error("The uploaded CSV file has more than 100 records. Please upload a CSV file with at most 100 records.")
            st.session_state['data'] = None
        else:
            st.success("CSV file uploaded successfully!")
            st.session_state['data'] = df
            st.subheader("Preview of Uploaded Data")
            st.dataframe(df)
    except Exception as e:
        st.error(f"Error reading CSV file: {e}")


# Analysis Menu
elif menu == "Analysis":
    st.header("🔍 Data Analysis")

    if st.session_state.data is None:
        st.warning("⚠️ Please upload a dataset first from the 'Upload Data' menu.")
    else:
        df = st.session_state.data

        st.subheader("Dataset")
        st.dataframe(df)

        st.subheader("Descriptive Statistics")
        st.write(f"**Mean:** {df['Sales'].mean():.2f}")
        st.write(f"**Median:** {df['Sales'].median():.2f}")
        st.write(f"**Standard Deviation:** {df['Sales'].std():.2f}")

        # Normality test
        stat, p = shapiro(df['Sales'])
        st.subheader("Normality Test (Shapiro-Wilk)")
        st.write(f"Statistic = {stat:.4f}, p-value = {p:.4f}")
        if p > 0.05:
            st.success("✅ Data appears to be normally distributed.")
        else:
            st.warning("⚠️ Data does not appear to be normally distributed.")

        # Rolling Mean plot
        st.subheader("Trend & Seasonality (Rolling Mean)")
        df['Rolling_Mean'] = df['Sales'].rolling(window=3).mean()

        fig, ax = plt.subplots()
        ax.plot(df['Week'], df['Sales'], label='Original')
        ax.plot(df['Week'], df['Rolling_Mean'], label='Rolling Mean (window=3)', color='orange')
        ax.set_xlabel("Week")
        ax.set_ylabel("Sales")
        ax.legend()
        st.pyplot(fig)

# Forecast Menu
elif menu == "Forecast":
    st.header("📈 Forecasting (ETS Model)")

    if st.session_state.data is None:
        st.warning("⚠️ Please upload a dataset first from the 'Upload Data' menu.")
    else:
        df = st.session_state.data

        try:
            model = ExponentialSmoothing(df['Sales'], trend='add', seasonal=None)
            fit = model.fit()

            forecast = fit.forecast(3)
            forecast_weeks = list(range(df['Week'].max() + 1, df['Week'].max() + 4))
            forecast_df = pd.DataFrame({'Week': forecast_weeks, 'Forecast_Sales': forecast})

            st.subheader("Forecasted Sales for Next 3 Weeks")
            st.dataframe(forecast_df)

            fig, ax = plt.subplots()
            ax.plot(df['Week'], df['Sales'], label='Historical Sales')
            ax.plot(forecast_weeks, forecast, label='Forecast', color='red', marker='o')
            ax.set_xlabel("Week")
            ax.set_ylabel("Sales")
            ax.legend()
            st.pyplot(fig)
        except Exception as e:
            st.error(f"❌ Error in forecasting: {e}")
