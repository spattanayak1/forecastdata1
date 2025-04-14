import streamlit as st
import pandas as pd
import numpy as np
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from scipy.stats import shapiro
import matplotlib.pyplot as plt

import streamlit as st
import pandas as pd

st.set_page_config(layout="wide", page_title="Sales Forecast App")

# Ensure session state initialized
if "data" not in st.session_state:
    st.session_state["data"] = None

menu = st.sidebar.radio("Navigation", ["Upload Data", "Analysis", "Forecast"])

if menu == "Upload Data":
    st.header("📁 Upload Your CSV File")

    uploaded_file = st.file_uploader("Upload CSV", type=["csv"])
    if uploaded_file is not None:
        try:
            encodings_to_try = ["utf-8", "ISO-8859-1", "utf-16"]
            read_success = False

            for enc in encodings_to_try:
                try:
                    df = pd.read_csv(uploaded_file, encoding=enc)
                    if not df.empty and df.shape[1] > 0:
                        read_success = True
                        break
                except Exception:
                    continue

            if not read_success:
                st.error("❌ Could not read file. Make sure it's a CSV with proper encoding and column structure.")
            elif df.shape[0] > 100:
                st.error("❌ Maximum 100 records allowed. Your file has more than 100 rows.")
            elif 'Week' not in df.columns or 'Sales' not in df.columns:
                st.error("❌ CSV must contain 'Week' and 'Sales' columns.")
            else:
                df = df[['Week', 'Sales']].copy()
                df['Week'] = pd.to_numeric(df['Week'], errors='coerce')
                df['Sales'] = pd.to_numeric(df['Sales'], errors='coerce')
                df.dropna(inplace=True)

                st.session_state["data"] = df
                st.success("✅ File uploaded and data saved.")
                st.dataframe(df)
        except Exception as e:
            st.error(f"❌ Unexpected error: {e}")



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
