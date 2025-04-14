import streamlit as st
import pandas as pd
import numpy as np
from scipy.stats import shapiro
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.holtwinters import ExponentialSmoothing
import warnings

warnings.filterwarnings("ignore")

# Page config
st.set_page_config(layout="wide", page_title="📈 Sales Forecast App")

# Session state init
if "data" not in st.session_state:
    st.session_state["data"] = None

# Sidebar menu
st.sidebar.title("📊 Menu")
menu = st.sidebar.radio("Select", ["Forecast"])

# Main content
if menu == "Forecast":
    st.title("📦 Upload Data & Forecast Sales")

    uploaded_file = st.file_uploader("Upload CSV with 'Week' and 'Sales' columns (max 100 rows)", type=["csv"])
    if uploaded_file is not None:
        try:
            # Try different encodings
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
                st.error("❌ Could not read file. Ensure valid CSV format.")
            elif df.shape[0] > 100:
                st.error("❌ Maximum 100 records allowed. Your file has more than 100 rows.")
            elif "Week" not in df.columns or "Sales" not in df.columns:
                st.error("❌ CSV must contain 'Week' and 'Sales' columns.")
            else:
                # Clean and store
                df = df[["Week", "Sales"]].copy()
                df["Week"] = pd.to_numeric(df["Week"], errors="coerce")
                df["Sales"] = pd.to_numeric(df["Sales"], errors="coerce")
                df.dropna(inplace=True)
                df = df.sort_values("Week").reset_index(drop=True)
                st.session_state["data"] = df
                st.success("✅ Data uploaded successfully!")

        except Exception as e:
            st.error(f"❌ Error reading file: {e}")

    # Show analysis + forecast if data is available
    if st.session_state["data"] is not None:
        df = st.session_state["data"]

        st.subheader("📊 Data Preview")
        st.dataframe(df)

        st.subheader("🧪 Data Analysis")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Mean", round(df["Sales"].mean(), 2))
        with col2:
            st.metric("Median", round(df["Sales"].median(), 2))
        with col3:
            st.metric("Standard Deviation", round(df["Sales"].std(), 2))

        # Normality test
        stat, p = shapiro(df["Sales"])
        st.write("**Normality Test (Shapiro-Wilk)**")
        st.write(f"Statistic: {stat:.4f}, p-value: {p:.4f}")
        if p > 0.05:
            st.success("✅ Data looks normally distributed.")
        else:
            st.warning("⚠️ Data may not be normally distributed.")

        # Seasonality check
        st.write("**Seasonality Check**")
        try:
            result = seasonal_decompose(df["Sales"], model="additive", period=3)
            fig, ax = plt.subplots(4, 1, figsize=(10, 8))
            result.observed.plot(ax=ax[0], title="Observed")
            result.trend.plot(ax=ax[1], title="Trend")
            result.seasonal.plot(ax=ax[2], title="Seasonal")
            result.resid.plot(ax=ax[3], title="Residual")
            st.pyplot(fig)
        except Exception as e:
            st.warning(f"⚠️ Could not perform seasonality check: {e}")

        # Forecast section
        st.subheader("📈 Forecast (Next 3 Weeks)")
        try:
            model = ExponentialSmoothing(df["Sales"], trend="add", seasonal="add", seasonal_periods=3).fit()
            forecast = model.forecast(3)
            forecast_df = pd.DataFrame({
                "Week": range(int(df["Week"].max()) + 1, int(df["Week"].max()) + 4),
                "Forecasted_Sales": forecast
            })
            st.dataframe(forecast_df)

            # Plot
            fig2, ax2 = plt.subplots(figsize=(10, 4))
            ax2.plot(df["Week"], df["Sales"], label="Historical", marker="o")
            ax2.plot(forecast_df["Week"], forecast_df["Forecasted_Sales"], label="Forecast", marker="o", linestyle="--")
            ax2.set_xlabel("Week")
            ax2.set_ylabel("Sales")
            ax2.set_title("Sales Forecast")
            ax2.legend()
            st.pyplot(fig2)
        except Exception as e:
            st.error(f"❌ Forecasting failed: {e}")
