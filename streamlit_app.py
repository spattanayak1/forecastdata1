import warnings  # ← Add this line at the very top
import streamlit as st
import pandas as pd
import numpy as np
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.arima.model import ARIMA
from prophet import Prophet
from datetime import timedelta
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings("ignore")
# SET PAGE CONFIG FIRST
st.set_page_config(layout="wide", page_title="📈 Sales Forecast App")

# Now you can hide the GitHub icon and other elements
hide_streamlit_style = """
    <style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    .stDeployButton {display:none;}
    .st-emotion-cache-6qob1r {display: none;} /* GitHub icon */
    </style>
"""
st.markdown(hide_streamlit_style, unsafe_allow_html=True)





# Session init
if "data" not in st.session_state:
    st.session_state["data"] = None

st.sidebar.title("📊 Menu")
menu = st.sidebar.radio("Select", ["Forecast"])

if menu == "Forecast":
    st.title("📦 Upload CSV & Generate Forecasts")

    uploaded_file = st.file_uploader("Upload CSV with 'Week' and 'Sales' (max 100 rows)", type=["csv"])
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            if df.shape[0] > 100:
                st.error("❌ Maximum 100 rows allowed.")
            elif 'Week' not in df.columns or 'Sales' not in df.columns:
                st.error("❌ CSV must contain 'Week' and 'Sales' columns.")
            else:
                df = df[['Week', 'Sales']].copy()
                df['Week'] = pd.to_numeric(df['Week'], errors='coerce')
                df['Sales'] = pd.to_numeric(df['Sales'], errors='coerce')
                df.dropna(inplace=True)
                df.sort_values('Week', inplace=True)
                df.reset_index(drop=True, inplace=True)
                st.session_state["data"] = df
                st.success("✅ Data uploaded successfully.")
        except Exception as e:
            st.error(f"❌ Error: {e}")

    if st.session_state["data"] is not None:
        df = st.session_state["data"]

        # --- Data Analysis ---
        st.subheader("🧪 Data Analysis")
        st.dataframe(df)
        col1, col2, col3 = st.columns(3)
        col1.metric("Mean", round(df["Sales"].mean(), 2))
        col2.metric("Median", round(df["Sales"].median(), 2))
        col3.metric("Std Dev", round(df["Sales"].std(), 2))

        stat, p = shapiro(df["Sales"])
        st.write(f"**Shapiro-Wilk Test**: Statistic={stat:.4f}, p-value={p:.4f}")
        st.success("✅ Likely Normal") if p > 0.05 else st.warning("⚠️ Possibly Not Normal")

        try:
            result = seasonal_decompose(df["Sales"], model="additive", period=3)
            fig, ax = plt.subplots(4, 1, figsize=(10, 8))
            result.observed.plot(ax=ax[0], title="Observed")
            result.trend.plot(ax=ax[1], title="Trend")
            result.seasonal.plot(ax=ax[2], title="Seasonal")
            result.resid.plot(ax=ax[3], title="Residual")
            st.pyplot(fig)
        except Exception as e:
            st.warning("⚠️ Seasonality check skipped: " + str(e))

        # Forecast Horizon
        forecast_weeks = 3
        last_week = df["Week"].max()
        new_weeks = list(range(last_week + 1, last_week + forecast_weeks + 1))

        # --- ETS Forecast ---
        st.subheader("📈 ETS Forecast")
        ets_model = ExponentialSmoothing(df["Sales"], trend="add", seasonal="add", seasonal_periods=3).fit()
        ets_forecast = ets_model.forecast(forecast_weeks)

        fig1, ax1 = plt.subplots()
        ax1.plot(df["Week"], df["Sales"], label="Actual")
        ax1.plot(new_weeks, ets_forecast, label="ETS Forecast", linestyle="--", marker="o")
        ax1.legend()
        ax1.set_title("ETS Forecast")
        st.pyplot(fig1)

        # --- ARIMA Forecast ---
        st.subheader("📈 ARIMA Forecast")
        arima_model = ARIMA(df["Sales"], order=(1, 1, 1)).fit()
        arima_forecast = arima_model.forecast(forecast_weeks)

        fig2, ax2 = plt.subplots()
        ax2.plot(df["Week"], df["Sales"], label="Actual")
        ax2.plot(new_weeks, arima_forecast, label="ARIMA Forecast", linestyle="--", marker="o")
        ax2.legend()
        ax2.set_title("ARIMA Forecast")
        st.pyplot(fig2)

        # --- Prophet Forecast ---
        st.subheader("📈 Prophet Forecast")
        prophet_df = df.rename(columns={"Week": "ds", "Sales": "y"})
        prophet_df["ds"] = pd.date_range(start="2023-01-01", periods=len(df), freq="W")

        future_dates = pd.date_range(start=prophet_df["ds"].max() + pd.Timedelta(weeks=1), periods=forecast_weeks, freq="W")

        prophet = Prophet()
        prophet.fit(prophet_df)
        future = prophet.make_future_dataframe(periods=forecast_weeks, freq='W')
        forecast = prophet.predict(future)

        prophet_forecast = forecast[['ds', 'yhat']].tail(forecast_weeks).copy()
        prophet_forecast["Week"] = new_weeks
        prophet_forecast = prophet_forecast[["Week", "yhat"]].rename(columns={"yhat": "Prophet_Forecast"})

        fig3 = prophet.plot(forecast)
        st.pyplot(fig3)

        # --- Final Combined Output ---
        st.subheader("📊 Combined Forecast Table")

        final_df = pd.DataFrame({
            "Week": new_weeks,
            "ETS_Forecast": ets_forecast.values,
            "ARIMA_Forecast": arima_forecast.values,
            "Prophet_Forecast": prophet_forecast["Prophet_Forecast"].values
        })

        st.dataframe(final_df)

        # Combine with original for graph
        combined_plot_df = df.copy()
        combined_plot_df["Source"] = "Actual"
        forecast_dfs = [
            pd.DataFrame({"Week": new_weeks, "Sales": ets_forecast, "Source": "ETS"}),
            pd.DataFrame({"Week": new_weeks, "Sales": arima_forecast, "Source": "ARIMA"}),
            pd.DataFrame({"Week": new_weeks, "Sales": prophet_forecast["Prophet_Forecast"], "Source": "Prophet"})
        ]
        full_plot_df = pd.concat([combined_plot_df] + forecast_dfs)

        # Plot
        st.subheader("📉 Combined Forecast Plot")
        fig_all, ax_all = plt.subplots(figsize=(10, 5))
        for label, group in full_plot_df.groupby("Source"):
            ax_all.plot(group["Week"], group["Sales"], label=label, marker="o", linestyle="--" if label != "Actual" else "-")
        ax_all.set_xlabel("Week")
        ax_all.set_ylabel("Sales")
        ax_all.set_title("Actual vs Forecast (ETS, ARIMA, Prophet)")
        ax_all.legend()
        st.pyplot(fig_all)
