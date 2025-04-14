import streamlit as st
import pandas as pd
import numpy as np
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from scipy.stats import shapiro
import matplotlib.pyplot as plt

# Load data
data = {
    'Week': list(range(1, 19)),
    'Sales': [4275, 3357, 2019, 1315, 1805, 2034, 4568, 1340, 3141, 3984,
              1942, 1696, 3581, 2107, 3084, 1734, 1310, 3359]
}
df = pd.DataFrame(data)

# Title
st.title("📊 Sales Analysis & Forecasting App")

# Sidebar menu using radio buttons
menu = st.sidebar.radio("Navigation", ["Analysis", "Forecast"])

if menu == "Analysis":
    st.header("🔍 Data Analysis")

    st.subheader("Dataset")
    st.dataframe(df)

    st.subheader("Descriptive Statistics")
    st.write(f"**Mean:** {df['Sales'].mean():.2f}")
    st.write(f"**Median:** {df['Sales'].median():.2f}")
    st.write(f"**Standard Deviation:** {df['Sales'].std():.2f}")

    # Normality check (Shapiro-Wilk Test)
    stat, p = shapiro(df['Sales'])
    st.subheader("Normality Test (Shapiro-Wilk)")
    st.write(f"Statistic = {stat:.4f}, p-value = {p:.4f}")
    if p > 0.05:
        st.success("✅ Data appears to be normally distributed.")
    else:
        st.warning("⚠️ Data does not appear to be normally distributed.")

    # Rolling mean plot
    st.subheader("Trend & Seasonality (Rolling Mean)")
    df['Rolling_Mean'] = df['Sales'].rolling(window=3).mean()

    fig, ax = plt.subplots()
    ax.plot(df['Week'], df['Sales'], label='Original')
    ax.plot(df['Week'], df['Rolling_Mean'], label='Rolling Mean (window=3)', color='orange')
    ax.set_xlabel("Week")
    ax.set_ylabel("Sales")
    ax.legend()
    st.pyplot(fig)

elif menu == "Forecast":
    st.header("📈 Forecasting (ETS Model)")
    
    model = ExponentialSmoothing(df['Sales'], trend='add', seasonal=None)
    fit = model.fit()

    forecast = fit.forecast(3)
    forecast_weeks = list(range(df['Week'].max() + 1, df['Week'].max() + 4))
    forecast_df = pd.DataFrame({'Week': forecast_weeks, 'Forecast_Sales': forecast})

    st.subheader("Forecasted Sales for Next 3 Weeks")
    st.dataframe(forecast_df)

    # Plot
    fig, ax = plt.subplots()
    ax.plot(df['Week'], df['Sales'], label='Historical Sales')
    ax.plot(forecast_weeks, forecast, label='Forecast', color='red', marker='o')
    ax.set_xlabel("Week")
    ax.set_ylabel("Sales")
    ax.legend()
    st.pyplot(fig)
