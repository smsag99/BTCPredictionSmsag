import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from sklearn.metrics import mean_absolute_error

st.set_page_config(layout="wide")

st.title("📈 Bitcoin Price Forecast using Holt-Winters")

start_date = st.date_input("Start Date", pd.to_datetime("2014-01-01"))
end_date = st.date_input("End Date")

if start_date >= end_date:
    st.warning("⚠️ End date must be after start date.")
    st.stop()

# Load data
@st.cache_data
def load_data(start, end):
    df = yf.download("BTC-USD", start=start, end=end)
    return df

btc_data = load_data(start_date, end_date)
st.subheader("Raw Bitcoin Data")
st.line_chart(btc_data['Close'])

# Forecast section
if st.button("Run Forecast"):
    try:
        ts = btc_data['Close'].dropna()
        model = ExponentialSmoothing(ts, trend='add', seasonal='add', seasonal_periods=365)
        fit = model.fit()
        forecast = fit.forecast(30)

        fig, ax = plt.subplots(figsize=(12, 5))
        ts.plot(label="Observed", ax=ax)
        forecast.plot(label="Forecast", ax=ax)
        plt.legend()
        st.pyplot(fig)
    except Exception as e:
        st.error(f"❌ Forecasting failed: {e}")
