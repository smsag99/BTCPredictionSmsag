import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from sklearn.metrics import mean_absolute_error

## functions
def preprocessing(df):
    # df['Daily_growth_rate'] = df['Close'].pct_change() * 100
    # df['7_day_average'] = df['Volume'].rolling(window=7).mean()
    # df['30_day_average'] = df['Volume'].rolling(window=30).mean()
    # df['180_day_average'] = df['Volume'].rolling(window=180).mean()
    # df['Year'] = df.index.year
    # df['Month'] = df.index.month
    # df['Day'] = df.index.day
    # df['DayOfWeek'] = df.index.dayofweek
    # df['DayName'] = df.index.day_name()
    # df['IsWeekEnd'] = df.index.dayofweek > 4
    # df['Lag_1'] = df['Close'].shift(1)
    # df['Lag_2'] = df['Close'].shift(2)
    # df['Lag_3'] = df['Close'].shift(3)
    # df['Lag_4'] = df['Close'].shift(4)
    # df['Lag_5'] = df['Close'].shift(5)
    # df['Lag_6'] = df['Close'].shift(6)
    # df['Lag_7'] = df['Close'].shift(7)
    # df.fillna(method='bfill',inplace=True)

    return df

def forecast(df,lenght):
    portion = len(df)-lenght-1
    train,test = df.iloc[:portion,3],df.iloc[-lenght:,3]
    len(train),len(test)

    model_exponential_tripple = ExponentialSmoothing(df.Close,trend = 'add', seasonal = 'add', seasonal_periods = 12).fit(
        optimized=True, smoothing_level=0.2, smoothing_trend=0.1, smoothing_seasonal=0.1)
    pred_exponential_tripple = model_exponential_tripple.forecast(len(test))
    return pred_exponential_tripple



st.set_page_config(layout="wide")

st.title("📈 Bitcoin Price Forecast using Holt-Winters")

start_date = st.date_input("Start Date", pd.to_datetime("2024-01-01"))
end_date = st.date_input("End Date")

if start_date >= end_date:
    st.warning("⚠️ End date must be after start date.")
    st.stop()

# Load data
@st.cache_data
def load_data(start, end):
    df = yf.download("BTC-USD", start=start, end=end)
    df.columns = ['Close', 'High', 'Low', 'Open', 'Volume']
    df.index = pd.to_datetime(df.index)
    df.index.freq='D'
    return df

btc_data = load_data(start_date, end_date)
st.subheader("Raw Bitcoin Data")
st.line_chart(btc_data['Close'])

# Forecast section
if st.button("Run Forecast"):
    try:
        forecast = forecast(btc_data,21)

        fig, ax = plt.subplots(figsize=(12, 5))
        btc_data.Close[-150:].plot(label="Observed", ax=ax)
        forecast.plot(label="Forecast", ax=ax)
        plt.legend()
        st.pyplot(fig)
        df_combined = pd.DataFrame({
            'train': btc_data.Close[-150:],
            'prediction': forecast
        })

        # Streamlit chart
        st.line_chart(df_combined)
    except Exception as e:
        st.error(f"❌ Forecasting failed: {e}")
