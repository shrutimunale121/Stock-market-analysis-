import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import pickle
from statsmodels.tsa.arima.model import ARIMA

# Streamlit app title
st.title("ARIMA Time Series Forecasting App")

# Upload dataset
uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])

if uploaded_file is not None:
    # Load the dataset
    data = pd.read_csv("C:\\Users\\91705\\data science\\DS19thJuly24-2-30-to-4-30-main_(1)[1]\\AAPL.csv")
    data['Date'] = pd.to_datetime(data['Date'])
    data.set_index('Date', inplace=True)
    
    # Display dataset preview
    st.write("Dataset Preview:")
    st.dataframe(data.head())

    # Select column for forecasting
    target_column = st.selectbox("Select the column to forecast", data.columns)
    series = data[target_column]

    # Plot the time series data
    st.subheader("Original Time Series")
    st.line_chart(series)

    # Load the trained ARIMA model
    if st.button("Load ARIMA Model"):
        try:
            with open('arima_model.pkl', 'rb') as f:
                loaded_model = pickle.load(f)
            st.success("Model loaded successfully!")
        except Exception as e:
            st.error(f"Error loading model: {e}")

    # Forecasting using the loaded model
    if st.button("Forecast Future Values"):
        try:
            with open('arima_model.pkl', 'rb') as f:
                loaded_model = pickle.load(f)
            st.success("Model loaded successfully!")

    
            steps = st.slider("Select Forecast Steps", min_value=1, max_value=1000, value=900)
            forecast = loaded_model.forecast(steps=steps)
            
            # Plot forecast
            st.subheader("Forecasted Values")
            plt.figure(figsize=(10, 6))
            plt.plot(series, label="Observed Data", color="blue")
            plt.plot(pd.date_range(series.index[-1], periods=steps + 1, freq='B')[1:], forecast, label="Forecast", color="red")
            plt.legend()
            plt.title("ARIMA Forecast")
            st.pyplot(plt)
        except Exception as e:
            st.error(f"Error in forecasting: {e}")