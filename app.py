import os

import pandas as pd
import streamlit as st
import yfinance as yf
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score


# Function to load data
def load_data(ticker):
    try:
        stock_data = yf.Ticker(ticker)
        stock_history = stock_data.history(period="max")
        stock_history.to_csv("stock_data.csv")
        stock_history.index = pd.to_datetime(stock_history.index)
        stock_history["Tomorrow"] = stock_history["Close"].shift(-1)
        stock_history["Target"] = (stock_history["Tomorrow"] > stock_history["Close"]).astype(int)
        stock_history.index = pd.to_datetime(stock_history.index, utc=True)
        stock_history = stock_history.sort_index()
        stock_history = stock_history.loc["2000-01-01":].copy()
        return stock_history, True
    except Exception as e:
        return None, False

# Function to train model
def train_model(data, predictors):
    if data is None:  # Check if data is None
        return None  # Return None or some other default value
    model = RandomForestClassifier(n_estimators=100, min_samples_split=100, random_state=1)
    train = data.iloc[:-100]
    model.fit(train[predictors], train["Target"])
    return model


# Function to make predictions
def make_predictions(model, test, predictors):
    preds = model.predict(test[predictors])
    preds = pd.Series(preds, index=test.index, name="Predictions")
    combined = pd.concat([test["Target"], preds], axis=1)
    return combined

# Main function
def main():
    st.title('S&P 500 Prediction App')
    st.sidebar.title('Home')

    # Input ticker symbol
    ticker_input = st.sidebar.text_input('Enter Ticker Symbol', value='^GSPC')

    if not ticker_input.strip():
        ticker_input = '^GSPC'  # Set default ticker symbol to ^GSPC if input is empty

    # Load data
    data, is_valid_ticker = load_data(ticker_input)

    if is_valid_ticker:
        if data is not None:
            # Define predictors
            predictors = ["Close", "Volume", "Open", "High", "Low"]

            # Train model
            model = train_model(data, predictors)

            # Display some data
            st.subheader('Data')
            st.write(data.head())

            # Plot Close values
            st.subheader('Close Values')
            st.line_chart(data['Close'])

            # Make predictions
            predictions = make_predictions(model, data[-100:], predictors)

            # Display predictions
            st.subheader('Predictions')
            st.write(predictions)

            # Display future close value
            future_close = data.iloc[-1]["Close"]
            st.subheader('Future Close Value')
            st.write(future_close)
    else:
        st.subheader('Invalid Ticker')
        st.write(f"The entered ticker symbol '{ticker_input}' is invalid.")

if __name__ == "__main__":
    main()
