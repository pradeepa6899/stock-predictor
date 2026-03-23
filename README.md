# AI Stock & Crypto Price Trend Predictor

An AI-powered web application that predicts whether a stock or 
cryptocurrency price will go UP or DOWN using Machine Learning.

## What it does
- Fetches real-time stock and crypto data using yfinance
- Calculates technical indicators like RSI and Moving Average
- Trains a Random Forest AI model on historical price data
- Predicts tomorrow's price trend — UP or DOWN
- Displays everything in an interactive web app built with Streamlit

## Tech Stack
- Python
- yfinance — real stock and crypto data
- pandas — data processing
- scikit-learn — AI model (Random Forest Classifier)
- Streamlit — web app frontend
- Plotly — interactive charts

## How to Run
1. Install dependencies:
   pip install yfinance pandas scikit-learn streamlit plotly matplotlib

2. Run the app:
   streamlit run streamlit_app.py

3. Enter any ticker symbol (AAPL, TSLA, BTC-USD) and click Predict!

## Project Structure
stock-predictor/
├── app.py               # data fetch, indicators, AI model
├── streamlit_app.py     # web app frontend
└── README.md            # project description

## Supported Assets
- Stocks: AAPL, TSLA, NVDA, MSFT, GOOGL
- Crypto: BTC-USD, ETH-USD, SOL-USD

## Disclaimer
This app is for educational purposes only. Not financial advice.