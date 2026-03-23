import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

st.set_page_config(page_title="AI Stock Predictor", page_icon="📈", layout="wide")

st.title("AI Stock & Crypto Price Trend Predictor")
st.caption("Built with Python, yfinance, scikit-learn and Streamlit")

# Sidebar inputs
st.sidebar.header("Choose Your Asset")
ticker = st.sidebar.text_input("Enter Ticker Symbol", value="AAPL")
period = st.sidebar.selectbox("Time Period", ["6mo", "1y", "2y"])
st.sidebar.markdown("**Examples:** AAPL, TSLA, BTC-USD, ETH-USD")

if st.sidebar.button("Predict"):

    with st.spinner("Fetching data and training AI..."):

        # Get data
        df = yf.download(ticker, period=period)
        df.columns = df.columns.get_level_values(0)

        # Indicators
        df["MA20"] = df["Close"].rolling(20).mean()
        df["MA50"] = df["Close"].rolling(50).mean()
        delta = df["Close"].diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        df["RSI"] = 100 - (100 / (1 + gain.rolling(14).mean() / loss.rolling(14).mean()))
        df["Target"] = (df["Close"].shift(-1) > df["Close"]).astype(int)
        df.dropna(inplace=True)

        # Train model
        features = ["Close", "MA20", "MA50", "RSI"]
        X = df[features]
        y = df["Target"]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        acc = accuracy_score(y_test, model.predict(X_test))

        # Prediction
        prediction = model.predict(df[features].tail(1))[0]
        current_price = round(float(df["Close"].iloc[-1]), 2)
        rsi_val = round(float(df["RSI"].iloc[-1]), 2)

    # Display results
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Current Price", f"${current_price}")
    col2.metric("Prediction", "UP" if prediction == 1 else "DOWN")
    col3.metric("Model Accuracy", f"{round(acc * 100, 2)}%")
    col4.metric("RSI", rsi_val)

    # Chart
    st.subheader("Price Chart with Indicators")
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df.index, y=df["Close"], name="Close", line=dict(color="#534AB7")))
    fig.add_trace(go.Scatter(x=df.index, y=df["MA20"], name="MA20", line=dict(color="#1D9E75", dash="dash")))
    fig.add_trace(go.Scatter(x=df.index, y=df["MA50"], name="MA50", line=dict(color="#D85A30", dash="dot")))
    fig.update_layout(xaxis_title="Date", yaxis_title="Price (USD)", height=400)
    st.plotly_chart(fig, use_container_width=True)

    # RSI Chart
    st.subheader("RSI Indicator")
    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(x=df.index, y=df["RSI"], name="RSI", line=dict(color="#D85A30")))
    fig2.add_hline(y=70, line_dash="dash", line_color="red", annotation_text="Overbought")
    fig2.add_hline(y=30, line_dash="dash", line_color="green", annotation_text="Oversold")
    fig2.update_layout(height=300)
    st.plotly_chart(fig2, use_container_width=True)

    st.caption("This app is for educational purposes only. Not financial advice.")