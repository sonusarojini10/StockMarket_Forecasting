# src/app/app.py
"""
Streamlit realtime-ish forecasting dashboard (PyTorch LSTM)

- Loads trained PyTorch LSTM: models/<TICKER>_lstm.pth
- Downloads latest OHLC from Yahoo Finance
- Recomputes SAME 26 features used during LSTM training
- Uses last window_size timesteps to predict next day's Close
- Plots Actual + LSTM Forecast + Naive baseline
"""

import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

import torch
import torch.nn as nn

st.set_page_config(layout="wide", page_title="Realtime Stock Forecast Dashboard")


# ========================
# LSTM MODEL (same as training)
# ========================
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size=64, num_layers=2):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True
        )
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = out[:, -1, :]
        out = self.fc(out)
        return out


# ========================
# FEATURE FUNCTIONS (same as training)
# ========================
def SMA(series, period=20):
    return series.rolling(period).mean()

def EMA(series, period=20):
    return series.ewm(span=period, adjust=False).mean()

def RSI(series, period=14):
    delta = series.diff()
    up = delta.clip(lower=0)
    down = -delta.clip(upper=0)
    ma_up = up.rolling(period).mean()
    ma_down = down.rolling(period).mean()
    rs = ma_up / ma_down
    return 100 - (100 / (1 + rs))

def MACD(series):
    ema12 = series.ewm(span=12, adjust=False).mean()
    ema26 = series.ewm(span=26, adjust=False).mean()
    macd = ema12 - ema26
    signal = macd.ewm(span=9, adjust=False).mean()
    hist = macd - signal
    return macd, signal, hist

def bollinger(series, period=20):
    sma = series.rolling(period).mean()
    std = series.rolling(period).std()
    upper = sma + 2 * std
    lower = sma - 2 * std
    return sma, upper, lower


def create_features_from_ohlc(df):
    df = df.copy()

    df["SMA_20"] = SMA(df["Close"], 20)
    df["EMA_20"] = EMA(df["Close"], 20)
    df["RSI_14"] = RSI(df["Close"], 14)

    macd, macd_signal, macd_hist = MACD(df["Close"])
    df["MACD"] = macd
    df["MACD_signal"] = macd_signal
    df["MACD_hist"] = macd_hist

    df["BB_mid"], df["BB_upper"], df["BB_lower"] = bollinger(df["Close"])

    df["Return"] = df["Close"].pct_change()
    df["LogReturn"] = np.log(df["Close"] / df["Close"].shift(1))
    df["Volatility_20"] = df["LogReturn"].rolling(20).std()

    for lag in range(1, 11):
        df[f"Close_lag_{lag}"] = df["Close"].shift(lag)

    df = df.dropna()
    return df


# ========================
# LOAD TRAINED MODEL
# ========================
@st.cache_resource(ttl=3600)
def load_model(ticker, input_size, hidden_size=64, num_layers=2):
    model_path = os.path.join("models", f"{ticker}_lstm.pth")
    if not os.path.exists(model_path):
        return None

    device = torch.device("cpu")
    model = LSTMModel(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    return model


# ========================
# MAKE PREDICTION
# ========================
def predict(model, X_window, scaler_y):
    """
    X_window: numpy array shape (1, window, features)
    scaler_y: fitted MinMaxScaler for y (expects 2D)
    """
    with torch.no_grad():
        inp = torch.tensor(X_window, dtype=torch.float32)
        out_tensor = model(inp)  # tensor shape (1,1)
        out_np = out_tensor.cpu().numpy()  # ensure on cpu and numpy array

    # inverse transform (scaler expects 2D array)
    if scaler_y is not None:
        pred = scaler_y.inverse_transform(out_np)
    else:
        pred = out_np

    return float(pred.flatten()[0])


# ========================
# STREAMLIT UI
# ========================
st.title("📈 Live Stock Forecast Dashboard (LSTM – PyTorch)")

col1, col2 = st.columns([1, 3])

with col1:
    st.subheader("Controls")
    ticker = st.selectbox("Ticker", ("AAPL", "MSFT", "TSLA", "^GSPC"))
    interval = st.selectbox("Interval", ("1d", "1m"))
    period = st.selectbox("History", ("3mo", "6mo", "1y", "2y", "5y", "10y"), index=2)
    window_size = st.slider("Window Size", 20, 120, 60, 5)
    btn = st.button("Refresh / Predict")

with col2:
    st.subheader("Live Chart")
    chart_area = st.empty()


# ========================
# MAIN LOGIC
# ========================
if btn:
    status = st.empty()
    status.info("Fetching data...")

    try:
        df = yf.download(ticker, period=period, interval=interval, progress=False)
    except Exception as e:
        status.error(f"Failed to fetch data: {e}")
        st.stop()

    if df is None or df.empty:
        status.error("No data fetched")
        st.stop()

    # 🔥 FIX FOR YFINANCE MULTI-INDEX (important)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    # Clean column names (remove spaces)
    df = df.rename(columns=lambda x: x.strip())

    # Ensure required columns exist
    required_cols = ["Open", "High", "Low", "Close", "Volume"]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        status.error(f"Missing required columns: {missing}")
        st.stop()

    status.info("Computing features...")
    feat_df = create_features_from_ohlc(df[["Open", "High", "Low", "Close", "Volume"]])

    # ===== FIXED FEATURE LIST (MATCHES LSTM TRAINING EXACTLY) =====
    FEATURE_LIST = (
        ["Open", "High", "Low", "Volume"] +  # raw inputs used during training
        ["SMA_20", "EMA_20", "RSI_14",
         "MACD", "MACD_signal", "MACD_hist",
         "BB_mid", "BB_upper", "BB_lower",
         "Return", "LogReturn", "Volatility_20"] +
        [f"Close_lag_{i}" for i in range(1, 11)]
    )
    # =============================================================

    # ensure feature columns exist
    missing = [c for c in FEATURE_LIST if c not in feat_df.columns]
    if missing:
        status.error(f"Missing features after computation: {missing}")
        st.stop()

    X = feat_df[FEATURE_LIST].values
    y = feat_df["Close"].values.reshape(-1, 1)

    scaler_x = MinMaxScaler()
    scaler_y = MinMaxScaler()

    X_scaled = scaler_x.fit_transform(X.astype(float))
    y_scaled = scaler_y.fit_transform(y.astype(float))

    if len(X_scaled) < window_size:
        status.error("Not enough data after feature creation.")
        st.stop()

    X_window = np.expand_dims(X_scaled[-window_size:], axis=0)

    status.info("Loading model…")
    model = load_model(ticker, input_size=X_window.shape[2])

    naive_pred = float(y[-1])  # baseline

    if model is None:
        status.warning("No model found. Showing naive forecast only.")
        pred = naive_pred
    else:
        status.info("Running prediction…")
        try:
            pred = predict(model, X_window, scaler_y)
        except Exception as e:
            status.error(f"Prediction failed: {e}")
            st.stop()

    # ===== PLOT =====
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(feat_df.index[-200:], feat_df["Close"].iloc[-200:], label="Actual")
    future_time = feat_df.index[-1] + pd.Timedelta(days=1)
    ax.scatter([future_time], [pred], color="orange", s=80, label="LSTM Forecast")
    ax.scatter([future_time], [naive_pred], color="red", s=40, label="Naive")
    ax.set_xlabel("Date")        # ← ADD THIS
    ax.set_ylabel("Close Price") # ← ADD THIS


    ax.legend()
    ax.set_title(f"{ticker} – Next-Day Forecast")
    chart_area.pyplot(fig)

    status.success(f"Predicted Next Close: **{pred:.4f}** (Naive: {naive_pred:.4f})")
