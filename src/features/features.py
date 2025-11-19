"""
features.py
Feature engineering for stock forecasting.

Reads raw CSVs from data/raw/
Generates:
    - Technical indicators (SMA, EMA, RSI, MACD, BBands)
    - Returns & log returns
    - Rolling Volatility
    - Lag features
Saves processed CSVs to data/processed/<TICKER>_features.csv
"""

import pandas as pd
import numpy as np
import os
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

DATA_RAW_PATH = "data/raw"
DATA_PROCESSED_PATH = "data/processed"

# -----------------------------
# Technical Indicators
# -----------------------------
def SMA(df, period=20):
    return df["Close"].rolling(period).mean()

def EMA(df, period=20):
    return df["Close"].ewm(span=period, adjust=False).mean()

def RSI(df, period=14):
    delta = df["Close"].diff()
    up = delta.clip(lower=0)
    down = -delta.clip(upper=0)
    ma_up = up.rolling(period).mean()
    ma_down = down.rolling(period).mean()
    rs = ma_up / ma_down
    return 100 - (100 / (1 + rs))

def MACD(df):
    ema12 = df["Close"].ewm(span=12, adjust=False).mean()
    ema26 = df["Close"].ewm(span=26, adjust=False).mean()
    macd = ema12 - ema26
    signal = macd.ewm(span=9, adjust=False).mean()
    hist = macd - signal
    return macd, signal, hist

def Bollinger_Bands(df, period=20):
    sma = df["Close"].rolling(period).mean()
    std = df["Close"].rolling(period).std()
    upper = sma + (2 * std)
    lower = sma - (2 * std)
    return sma, upper, lower


# -----------------------------
# Add All Features
# -----------------------------
def add_features(df):
    df["SMA_20"] = SMA(df)
    df["EMA_20"] = EMA(df)

    df["RSI_14"] = RSI(df)

    df["MACD"], df["MACD_signal"], df["MACD_hist"] = MACD(df)

    df["BB_mid"], df["BB_upper"], df["BB_lower"] = Bollinger_Bands(df)

    df["Return"] = df["Close"].pct_change()
    df["LogReturn"] = np.log(df["Close"] / df["Close"].shift(1))

    df["Volatility_20"] = df["LogReturn"].rolling(20).std()

    # Lag features
    for lag in range(1, 11):
        df[f"Close_lag_{lag}"] = df["Close"].shift(lag)

    df.dropna(inplace=True)
    return df


# -----------------------------
# PROCESS SINGLE FILE
# -----------------------------
def process_file(file_path):
    ticker = os.path.basename(file_path).replace(".csv", "")
    logging.info(f"Processing {ticker}...")

    # Skip the first 3 rows: Price row, Ticker row, Date row
    df = pd.read_csv(file_path, skiprows=3, header=None)

    # Assign correct column names
    df.columns = ["Date", "Close", "High", "Low", "Open", "Volume"]

    # Strip whitespace and BOM if present
    df["Date"] = df["Date"].astype(str).str.strip()
    if df["Date"].iloc[0].startswith("\ufeff"):
        df["Date"] = df["Date"].str.lstrip("\ufeff")

    # Parse Date
    try:
        df["Date"] = pd.to_datetime(df["Date"], format="%Y-%m-%d")
    except:
        logging.warning("Flexible date parsing due to format issue.")
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce")

    # Check for parse failures
    if df["Date"].isna().any():
        logging.error("Date parsing failed! Debug rows:")
        print(df[df["Date"].isna()].head())
        raise ValueError("Date parsing failed.")

    # Set index
    df.set_index("Date", inplace=True)

    # Ensure numeric columns
    numeric_cols = ["Close", "High", "Low", "Open", "Volume"]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df.dropna(inplace=True)

    # Add all engineered features
    df = add_features(df)

    # Save processed CSV
    os.makedirs(DATA_PROCESSED_PATH, exist_ok=True)
    output_path = os.path.join(DATA_PROCESSED_PATH, f"{ticker}_features.csv")
    df.to_csv(output_path)

    logging.info(f"Saved processed file: {output_path}")
    logging.info(f"Preview for {ticker}:\n{df.head(3)}\n")


# -----------------------------
# RUN FEATURE GENERATION
# -----------------------------
def run_feature_engineering():
    files = [f for f in os.listdir(DATA_RAW_PATH) if f.endswith(".csv")]
    for file in files:
        process_file(os.path.join(DATA_RAW_PATH, file))

if __name__ == "__main__":
    run_feature_engineering()
