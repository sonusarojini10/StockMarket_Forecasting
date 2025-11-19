"""
baseline.py
Baseline forecasting models for stock prediction.

Models included:
    - Naive Model (tomorrow = today)
    - Rolling Mean Model (last N days)
    - Drift / Trend Model

Evaluates:
    - RMSE, MAE, MAPE
    - Direction Accuracy

Outputs:
    - Plots in reports/figures/
    - Printed metrics
"""

import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error


DATA_PROCESSED_PATH = "data/processed"
FIGURES_PATH = "reports/figures"

os.makedirs(FIGURES_PATH, exist_ok=True)


# --------------------------
# Baseline Models
# --------------------------

def naive_forecast(df):
    """Predict next close = today's close."""
    df["Pred_naive"] = df["Close"].shift(1)
    return df


def rolling_mean_forecast(df, window=5):
    """Predict next close = mean of last N closes."""
    df["Pred_roll"] = df["Close"].rolling(window).mean().shift(1)
    return df


def drift_forecast(df):
    """
    Drift method: predict future based on linear trend.
    Formula: F(t+1) = y(t) + (y(t) - y(1)) / (t - 1)
    """
    df = df.copy()
    df["drift"] = (df["Close"] - df["Close"].iloc[0]) / (np.arange(len(df)))
    df["Pred_drift"] = df["Close"] + df["drift"].shift(1)
    return df


# --------------------------
# Evaluation Metrics
# --------------------------

def direction_accuracy(actual, predicted):
    return np.mean((np.sign(actual.diff()) == np.sign(predicted.diff()))) * 100


def evaluate_model(df, pred_col):
    df = df.dropna(subset=[pred_col])
    actual = df["Close"]
    predicted = df[pred_col]

    rmse = np.sqrt(mean_squared_error(actual, predicted))
    mae = mean_absolute_error(actual, predicted)
    mape = np.mean(np.abs((actual - predicted) / actual)) * 100

    dir_acc = direction_accuracy(actual, predicted)

    return rmse, mae, mape, dir_acc


# --------------------------
# Plot Function
# --------------------------

def plot_predictions(df, ticker, col):
    plt.figure(figsize=(12, 6))
    plt.plot(df.index, df["Close"], label="Actual", linewidth=2)
    plt.plot(df.index, df[col], label=col, alpha=0.7)
    plt.title(f"{ticker} - Actual vs {col}")
    plt.xlabel("Date")
    plt.ylabel("Price")
    plt.legend()

    save_path = os.path.join(FIGURES_PATH, f"{ticker}_{col}.png")
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"[INFO] Plot saved: {save_path}")


# --------------------------
# Main Runner
# --------------------------

def run_baseline(ticker):
    file_path = os.path.join(DATA_PROCESSED_PATH, f"{ticker}_features.csv")
    
    if not os.path.exists(file_path):
        print(f"[ERROR] Processed file not found for ticker: {ticker}")
        return

    df = pd.read_csv(file_path, parse_dates=["Date"], index_col="Date")

    print(f"\n=== BASELINE FORECASTS FOR {ticker} ===")

    # Naive Model
    df = naive_forecast(df)
    rmse, mae, mape, dir_acc = evaluate_model(df, "Pred_naive")
    print(f"\n[Naive Model]")
    print(f"RMSE: {rmse:.4f}, MAE: {mae:.4f}, MAPE: {mape:.2f}%, Direction Accuracy: {dir_acc:.2f}%")
    plot_predictions(df, ticker, "Pred_naive")

    # Rolling Mean
    df = rolling_mean_forecast(df)
    rmse, mae, mape, dir_acc = evaluate_model(df, "Pred_roll")
    print(f"\n[Rolling Mean Model]")
    print(f"RMSE: {rmse:.4f}, MAE: {mae:.4f}, MAPE: {mape:.2f}%, Direction Accuracy: {dir_acc:.2f}%")
    plot_predictions(df, ticker, "Pred_roll")

    # Drift Model
    df = drift_forecast(df)
    rmse, mae, mape, dir_acc = evaluate_model(df, "Pred_drift")
    print(f"\n[Drift Model]")
    print(f"RMSE: {rmse:.4f}, MAE: {mae:.4f}, MAPE: {mape:.2f}%, Direction Accuracy: {dir_acc:.2f}%")
    plot_predictions(df, ticker, "Pred_drift")

    print(f"\n[INFO] Baseline models completed for {ticker}.")


if __name__ == "__main__":
    # Run baselines for key tickers
    tickers = ["AAPL", "MSFT", "TSLA", "^GSPC"]
    for t in tickers:
        run_baseline(t)
