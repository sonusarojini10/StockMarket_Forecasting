"""
arima_model.py
ARIMA & Auto-ARIMA forecasting for stock prices.

Steps:
    - Load processed features
    - Test stationarity
    - Auto-select ARIMA(p,d,q)
    - Fit ARIMA model
    - 1-step ahead rolling forecast
    - Evaluate & plot
"""

import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from pmdarima.arima import auto_arima
from sklearn.metrics import mean_absolute_error, mean_squared_error

DATA_PROCESSED_PATH = "data/processed"
FIGURES_PATH = "reports/figures"

os.makedirs(FIGURES_PATH, exist_ok=True)

# --------------------------
# Direction Accuracy
# --------------------------
def direction_accuracy(actual, predicted):
    return np.mean(np.sign(actual.diff()) == np.sign(predicted.diff())) * 100


# --------------------------
# ARIMA Modeling Function
# --------------------------
def run_arima(ticker):
    print(f"\n=== ARIMA FORECAST FOR {ticker} ===")

    file_path = os.path.join(DATA_PROCESSED_PATH, f"{ticker}_features.csv")
    if not os.path.exists(file_path):
        print(f"[ERROR] Processed file not found for {ticker}. Run feature engineering first.")
        return

    # Load Close price only (ARIMA works on single time series)
    df = pd.read_csv(file_path, parse_dates=["Date"], index_col="Date")
    close = df["Close"]

    # --------------------------
    # Auto ARIMA to select parameters
    # --------------------------
    print("[INFO] Running auto_arima... this may take a moment.")
    model = auto_arima(
        close,
        seasonal=False,
        trace=False,
        suppress_warnings=True,
        error_action="ignore",
        stepwise=True,
        max_p=5,
        max_q=5,
        max_d=2
    )

    print(f"[INFO] Best ARIMA order selected: {model.order}")

    # --------------------------
    # Fit model using 1-step rolling forecast
    # --------------------------
    predictions = []
    history = list(close)

    for i in range(len(close)):
        model_fit = model.fit(history)
        pred = model_fit.predict(n_periods=1)[0]
        predictions.append(pred)
        history.append(close.iloc[i])  # update with actual value

    df["Pred_ARIMA"] = predictions
    df = df.dropna()

    # --------------------------
    # Evaluation
    # --------------------------
    actual = df["Close"]
    predicted = df["Pred_ARIMA"]

    rmse = np.sqrt(mean_squared_error(actual, predicted))
    mae = mean_absolute_error(actual, predicted)
    mape = np.mean(np.abs((actual - predicted) / actual)) * 100
    dir_acc = direction_accuracy(actual, predicted)

    print(f"\n[ARIMA Results for {ticker}]")
    print(f"RMSE: {rmse:.4f}")
    print(f"MAE: {mae:.4f}")
    print(f"MAPE: {mape:.2f}%")
    print(f"Direction Accuracy: {dir_acc:.2f}%")

    # --------------------------
    # Plot Actual vs Predicted
    # --------------------------
    plt.figure(figsize=(12, 6))
    plt.plot(df.index, actual, label="Actual Close", linewidth=2)
    plt.plot(df.index, predicted, label="ARIMA Prediction", alpha=0.7)
    plt.title(f"ARIMA Forecast - {ticker}")
    plt.xlabel("Date")
    plt.ylabel("Price")
    plt.legend()

    save_path = os.path.join(FIGURES_PATH, f"{ticker}_ARIMA.png")
    plt.savefig(save_path, dpi=300)
    plt.close()

    print(f"[INFO] Plot saved: {save_path}")


# --------------------------
# Main Runner
# --------------------------
if __name__ == "__main__":
    tickers = ["AAPL", "MSFT", "TSLA", "^GSPC"]  # You can modify list
    for t in tickers:
        run_arima(t)
