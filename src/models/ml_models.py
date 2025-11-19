"""
ml_models.py
Machine Learning models for stock forecasting.

Models included:
    - Linear Regression
    - RandomForestRegressor
    - XGBoostRegressor (optional)

Uses engineered features from data/processed/
Saves plots in reports/figures/
"""

import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split

# Try importing XGBoost
try:
    from xgboost import XGBRegressor
    XGB_AVAILABLE = True
except:
    XGB_AVAILABLE = False

DATA_PROCESSED_PATH = "data/processed"
FIGURES_PATH = "reports/figures"

os.makedirs(FIGURES_PATH, exist_ok=True)


def direction_accuracy(actual, predicted):
    actual_diff = actual.diff().dropna()
    predicted_diff = pd.Series(predicted, index=actual.index).diff().dropna()
    return np.mean(np.sign(actual_diff) == np.sign(predicted_diff)) * 100


# --------------------------
# Model Training Function
# --------------------------

def train_ml_models(ticker):
    print(f"\n=== ML MODELS FOR {ticker} ===")

    file_path = os.path.join(DATA_PROCESSED_PATH, f"{ticker}_features.csv")

    if not os.path.exists(file_path):
        print(f"[ERROR] Processed file not found for {ticker}.")
        return

    df = pd.read_csv(file_path, parse_dates=["Date"], index_col="Date")

    # Drop NaNs (very important)
    df = df.dropna()

    # Target variable
    y = df["Close"]

    # Features
    X = df.drop(columns=["Close"])

    # Train/validation split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, shuffle=False
    )

    # Scaling
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    results = {}

    # --------------------------
    # 1. Linear Regression
    # --------------------------
    lr = LinearRegression()
    lr.fit(X_train_scaled, y_train)
    lr_pred = lr.predict(X_test_scaled)

    results["LinearRegression"] = lr_pred

    # --------------------------
    # 2. Random Forest
    # --------------------------
    rf = RandomForestRegressor(n_estimators=300, random_state=42)
    rf.fit(X_train, y_train)
    rf_pred = rf.predict(X_test)

    results["RandomForest"] = rf_pred

    # --------------------------
    # 3. XGBoost (if installed)
    # --------------------------
    if XGB_AVAILABLE:
        xgb = XGBRegressor(
            n_estimators=300,
            learning_rate=0.05,
            max_depth=6,
            subsample=0.9,
            colsample_bytree=0.9,
            random_state=42
        )
        xgb.fit(X_train, y_train)
        xgb_pred = xgb.predict(X_test)
        results["XGBoost"] = xgb_pred
    else:
        print("[WARNING] XGBoost not installed. Skipping.")

    # --------------------------
    # Evaluation + Plotting
    # --------------------------
    for model_name, pred in results.items():
        rmse = np.sqrt(mean_squared_error(y_test, pred))
        mae = mean_absolute_error(y_test, pred)
        mape = np.mean(np.abs((y_test - pred) / y_test)) * 100
        dir_acc = direction_accuracy(y_test, pred)

        print(f"\n[{model_name}]")
        print(f"RMSE: {rmse:.4f}")
        print(f"MAE: {mae:.4f}")
        print(f"MAPE: {mape:.2f}%")
        print(f"Direction Accuracy: {dir_acc:.2f}%")

        # Plot predictions
        plt.figure(figsize=(12, 6))
        plt.plot(y_test.index, y_test, label="Actual", linewidth=2)
        plt.plot(y_test.index, pred, label=model_name, alpha=0.8)
        plt.title(f"{ticker} - {model_name} Prediction")
        plt.xlabel("Date")
        plt.ylabel("Price")
        plt.legend()

        save_path = os.path.join(FIGURES_PATH, f"{ticker}_{model_name}.png")
        plt.savefig(save_path, dpi=300)
        plt.close()
        print(f"[INFO] Plot saved: {save_path}")

    print(f"\n[INFO] ML models completed for {ticker}.")


# --------------------------
# Run for all ticker files
# --------------------------
if __name__ == "__main__":
    tickers = ["AAPL", "MSFT", "TSLA", "^GSPC"]
    for t in tickers:
        train_ml_models(t)
        
