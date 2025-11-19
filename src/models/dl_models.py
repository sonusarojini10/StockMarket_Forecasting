"""
dl_models.py
PyTorch LSTM model for stock forecasting.

✓ Works on Windows + Python 3.12
✓ No TensorFlow needed
✓ Uses processed features to predict Close price
✓ Saves model, predictions, and plots
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

# Paths
DATA_PROCESSED_PATH = "data/processed"
FIGURES_PATH = "reports/figures"
MODELS_PATH = "models"

os.makedirs(FIGURES_PATH, exist_ok=True)
os.makedirs(MODELS_PATH, exist_ok=True)

# -------------------------------
# PyTorch LSTM Model
# -------------------------------

class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size=64, num_layers=2):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True
        )
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = out[:, -1, :]      # last timestep
        out = self.fc(out)
        return out


# -------------------------------
# Sequence Generator
# -------------------------------

def create_sequences(X, y, window=60):
    xs, ys = [], []
    for i in range(window, len(X)):
        xs.append(X[i-window:i])
        ys.append(y[i])
    return np.array(xs), np.array(ys)


# -------------------------------
# Training Function
# -------------------------------

def train_lstm(ticker, window=60, epochs=30, batch_size=32, lr=0.001):
    print(f"\n=== Training LSTM for {ticker} (PyTorch) ===")

    file_path = os.path.join(DATA_PROCESSED_PATH, f"{ticker}_features.csv")
    if not os.path.exists(file_path):
        print("[ERROR] Processed file not found.")
        return

    df = pd.read_csv(file_path, parse_dates=["Date"], index_col="Date")
    df = df.dropna()

    # Features (all except Close)
    X = df.drop(columns=["Close"]).values
    y = df["Close"].values.reshape(-1, 1)

    # Scale features & target
    X_scaler = MinMaxScaler()
    y_scaler = MinMaxScaler()

    X_scaled = X_scaler.fit_transform(X)
    y_scaled = y_scaler.fit_transform(y)

    # Train-test split (80/20)
    split = int(len(X_scaled) * 0.8)
    X_train, X_test = X_scaled[:split], X_scaled[split:]
    y_train, y_test = y_scaled[:split], y_scaled[split:]

    # Sequences
    X_train_seq, y_train_seq = create_sequences(X_train, y_train, window)
    X_test_seq, y_test_seq = create_sequences(
        np.concatenate([X_train[-window:], X_test]),
        np.concatenate([y_train[-window:], y_test]),
        window
    )

    # Convert to PyTorch tensors
    X_train_t = torch.tensor(X_train_seq, dtype=torch.float32)
    y_train_t = torch.tensor(y_train_seq, dtype=torch.float32)

    X_test_t = torch.tensor(X_test_seq, dtype=torch.float32)

    train_loader = DataLoader(TensorDataset(X_train_t, y_train_t),
                              batch_size=batch_size, shuffle=True)

    # Model
    model = LSTMModel(input_size=X_train_t.shape[2])
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # Training Loop
    model.train()
    for epoch in range(epochs):
        epoch_loss = 0
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            output = model(X_batch)
            loss = criterion(output, y_batch)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        print(f"Epoch {epoch+1}/{epochs} | Loss: {epoch_loss:.4f}")

    # Prediction
    model.eval()
    with torch.no_grad():
        pred_scaled = model(X_test_t).numpy()

    # Inverse scale
    pred = y_scaler.inverse_transform(pred_scaled)
    y_true = y_scaler.inverse_transform(y_test_seq)

    # Metrics
    rmse = np.sqrt(mean_squared_error(y_true, pred))
    mae = mean_absolute_error(y_true, pred)
    mape = np.mean(np.abs((y_true - pred) / y_true)) * 100

    print(f"\n[{ticker} LSTM Results]")
    print(f"RMSE: {rmse:.4f}")
    print(f"MAE: {mae:.4f}")
    print(f"MAPE: {mape:.2f}%")

    # Save predictions plot
    plt.figure(figsize=(12,6))
    plt.plot(y_true, label="Actual")
    plt.plot(pred, label="Predicted")
    plt.title(f"{ticker} - LSTM Prediction (PyTorch)")
    plt.legend()
    plot_path = os.path.join(FIGURES_PATH, f"{ticker}_LSTM_PyTorch.png")
    plt.savefig(plot_path)
    plt.close()
    print(f"[INFO] Saved plot: {plot_path}")

    # Save model
    model_path = os.path.join(MODELS_PATH, f"{ticker}_lstm.pth")
    torch.save(model.state_dict(), model_path)
    print(f"[INFO] Model saved: {model_path}")


# -------------------------------
# Run for all tickers
# -------------------------------

if __name__ == "__main__":
    tickers = ["AAPL", "MSFT", "TSLA", "^GSPC"]
    for t in tickers:
        train_lstm(ticker=t, window=60, epochs=25, batch_size=32, lr=0.001)
