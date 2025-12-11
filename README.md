📈 Stock Market Forecasting Project

A Machine Learning + Deep Learning Based Predictive System with Real-Time Streamlit Dashboard

* Overview

This project implements end-to-end stock market forecasting using:

ARIMA – classical statistical forecasting

Machine Learning models (Linear Regression, Random Forest, XGBoost)

Deep Learning (LSTM – PyTorch)

Real-Time Prediction Dashboard (Streamlit)

It fetches historical stock data, performs feature engineering, trains multiple models, evaluates them, and provides live forecasts through an interactive UI.

* Project Architecture
```python
StockMarket_Forecasting/
│
├── data/
│   ├── raw/              # downloaded OHLC data
│   └── processed/        # engineered feature sets
│
├── models/               # trained LSTM model weights (.pth)
│
├── reports/
│   └── figures/          # prediction graphs & evaluation plots
│
├── src/
│   ├── data/fetch_data.py         # download stock data
│   ├── features/features.py       # feature engineering
│   ├── models/
│   │     ├── baseline.py          # naive / drift / rolling mean
│   │     ├── arima_model.py       # ARIMA model
│   │     ├── ml_models.py         # ML models
│   │     └── dl_models.py         # LSTM (PyTorch)
│   └── app/app.py                 # Streamlit dashboard
│
├── presentation/                  # PDF + demo video (optional)
├── requirements.txt
└── README.md
```

🧪 Implemented Models
📌 1. Baseline Models
Model	Description
Naive	Forecast = last observed close
Rolling Mean	Moving average
Drift Model	Linear trend projection

📌 2. ARIMA Model

Uses pmdarima.auto_arima

Handles trend + seasonality

Evaluates using RMSE, MAE, MAPE

📌 3. Machine Learning Models
Model	Library
Linear Regression	scikit-learn
Random Forest Regressor	scikit-learn
XGBoost	optional

Uses engineered features such as SMA, EMA, RSI, MACD, Volatility, Lags.

📌 4. Deep Learning – LSTM (PyTorch)

2-layer LSTM

64 hidden units

Sliding window time-series input

Trained on engineered features

Output: next-day predicted Close price

Trained models saved in:

models/<TICKER>_lstm.pth

🧠 Feature Engineering

Generated in features.py:

Simple Moving Average (SMA-20)

Exponential Moving Average (EMA-20)

RSI-14

MACD + Signal + Histogram

Bollinger Bands (Mid, Upper, Lower)

Returns & Log Returns

Rolling Volatility (20)

Lag Features (1 to 10 days)

Total Features Used for ML & LSTM: 26

🖥️ Real-Time Prediction Dashboard (Streamlit)

Run:

streamlit run src/app/app.py


Features:

Fetches real-time data using yfinance

Recomputes all 26 features

Loads trained LSTM model

Produces live next-day forecast

Displays:

Actual trend

LSTM predicted point

Naive baseline prediction

📊 Example Outputs
LSTM Prediction Example

(Uploaded to reports/figures/)

AAPL_LSTM_PyTorch.png

MSFT_LSTM_PyTorch.png

TSLA_LSTM_PyTorch.png

Baseline Examples

AAPL_Pred_naive.png

AAPL_Pred_drift.png

ML Model Plots

AAPL_LinearRegression.png

AAPL_RandomForest.png

All included in your repository.

🛠️ Installation
1. Clone Repo
git clone https://github.com/sonusarojini10/StockMarket_Forecasting.git
cd StockMarket_Forecasting

2. Install Dependencies

(Use Anaconda recommended)

pip install -r requirements.txt

▶️ Running the Entire Pipeline
1. Fetch historical data
python src/data/fetch_data.py

2. Feature engineering
python src/features/features.py

3. Train models

Baseline

python src/models/baseline.py


ARIMA

python src/models/arima_model.py


Machine Learning

python src/models/ml_models.py


LSTM

python src/models/dl_models.py

4. Launch the Streamlit Dashboard
streamlit run src/app/app.py

 Demonstration (For Presentation)


Add hyperparameter tuning

Add Prophet / Transformer models

Build REST API for real-time inference

Deploy Streamlit app on Cloud

Add more tickers + crypto

🧑‍💻 Author

Munnangi Sonu Sarojini
Email: sonusarojini_munnangi@srmap.edu.in
video link: https://www.youtube.com/watch?v=HddlQncMeFc

GitHub: sonusarojini10
