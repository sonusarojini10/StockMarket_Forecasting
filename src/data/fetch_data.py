"""
fetch_data.py
Simple, robust stock data downloader using yfinance.

Saves CSV files in: ../data/raw/<TICKER>.csv

Usage examples (from project root `D:\StockMarket_forcasting`):
> python src\data\fetch_data.py --tickers AAPL MSFT TSLA --start 2015-01-01 --end 2025-11-18 --interval 1d
> python src\data\fetch_data.py --tickers AAPL --period 10y

Notes:
- If both --period and --start/--end provided, --start/--end take precedence.
- Interval examples: 1d, 1wk, 1mo, 60m (intraday may require more handling)
"""

import argparse
import os
import sys
from datetime import datetime
import logging

import pandas as pd
import yfinance as yf

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")


def ensure_dirs(project_root: str):
    raw_dir = os.path.join(project_root, "data", "raw")
    processed_dir = os.path.join(project_root, "data", "processed")
    os.makedirs(raw_dir, exist_ok=True)
    os.makedirs(processed_dir, exist_ok=True)
    return raw_dir, processed_dir


def download_ticker(ticker: str, raw_dir: str, start: str = None, end: str = None, period: str = None, interval: str = "1d", threads: bool = True):
    """
    Download OHLCV for a ticker and save to CSV.
    Returns the DataFrame.
    """
    logging.info(f"Downloading {ticker} (interval={interval}) ...")
    try:
        # yfinance handles period OR start/end
        if start or end:
            df = yf.download(ticker, start=start, end=end, interval=interval, progress=False, threads=threads)
        else:
            df = yf.download(ticker, period=period or "max", interval=interval, progress=False, threads=threads)

        if df is None or df.empty:
            logging.warning(f"No data returned for {ticker}.")
            return None

        # Ensure datetime index, sort, and drop duplicates
        df.index = pd.to_datetime(df.index)
        df = df.sort_index()
        df = df[~df.index.duplicated(keep="first")]

        csv_path = os.path.join(raw_dir, f"{ticker.upper()}.csv")
        df.to_csv(csv_path, float_format="%.8f", index_label="Date")
        logging.info(f"Saved {len(df)} rows to {csv_path}")
        return df
    except Exception as e:
        logging.exception(f"Failed to download {ticker}: {e}")
        return None


def download_batch(tickers, raw_dir, start=None, end=None, period=None, interval="1d"):
    results = {}
    for t in tickers:
        df = download_ticker(t, raw_dir, start=start, end=end, period=period, interval=interval)
        results[t.upper()] = df
    return results


def parse_args(argv):
    p = argparse.ArgumentParser(description="Download stock OHLCV data using yfinance and save CSVs to data/raw")
    p.add_argument("--tickers", "-t", nargs="+", required=True, help="List of tickers (e.g. AAPL MSFT ^GSPC)")
    p.add_argument("--start", type=str, default=None, help="Start date YYYY-MM-DD")
    p.add_argument("--end", type=str, default=None, help="End date YYYY-MM-DD (exclusive)")
    p.add_argument("--period", type=str, default=None, help="Period e.g. 1y, 5y, 10y, max (used if start/end not provided)")
    p.add_argument("--interval", type=str, default="1d", help="Data interval (1d, 1wk, 1mo, 60m, etc.)")
    p.add_argument("--project-root", type=str, default=os.getcwd(), help="Project root (defaults to cwd)")
    return p.parse_args(argv)


def main(argv):
    args = parse_args(argv)

    # project root should be D:\StockMarket_forcasting if you run from there
    project_root = os.path.abspath(args.project_root)
    raw_dir, processed_dir = ensure_dirs(project_root)

    # Validate dates (if provided)
    start = args.start
    end = args.end
    if start:
        try:
            datetime.fromisoformat(start)
        except Exception:
            logging.error("Invalid --start date. Use YYYY-MM-DD")
            sys.exit(1)
    if end:
        try:
            datetime.fromisoformat(end)
        except Exception:
            logging.error("Invalid --end date. Use YYYY-MM-DD")
            sys.exit(1)

    tickers = [t.strip() for t in args.tickers if t.strip()]
    logging.info(f"Tickers to download: {tickers}")

    results = download_batch(tickers, raw_dir, start=start, end=end, period=args.period, interval=args.interval)

    # Print summary
    for t, df in results.items():
        if df is None:
            logging.warning(f"{t}: no data saved.")
        else:
            logging.info(f"{t}: saved {len(df)} rows ({df.index.min().date()} -> {df.index.max().date()})")

    logging.info("Done.")


if __name__ == "__main__":
    main(sys.argv[1:])
