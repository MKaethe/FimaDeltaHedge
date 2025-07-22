# Here are some utility functions for file management and stock data retrieval
from alpaca.data.requests import StockBarsRequest, CryptoBarsRequest
from alpaca.data.timeframe import TimeFrame, TimeFrameUnit
import pandas as pd
from alpaca.data.historical.stock import StockHistoricalDataClient
import pickle
import os
from config import API_KEY,API_SECRET


def verify_directory_exists(path):
    os.makedirs(path, exist_ok=True)


def save_to_path(path, object):
    with open(path, "wb") as f:
        pickle.dump(object, f)


def load_from_path(path):
    with open(path, "rb") as f:
        return pickle.load(f)


def get_stock_data(ticker):
    # With caching we don't have to redownload the data
    folder_path = "real_stock_data"
    verify_directory_exists(folder_path)
    filename = f"stock_data_{ticker}"
    filepath = os.path.join(folder_path, filename)

    if os.path.exists(filepath):
        return load_from_path(filepath)

    # Start timestamp for getting the real data
    start_timestamp = pd.Timestamp(year=2020, month=1, day=1, tz="UTC")

    stockdata_client = StockHistoricalDataClient(
        API_KEY, API_SECRET
    )

    sbr = StockBarsRequest(
        symbol_or_symbols=ticker,
        timeframe=TimeFrame(1, TimeFrameUnit.Minute),
        start=start_timestamp,
    )

    data = stockdata_client.get_stock_bars(sbr).df.rename(
        columns={
            "close": "Close",
            "high": "High",
            "low": "Low",
            "open": "Open",
            "volume": "Volume",
        }
    )

    data: pd.DataFrame = data.droplevel("symbol")

    save_to_path(filepath,data)
    
    return data


if __name__ == "__main__":
    print(pd.Timedelta(days=305.3))
