import yfinance as yf
import pandas as pd
import numpy as np


class YahooFinanceReader:

    def __init__(self) -> None:
        pass

    def read_stock_price_data(self, ticker):
        stock = yf.Ticker(ticker)
        return stock.history(period="max", auto_adjust=True)