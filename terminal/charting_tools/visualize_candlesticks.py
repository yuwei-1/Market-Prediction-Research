import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from abc import ABC, abstractmethod
from tqdm import tqdm
import plotly.graph_objects as go
import plotly.express as px


def plot_candlestick(df):

    fig = go.Figure(data=[go.Candlestick(x=df['Date'],
            open=df['Open'],
            high=df['High'],
            low=df['Low'],
            close=df['Close'])])

    return fig

def plot_added_features(df, fig, features):

    for f in features:
        assert f in df.columns, "This feature is not present within the dataframe"

        fig.add_traces(data=[go.Scatter(x=df["Date"], y=df[f])])

    return fig

def plot_all(df, features):

    fig = plot_candlestick(df)
    fig = plot_added_features(df, fig, features)

    fig.show()