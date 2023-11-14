import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from abc import ABC, abstractmethod
from tqdm import tqdm
import plotly.graph_objects as go
import plotly.express as px

class DataReader():

    def __init__(self, path: str) -> None:
        self.path = path
        self._read()

    @abstractmethod
    def _read(self):
        pass
    
    @abstractmethod
    def get_df(self):
        self.df = pd.DataFrame()

    def get_properties(self):
        self.col_names = self.df.columns
        self.num_cols = len(self.col_names)
        self.num_rows = len(self.df)
        self.dtypes = self.df.dtypes

        print(self.df.head(5))

    def visualize_data_distribution(self):
        
        fig, axs = plt.subplots(nrows=self.num_cols, ncols=1, figsize=(10,40))

        for i in tqdm(range(self.num_cols)):
            axs[i].hist(self.df[self.col_names[i]])

        plt.tight_layout()