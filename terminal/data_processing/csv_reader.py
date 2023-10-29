import pandas as pd
import numpy as np
from data_processing.reader import DataReader
import os


class CSVReader(DataReader):

    def __init__(self, path: str) -> None:
        super().__init__(path)

    def _read(self):
        self.df = pd.read_csv(self.path)

    def get_df(self):
        return self.df
    


