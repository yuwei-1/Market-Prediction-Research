import numpy as np
import pandas as pd


class Differencer():

    def __init__(self) -> None:
        pass

    def fit_difference(self, data: pd.Series, order = 1) -> np.ndarray:

        assert order == 1, "Only first order differencing is supported at the moment"

        self.order = order
        data = np.array(data)
        differences = data[1:] - data[:-1]
        self.start = data[0]

        return differences
    
    def inverse_difference(self, data: np.ndarray) -> np.ndarray:

        # TODO: make this more efficient
        data[0] += self.start

        for i in range(1, len(data)):
            data[i] += data[i-1]
        
        return data  