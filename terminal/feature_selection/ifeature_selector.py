from abc import abstractmethod, ABC


class IFeatureSelector(ABC):

    def __init__(self) -> None:
        super().__init__()

    @abstractmethod
    def run(self):
        pass

    @abstractmethod
    def plot(self):
        pass