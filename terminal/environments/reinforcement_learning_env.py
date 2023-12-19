from abc import abstractmethod, ABC
from collections import namedtuple


class ReinforcementLearningEnvironment:

    def __init__(self) -> None:
        self.Aspace = namedtuple('action_space', ('n', 'sample'))
        self.Obspace = namedtuple('observation_space', 'shape')
        self.actions = {"buy":1, "sell":0, "hold":2}

    @abstractmethod
    def make(self):
        pass

    @abstractmethod
    def step(self):
        pass
    
    @abstractmethod
    def reset(self):
        pass

    @abstractmethod
    def make_portfolio(self):
        pass

    @abstractmethod
    def close(self):
        pass