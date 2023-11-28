from abc import abstractmethod, ABC


class ReinforcementLearningEnvironment:

    def __init__(self) -> None:
        pass

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