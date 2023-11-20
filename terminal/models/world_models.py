from abc import abstractmethod, ABC

class WorldModel:
    '''
    World model is for model based (Dyna) agents for constructing
    an internal representation of its environment

    WorldModel(s, a) -> s', r
    '''
    def __init__(self) -> None:
        pass
    
    @abstractmethod
    def observe(self):
        pass

    @abstractmethod
    def predict(self):
        pass