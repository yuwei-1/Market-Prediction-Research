import torch
import torch.nn as nn
import torch.nn.functional as F
from abc import ABC, abstractmethod


class ModelTrainer(ABC):

    def __init__(self) -> None:
        super().__init__()
        self.create_model_instance()
    
    @abstractmethod
    def create_model_instance(self):
        pass

    @abstractmethod
    def train(self):
        pass

    @abstractmethod
    def evaluate(self):
        pass