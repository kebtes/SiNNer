from abc import ABC, abstractmethod
from typing import Dict

class Layer(ABC):
    def __init__(self):
        self.inputs = None
        self.output = None
        self.dinputs = None

        self.n_inputs = None
        self.n_neurons = None

        self.trainable = False
        
        self.name = self.__class__.__name__
    
    @abstractmethod
    def forward(self, inputs):
        pass

    @abstractmethod
    def backward(self, dvalues):
        pass

    def get_params(self):
        return {}

    def set_params(self, params : Dict):
        pass