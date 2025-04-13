from abc import ABC, abstractmethod

class Layer(ABC):
    def __init__(self):
        self.inputs = None
        self.output = None
        self.dinputs = None

    @abstractmethod
    def forward(self, inputs):
        pass

    @abstractmethod
    def backward(self, dvalues):
        pass