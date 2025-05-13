import numpy as np
from typing import override, Dict, List

from nnf.layers.base import Layer

class Dense(Layer):
    """
    --------------------------------------------
    FULLY CONNECTED (DENSE) LAYER IMPLEMENTATION
    --------------------------------------------
    """
    
    def __init__(self, n_inputs = 1, n_neurons = 1):
        """
        The function initializes weights with random values and biases with zeros.
        """
        super().__init__()
        
        self.n_inputs = n_inputs
        self.n_neurons = n_neurons

        # Dense layer have weights hense trainable
        self.trainable = True

        # Initlize weights with random values, biases with zeros
        # self.weights = 0.1 * np.random.randn(n_inputs, n_neurons) 
        self.weights = np.random.randn(n_inputs, n_neurons) * np.sqrt(2.0 / (n_inputs + n_neurons))
        self.biases = np.zeros(shape=(1, n_neurons))
        
        # Gradients
        self.dweights = None
        self.dbiases = None

        # Parameters

        self.params = self.weights.size + self.biases.size

    def forward(self, inputs):
        """
        The `forward` function takes inputs, calculates the output using weights and biases, and returns
        the result.
        """
        self.inputs = inputs 

        self.output = np.dot(inputs, self.weights) + self.biases
        return self.output

    def backward(self, dvalues):
        """
        The `backward` function calculates the gradients of the weights, biases, and inputs in a neural
        network during backpropagation.
        """
        
        self.dweights = np.dot(self.inputs.T, dvalues)
        self.dbiases = np.sum(dvalues, axis=0, keepdims=True)

        self.dinputs = np.dot(dvalues, self.weights.T)
        return self.dinputs
    
    @override
    def get_params(self):
        return {
            "type"  : "Dense",
            "attrs" : {
                "n_inputs"  : self.n_inputs,
                "n_neurons" : self.n_neurons,
                "trainable" : self.trainable,
                "weights"   : self.weights,
                "biases"    : self.biases,
                "dweights"  : self.dweights,
                "dbiases"   : self.dbiases
            }  
        }
        
    @override
    def set_params(self, params : Dict):
        for key, val in params.items():
            # If the key is one of the specified attributes (e.g., "n_inputs", "n_neurons", etc.)
            # and the value is a list, convert the value to a NumPy array and set it as an attribute.
            # Basically, to how they were originally!
            if key in ("n_inputs", "n_neurons", "weights", "biases", "dweights", "dbiases") and type(key) == List:
                setattr(self, key, np.array(val))
            else:
                setattr(self, key, val)
    