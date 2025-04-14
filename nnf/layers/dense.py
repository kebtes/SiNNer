import numpy as np
from nnf.layers.base import Layer

class Dense(Layer):
    """
    --------------------------------------------
    FULLY CONNECTED (DENSE) LAYER IMPLEMENTATION
    --------------------------------------------
    """
    
    def __init__(self, n_inputs, n_neurons):
        """
        The function initializes weights with random values and biases with zeros.
        """
        super().__init__()

        self.n_inputs = n_inputs
        self.n_neurons = n_neurons

        # Dense layer have weights hense trainable
        self.trainable = True

        # Initlize weights with random values, biases with zeros
        self.weights = 0.1 * np.random.randn(n_inputs, n_neurons) 
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