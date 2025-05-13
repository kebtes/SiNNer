"""
---------------------------------------------------------
                    TANH ACTIVATION FUNCTION
---------------------------------------------------------
The Tanh activation function maps input values to the range of 
[-1, 1]. It is commonly used in hidden layers of neural networks.

Given an input, the tanh function is defined as:

    f(x) = (exp(x) - exp(-x)) / (exp(x) + exp(-x))

Where:
    - x is the input value.

The function outputs values between -1 and 1, and it is symmetric 
around the origin, making it suitable for tasks where both positive 
and negative values are important.

Attributes:
    inputs (ndarray): Input data passed to the activation function.
    output (ndarray): Output values after applying the tanh function.
    dinputs (ndarray): Gradient of the loss with respect to the inputs.
    
Methods:
    forward(inputs):
        Computes the forward pass for the tanh activation function.
        
    backward(dvalues):
        Computes the backward pass, propagating gradients through the tanh function.
---------------------------------------------------------
"""

import numpy as np
from typing import override

from nnf.activations.base import Activation

class Tanh(Activation):
    def __init__(self):
        self.inputs = None
        self.output = None
        self.dinputs = None

    def forward(self, inputs):
        """
        Forward pass of Tanh.

        Args:
            inputs (ndarray): Input values.

        Returns:
            ndarray: Output values after applying tanh.
        """
        self.inputs = inputs

        output = np.tanh(inputs)
        if np.isnan(output).any():
            raise ValueError("NaN values detected in Softmax output.")
        
        self.output = output
        return self.output

    def backward(self, dvalues):
        """
        Backward pass of Tanh.

        Args:
            dvalues (ndarray): Gradient from the next layer.

        Returns:
            ndarray: Gradient with respect to the inputs.
        """
        self.dinputs = dvalues * (1 - self.output ** 2)
        return self.dinputs

    @override
    def get_params(self):
        return {
            "type"  : "Tanh",
            "attrs" : {}
        }