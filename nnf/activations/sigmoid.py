import numpy as np
from typing import override

from nnf.activations.base import Activation

class Sigmoid(Activation):
    """
    -----------------------------------------------
            SIGMOID ACTIVATION FUNCTION
    f(x) = 1 / (1 + e ^ (-x))
    USED TO MAP ANY INPUT TO A VALUE BETWEEN 0 AND 1
    -----------------------------------------------
    """

    def forward(self, inputs):
        """
        The forward function takes inputs, applies the sigmoid function, and returns the output.
        """
        self.inputs = inputs
        self.output = 1 / (1 + np.exp(-inputs))
        return self.output
    
    def backward(self, dvalues):
        """
        The `backward` function calculates the derivative of the inputs based on the output and the
        given derivative values.
        """

        # derivative of the sigmoid: f'(x) = f(x) * (1 - f(x))
        self.dinputs = dvalues * (self.output * (1 - self.output))
        return self.dinputs
    
    @override
    def get_params(self):
        return {
            "type" : "Sigmoid",
            "attrs" : {}
        }