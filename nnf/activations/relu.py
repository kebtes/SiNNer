import numpy as np
from nnf.activations.base import Activation

class ReLU(Activation):
    """
    -----------------------------------------------
    RECTIFIED LINEAR UNIT (ReLU) ACTIVATION FUNCTION
    f(x) = max(0, x)
    -----------------------------------------------
    """
    def forward(self, inputs):
        """
        The `forward` function takes an input array, applies a ReLU activation function element-wise,
        and returns the resulting output array.
        """
        self.inputs = inputs
        self.output = np.maximum(0, inputs)
        return self.output
    
    def backward(self, dvalues):
        """
        The `backward` function calculates the derivative of the inputs with respect to the given values
        and sets the derivative to zero for inputs less than or equal to zero.
        """

        self.dinputs = dvalues.copy()
        self.dinputs[self.inputs <= 0] = 0
        return self.dinputs