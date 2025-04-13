import numpy as np
from nnf.activations.base import Activation

class LeakyReLU(Activation):
    """
    -----------------------------------------------
              LEAKY RELU ACTIVATION FUNCTION
    f(x) = x if x > 0, otherwise f(x) = α * x
    This prevents neurons from "dying" by allowing a small,
    non-zero gradient when the unit is not active.
    -----------------------------------------------
    """

    def __init__(self, alpha=0.01):
        """
        Initialize the LeakyReLU with a given alpha.

        Parameters:
            alpha (float): the small constant multiplier for negative inputs. Default is 0.01.
        """
        self.alpha = alpha

    def forward(self, inputs):
        """
        The forward pass applies the LeakyReLU function.

        Parameters:
            inputs (ndarray): input data to the layer

        Returns:
            output (ndarray): transformed output of the layer
        """
        self.inputs = inputs
        # For each element, if positive keep it; if negative, multiply by alpha
        self.output = np.where(inputs > 0, inputs, self.alpha * inputs)
        return self.output
    
    def backward(self, dvalues):
        """
        The backward pass computes the gradient of the loss with respect to the inputs.

        Parameters:
            dvalues (ndarray): the gradient from the next layer

        Returns:
            dinputs (ndarray): the gradient with respect to the inputs of this layer
        """
        # Initialize gradient as a copy of dvalues
        self.dinputs = np.copy(dvalues)
        # For inputs <= 0, multiply the gradient by alpha
        self.dinputs[self.inputs <= 0] *= self.alpha
        return self.dinputs
