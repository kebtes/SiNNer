"""
---------------------------------------------------------
                    SOFTMAX ACTIVATION FUNCTION
---------------------------------------------------------
The Softmax activation function converts a vector of raw scores (logits)
into a probability distribution. It is commonly used as the activation 
function in the output layer of a classification model.

Given a vector of raw scores, the softmax function computes the 
probability for each class using the following formula:

    f(x_i) = exp(x_i) / sum(exp(x_j) for j in all classes)

Where:
    - x_i is the raw score (logit) for class i.
    - f(x_i) is the probability of class i.
    
The function ensures that all the probabilities sum up to 1, making it suitable 
for classification tasks where the model needs to output a probability distribution.

Attributes:
    inputs (ndarray): Raw input data (logits).
    output (ndarray): Output probabilities after applying softmax.
    dinputs (ndarray): Gradient of the loss with respect to the inputs.

Methods:
    forward(inputs):
        Computes the forward pass of the softmax activation function.
        
    backward(dvalues):
        Computes the backward pass, propagating gradients through the softmax function.
---------------------------------------------------------
"""

import numpy as np
from nnf.activations.base import Activation

class Softmax(Activation):
    def __init__(self):
        self.inputs = None
        self.output = None
        self.dinputs = None

    def forward(self, inputs):
        """
        Forward pass of Softmax.

        Args:
            inputs (ndarray): Raw scores (logits).

        Returns:
            ndarray: Probabilities.
        """
        self.inputs = inputs

        # Subtract the max value for numerical stability
        # This won't cause any error as Softmax isn't scale dependent
        exponent_values = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))

        probabilites = exponent_values / np.sum(exponent_values, axis=1, keepdims=True)
    
        if np.isnan(probabilites).any():
            raise ValueError("NaN values detected in Softmax output.")
        
        self.output = probabilites
        return self.output

    def backward(self, dvalues):
        """
        Backward pass of Softmax.

        NOTE: Normally used together with cross-entropy loss,
        and we use the combined derivative for efficiency.

        This method assumes dvalues already contains the proper gradient.

        Args:
            dvalues (ndarray): Gradient from loss.

        Returns:
            ndarray: Gradient of the loss with respect to the inputs.
        """
        self.dinputs = dvalues  # usually combined with loss
        return self.dinputs
