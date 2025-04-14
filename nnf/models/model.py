import math
from typing import List

from nnf.layers.base import Layer
from nnf.losses.base import Loss
from nnf.optimizers.base import Optimizer

# Module docstring
"""
This module defines a neural network Model class that combines layers and 
handles the training, forward pass, and backward pass operations.
"""

class Model:
    """
    --------------------------------------------------------------
    A Neural Network Model class that handles layers, training, 
    forward pass, backward pass, and prediction.
    --------------------------------------------------------------
    """

    def __init__(self, *layers: Layer):
        """
        Initialize the model with layers.

        Parameters:
            layers (Layer): Layers to be included in the model.
        """
        self.layers: List[Layer] = list(layers)
        self.loss: Loss = None
        self.optimizer: Optimizer = None

    def set(self, loss: Loss, optimizer: Optimizer):
        """
        Set the loss function and optimizer for the model.

        Parameters:
            loss (Loss): The loss function.
            optimizer (Optimizer): The optimizer to use.
        """
        self.loss = loss
        self.optimizer = optimizer

    def forward(self, X):
        """
        Perform a forward pass through the layers.

        Parameters:
            X (ndarray): Input data.

        Returns:
            Output after passing through all layers.
        """
        for layer in self.layers:
            X = layer.forward(X)
        return X

    def backward(self, output, y):
        """
        Perform a backward pass through the layers to calculate gradients.

        Parameters:
            output (ndarray): The predicted output from the forward pass.
            y (ndarray): The true labels.
        """
        dinputs = self.loss.backward(output, y)
        self.layers[-1].backward(dinputs)

        for i in reversed(range(len(self.layers) - 1)):
            self.layers[i].backward(self.layers[i + 1].dinputs)

    def train(self, X, y, *, epochs=1, batch_size: int = None):
        """
        Train the model using the provided data.

        Parameters:
            X (ndarray): Input data.
            y (ndarray): True labels.
            epochs (int): Number of epochs to train for.
            batch_size (int): Size of the training batches. Defaults to None.
        """
        if batch_size is None:
            batch_size = len(X)

        steps = math.ceil(len(X) / batch_size)

        for epoch in range(1, epochs + 1):
            loss = 0

            for step in range(steps):
                batch_start = step * batch_size
                batch_end = min(batch_start + batch_size, len(X))

                batch_x = X[batch_start:batch_end]
                batch_y = y[batch_start:batch_end]

                output = self.forward(batch_x)

                data_loss = self.loss.calculate(output, batch_y)
                loss += data_loss

                self.backward(output, batch_y)

                for layer in self.layers:
                    if hasattr(layer, "weights"):
                        self.optimizer.update_params(layer)

                self.optimizer.post_update_params()

            print(f"Epoch: {epoch}, Loss: {loss}")

    def predict(self, X):
        """
        Make predictions using the trained model.

        Parameters:
            X (ndarray): Input data.

        Returns:
            Predictions from the model.
        """
        return self.forward(X)
