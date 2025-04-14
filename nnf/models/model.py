import math
from typing import List

from tqdm import tqdm
from tabulate import tabulate

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

    def __init__(self, *layers: Layer, name: str = None):
        """
        Initialize the model with layers.

        Parameters:
            layers (Layer): Layers to be included in the model.
        """
        self.name = name
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

            step_progress = tqdm(range(steps), desc= f"Epoch {epoch}", ncols=None, unit="steps")
            for step in step_progress:
                batch_start = step * batch_size
                batch_end = min(batch_start + batch_size, len(X))

                batch_x = X[batch_start:batch_end]
                batch_y = y[batch_start:batch_end]

                output = self.forward(batch_x)

                data_loss = self.loss.calculate(output, batch_y)
                loss += data_loss

                self.backward(output, batch_y)

                for layer in self.layers:
                    if layer.trainable:
                        self.optimizer.update_params(layer)

                self.optimizer.post_update_params()

            step_progress.set_postfix(loss=loss)
            step_progress.update(1)
            # print(f"Epoch: {epoch}, Loss: {loss}")

    def predict(self, X):
        """
        Make predictions using the trained model.

        Parameters:
            X (ndarray): Input data.

        Returns:
            Predictions from the model.
        """
        return self.forward(X)
    
    def summary(self):
        """
        Prints a summary of the model, including details of each layer and 
        the total parameters in the model.

        The summary includes:
        - Input layer information.
        - Details of each layer (type, output shape, and the number of parameters).
        - Total number of layers in the model.
        - Total number of trainable parameters.
        - Loss function used by the model.
        - Input and output shapes.
        """
        # Define the header for the summary table
        header = ["Layer (type)", "Output Shape", "Param #"]
        model_summary = []

        # Add the input layer to the summary
        model_summary.append([
            "Input", f"(None, {self.layers[0].n_inputs})", "0"
        ])

        total_params = 0
        prev_output = self.layers[0].n_neurons

        # Loop through each layer to gather details
        for layer in self.layers:
            # Update total parameters for trainable layers
            if layer.trainable:
                total_params += layer.params

            # Append layer information to model summary
            model_summary.append([
                layer.name,
                f"(None, {prev_output})",
                0 if not layer.trainable else layer.params
            ])

            # Update prev_output for the next layer
            prev_output = layer.n_neurons

        # Print the model summary
        self._print_summary(header, model_summary, total_params, prev_output)

    def _print_summary(self, header: List, model_summary: List, total_params: int, prev_output: int):
        """
        Prints the formatted summary of the model, including the table of layers
        and the total number of parameters, along with additional details like loss function.

        Args:
            header (list): List containing the column headers for the summary table.
            model_summary (list): List containing the details of each layer in the model.
            total_params (int): The total number of parameters in the model.
            prev_output (int): The output shape of the last layer, used for the output shape of the last layer.
        """

        # Print the model's basic information
        print(f"Model Summary: {self.name}\n")

        # Create and print the table using the tabulate library
        table = tabulate(
            model_summary,
            headers=header,
            tablefmt="double_grid",
            numalign="right",
            stralign="center",
            colalign=("center", "center", "center")
        )
        print(table)

        # Print additional information such as total layers, total parameters, loss function, and shapes
        print(f"\nTotal Layers: {len(self.layers)}")
        print(f"Total parameters: {total_params:,}")  # Formatting the total parameters with commas
        print(f"Loss: {self.loss.name}")
        print(f"Input Shape: (None, {self.layers[0].n_inputs})")
        print(f"Output Shape: (None, {prev_output})")
