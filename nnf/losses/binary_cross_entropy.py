"""
---------------------------------------------------------
                    BINARY CROSS-ENTROPY LOSS
---------------------------------------------------------
The Binary Cross-Entropy (BCE) Loss is commonly used for binary classification problems, 
where the model predicts the probability of an instance belonging to a particular class.

The BCE Loss computes the difference between the predicted probability (from the model) 
and the actual label (either 0 or 1), using the following formula:

    L = -[y * log(p) + (1 - y) * log(1 - p)]

Where:
    - L is the loss.
    - y is the ground truth label (either 0 or 1).
    - p is the predicted probability of the instance belonging to the positive class (i.e., the class with label 1).

The loss function outputs the average loss across all samples.

Attributes:
    output (float): The computed loss for the current batch of data.
    dinputs (ndarray): The gradient of the loss with respect to the predicted probabilities, used for backpropagation.

Methods:
    forward(y_pred, y_true):
        Computes the forward pass of the Binary Cross-Entropy loss function. This method 
        calculates the loss based on the predicted probabilities and the true labels.
        
    backward(y_pred, y_true):
        Computes the backward pass of the Binary Cross-Entropy loss function. This method 
        calculates the gradient of the loss with respect to the predicted probabilities, 
        which is used for backpropagation during training.
---------------------------------------------------------
"""

import numpy as np
from nnf.losses.base import Loss


class BinaryCrossEntropy(Loss):
    """
    Binary Cross-Entropy loss for binary classification tasks.
    """

    def __init__(self):
        super().__init__()
        self.output = None
        self.dinputs = None

        # Threshold value used for calculating accuracy in binary classification.
        # Defaults to 0.5, but can be customized via the set_threshold() method.
        self._threshold = 0.5

    def forward(self, y_pred, y_true):
        """
        Forward pass to compute the binary cross-entropy loss.

        Args:
            y_pred (ndarray): Predicted probabilities.
            y_true (ndarray): Ground truth labels (0 or 1).

        Returns:
            float: Mean binary cross-entropy loss over all samples.
        """
        y_pred = np.clip(y_pred, 1e-7, 1 - 1e-7)
        self.output = -np.mean(
            y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred)
        )
        return self.output

    def backward(self, y_pred, y_true):
        """
        Backward pass to compute the gradient of the loss with respect to predictions.

        Args:
            y_pred (ndarray): Predicted probabilities.
            y_true (ndarray): Ground truth labels (0 or 1).

        Returns:
            ndarray: Gradient of the loss with respect to y_pred.
        """
        samples = len(y_pred)
        y_pred = np.clip(y_pred, 1e-7, 1 - 1e-7)
        self.dinputs = -(y_true / y_pred - (1 - y_true) / (1 - y_pred)) / samples
        return self.dinputs

    def set_threshold(self, threshold):
        if threshold > 1 or threshold < 0.0:
            raise ValueError("threshold value should be in between 0 and 1")
        
        if threshold == 0:
            import warnings

            warnings.warn(
                f"Threshold of {threshold} is unusual. Expected range is (0, 1). "
                "This may result in incorrect predictions.",
                category=UserWarning
            )
        
        self._threshold = threshold