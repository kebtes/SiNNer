import numpy as np
from nnf.layers import Layer

class Dropout(Layer):
    """
    Dropout layer implementing inverted dropout.

    During training, it randomly sets a fraction `p` of input units to zero 
    at each update and scales the rest by `1 / (1 - p)` to maintain the expected output. 
    During inference, it passes the input through unchanged.

    Attributes:
        p (float): Dropout probability. Fraction of inputs to drop.
        mask (np.ndarray): Binary mask used to drop/keep inputs during training.
        training (bool): Flag indicating whether the layer is in training mode.
    """

    def __init__(self, p: float = 0.5):
        """
        Initializes the Dropout layer.

        Args:
            p (float): Probability of dropping a unit. Must be between 0 and 1.
        """
        super().__init__()
        if not (0 <= p < 1):
            raise ValueError("Dropout probability p must be in the range [0, 1).")
        self.p = p 
        self.mask = None
        self.training = True

    def forward(self, X: np.ndarray) -> np.ndarray:
        """
        Applies dropout to the input during training, or passes it unchanged during inference.

        Args:
            X (np.ndarray): Input array of shape (batch_size, features).

        Returns:
            np.ndarray: Output after applying dropout (during training) or original input (during inference).
        """
        if not self.training:
            return X  # No dropout during inference
        
        self.mask = (np.random.rand(*X.shape) > self.p).astype(np.float32)
        return (X * self.mask) / (1 - self.p)

    def backward(self, doutput: np.ndarray) -> np.ndarray:
        """
        Backward pass through the dropout layer. Only propagates gradients 
        through the neurons that were active during the forward pass.

        Args:
            doutput (np.ndarray): Upstream gradient.

        Returns:
            np.ndarray: Gradient of the loss with respect to the input.
        """
        if self.mask is None:
            raise ValueError("Must call forward() before backward().")
        
        return (doutput * self.mask) / (1 - self.p)
