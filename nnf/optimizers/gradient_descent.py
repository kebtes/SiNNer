from nnf.optimizers.base import Optimizer
from nnf.layers.base import Layer

class GradientDescent(Optimizer):
    """
    --------------------------
    GRADIENT DESCENT OPTIMIZER
    --------------------------
    """
    def __init__(self, learning_rate = 0.1, decay=0):
        """
        Initialize Gradient Descent optimizer
        """
        super().__init__(learning_rate, decay)

    def update_params(self, layer : Layer):
        """
        The function updates the weights and biases of a neural network layer using the current learning
        rate.
        
        """

        layer.weights -= self.current_learning_rate * layer.dweights

        if layer.biases is not None:
            layer.biases -= self.current_learning_rate * layer.dbiases