import numpy as np
from nnf.optimizers.base import Optimizer
from nnf.layers.base import Layer


class Momentum(Optimizer):
    """
    -----------------------------------------------
    MOMENTUM OPTIMIZER IMPLEMENTATION
    -----------------------------------------------
    This class implements the Momentum optimizer, originally introduced by Boris Polyak.
    It helps accelerate gradient descent by incorporating a velocity term that accumulates
    past gradients in order to smooth out updates and speed up convergence, especially
    in the presence of high curvature, small but consistent gradients, or noisy gradients.

    This implementation uses Polyak's method of updating parameters based on accumulated velocity.
    """

    def __init__(self, learning_rate: int = 0.1, decay: int = 0, momentum: int = 0):
        """
        Initializes the Momentum optimizer with the given learning rate, decay, and momentum values.

        Args:
            learning_rate (int, optional): The learning rate for parameter updates. Defaults to 0.1.
            decay (int, optional): Learning rate decay factor (not used here). Defaults to 0.
            momentum (int, optional): The momentum coefficient (typically between 0 and 1). Defaults to 0.
        """
        super().__init__(learning_rate, decay)
        self.momentum = momentum
        self.velocities = {}

    def update_params(self, layer: Layer):
        """
        Updates the weights and biases of the provided layer using the momentum optimization technique.

        The update rule follows Polyak's method:
            velocity = momentum * previous_velocity + current_gradient
            parameter -= learning_rate * velocity

        Args:
            layer (Layer): The layer whose parameters are to be updated.
        """
        # Initialize velocities if not already done for this layer
        if layer not in self.velocities:
            self.velocities[layer] = {
                "weights": np.zeros_like(layer.weights),
                "biases": (
                    np.zeros_like(layer.biases) if layer.biases is not None else None
                ),
            }

        # Compute and apply momentum update for weights
        weight_velocity = (
            self.momentum * self.velocities[layer]["weights"] + layer.dweights
        )
        layer.weights -= self.learning_rate * weight_velocity
        self.velocities[layer]["weights"] = weight_velocity

        # Compute and apply momentum update for biases if they exist
        if layer.biases is not None:
            bias_velocity = (
                self.momentum * self.velocities[layer]["biases"] + layer.dbiases
            )
            layer.biases -= self.learning_rate * bias_velocity
            self.velocities[layer]["biases"] = bias_velocity
