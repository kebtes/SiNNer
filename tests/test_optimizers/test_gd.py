import numpy as np
import pytest
from nnf.optimizers import GradientDescent

class DummyLayer:
    def __init__(self, weights, biases=None):
        self.weights = weights
        self.biases = biases
        self.dweights = np.zeros_like(weights)
        self.dbiases = np.zeros_like(biases) if biases is not None else None

def test_update_weights_and_biases():
    initial_weights = np.array([[0.5, -0.5], [0.3, 0.3]])
    initial_biases = np.array([0.1, -0.1])
    layer = DummyLayer(initial_weights, initial_biases)
    layer.dweights = np.array([[0.1, -0.1], [-0.1, 0.1]])
    layer.dbiases = np.array([0.05, -0.05])
    optimizer = GradientDescent(learning_rate=0.1)
    optimizer.update_params(layer)
    expected_weights = initial_weights - 0.1 * layer.dweights
    expected_biases = initial_biases - 0.1 * layer.dbiases
    np.testing.assert_array_almost_equal(layer.weights, expected_weights)
    np.testing.assert_array_almost_equal(layer.biases, expected_biases)

def test_update_weights_and_biases():
    initial_weights = np.array([[0.5, -0.5], [0.3, 0.3]])
    initial_biases = np.array([0.1, -0.1])
    layer = DummyLayer(initial_weights, initial_biases)
    layer.dweights = np.array([[0.1, -0.1], [-0.1, 0.1]])
    layer.dbiases = np.array([0.05, -0.05])
    optimizer = GradientDescent(learning_rate=0.1)
    optimizer.update_params(layer)
    expected_weights = np.array([[0.49, -0.49], [0.31, 0.29]])
    expected_biases = np.array([0.095, -0.095])
    np.testing.assert_array_almost_equal(layer.weights, expected_weights, decimal=5)
    np.testing.assert_array_almost_equal(layer.biases, expected_biases, decimal=5)

def test_zero_gradients():
    initial_weights = np.array([[0.5, -0.5], [0.3, 0.3]])
    initial_biases = np.array([0.1, -0.1])
    layer = DummyLayer(initial_weights, initial_biases)
    layer.dweights = np.zeros_like(initial_weights)
    layer.dbiases = np.zeros_like(initial_biases)
    optimizer = GradientDescent(learning_rate=0.1)
    optimizer.update_params(layer)
    np.testing.assert_array_almost_equal(layer.weights, initial_weights)
    np.testing.assert_array_almost_equal(layer.biases, initial_biases)


