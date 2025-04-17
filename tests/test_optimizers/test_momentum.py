import numpy as np
import pytest
from nnf.optimizers.momentum import Momentum

# Dummy layer class to mimic the real Layer object
class DummyLayer:
    def __init__(self, weights, biases=None):
        self.weights = weights.copy()
        self.biases = biases.copy() if biases is not None else None
        self.dweights = np.ones_like(weights)
        self.dbiases = np.ones_like(biases) if biases is not None else None


@pytest.fixture
def setup_optimizer():
    learning_rate = 0.1
    momentum = 0.9
    optimizer = Momentum(learning_rate=learning_rate, momentum=momentum)
    return optimizer


def test_velocity_initialization(setup_optimizer):
    optimizer = setup_optimizer
    layer = DummyLayer(weights=np.array([[1.0, 2.0]]), biases=np.array([[0.5, 0.5]]))
    optimizer.update_params(layer)

    assert layer in optimizer.velocities
    assert np.array_equal(optimizer.velocities[layer]['weights'], np.ones_like(layer.weights))
    assert np.array_equal(optimizer.velocities[layer]['biases'], np.ones_like(layer.biases))


def test_update_weights_and_biases(setup_optimizer):
    optimizer = setup_optimizer
    layer = DummyLayer(weights=np.array([[2.0, 2.0]]), biases=np.array([[1.0, 1.0]]))
    optimizer.update_params(layer)

    # Since velocities are initialized to zeros,
    # velocity after 1st update: v = 0 + dweights = 1
    # update: weights -= learning_rate * v
    expected_weights = np.array([[2.0, 2.0]]) - optimizer.learning_rate * np.ones((1, 2))
    expected_biases = np.array([[1.0, 1.0]]) - optimizer.learning_rate * np.ones((1, 2))

    np.testing.assert_array_almost_equal(layer.weights, expected_weights)
    np.testing.assert_array_almost_equal(layer.biases, expected_biases)


def test_layer_without_biases(setup_optimizer):
    optimizer = setup_optimizer
    layer = DummyLayer(weights=np.array([[1.0, -1.0]]), biases=None)
    optimizer.update_params(layer)

    assert 'weights' in optimizer.velocities[layer]
    assert optimizer.velocities[layer].get('biases') is None
    # Bias update should be skipped without error
    assert layer.biases is None


def test_different_shapes(setup_optimizer):
    optimizer = setup_optimizer
    weights = np.random.randn(4, 5)
    biases = np.random.randn(1, 5)
    layer = DummyLayer(weights=weights, biases=biases)
    optimizer.update_params(layer)

    assert optimizer.velocities[layer]['weights'].shape == weights.shape
    assert optimizer.velocities[layer]['biases'].shape == biases.shape
