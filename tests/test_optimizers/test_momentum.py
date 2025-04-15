import unittest
import numpy as np
from nnf.optimizers.momentum import Momentum

# Dummy layer class to mimic the real Layer object
class DummyLayer:
    def __init__(self, weights, biases=None):
        self.weights = weights.copy()
        self.biases = biases.copy() if biases is not None else None
        self.dweights = np.ones_like(weights)
        self.dbiases = np.ones_like(biases) if biases is not None else None

class TestMomentumOptimizer(unittest.TestCase):

    def setUp(self):
        self.learning_rate = 0.1
        self.momentum = 0.9
        self.optimizer = Momentum(learning_rate=self.learning_rate, momentum=self.momentum)

    def test_velocity_initialization(self):
        layer = DummyLayer(weights=np.array([[1.0, 2.0]]), biases=np.array([[0.5, 0.5]]))
        self.optimizer.update_params(layer)
        self.assertIn(layer, self.optimizer.velocities)
        self.assertTrue(np.array_equal(self.optimizer.velocities[layer]['weights'], np.ones_like(layer.weights)))
        self.assertTrue(np.array_equal(self.optimizer.velocities[layer]['biases'], np.ones_like(layer.biases)))

    def test_update_weights_and_biases(self):
        layer = DummyLayer(weights=np.array([[2.0, 2.0]]), biases=np.array([[1.0, 1.0]]))
        self.optimizer.update_params(layer)

        # Since velocities are initialized to zeros,
        # velocity after 1st update: v = 0 + dweights = 1
        # update: weights -= learning_rate * v
        expected_weights = np.array([[2.0, 2.0]]) - self.learning_rate * np.ones((1, 2))
        expected_biases = np.array([[1.0, 1.0]]) - self.learning_rate * np.ones((1, 2))

        np.testing.assert_array_almost_equal(layer.weights, expected_weights)
        np.testing.assert_array_almost_equal(layer.biases, expected_biases)
        
    def test_layer_without_biases(self):
        layer = DummyLayer(weights=np.array([[1.0, -1.0]]), biases=None)
        self.optimizer.update_params(layer)

        self.assertIn('weights', self.optimizer.velocities[layer])
        self.assertIsNone(self.optimizer.velocities[layer]['biases'])

        # Bias update should be skipped without error
        self.assertIsNone(layer.biases)

    def test_different_shapes(self):
        weights = np.random.randn(4, 5)
        biases = np.random.randn(1, 5)
        layer = DummyLayer(weights=weights, biases=biases)
        self.optimizer.update_params(layer)

        self.assertEqual(self.optimizer.velocities[layer]['weights'].shape, weights.shape)
        self.assertEqual(self.optimizer.velocities[layer]['biases'].shape, biases.shape)


if __name__ == '__main__':
    unittest.main()
