import unittest
import numpy as np
from nnf.losses.cross_entropy import CrossEntropy

class TestCrossEntropyLoss(unittest.TestCase):

    def setUp(self):
        self.loss_fn = CrossEntropy()

    def test_forward_loss(self):
        # Simulated softmax outputs
        y_pred = np.array([
            [0.7, 0.2, 0.1],
            [0.1, 0.8, 0.1],
            [0.2, 0.2, 0.6]
        ])
        # One-hot true labels
        y_true = np.array([
            [1, 0, 0],
            [0, 1, 0],
            [0, 0, 1]
        ])
        # Expected: -log(correct class prob) for each sample
        expected_losses = -np.log(np.array([0.7, 0.8, 0.6]))
        expected_mean_loss = np.mean(expected_losses)

        losses = self.loss_fn.forward(y_pred, y_true)
        mean_loss = np.mean(losses)

        np.testing.assert_array_almost_equal(losses, expected_losses)
        self.assertAlmostEqual(mean_loss, expected_mean_loss, places=6)

    def test_backward_gradient(self):
        y_pred = np.array([
            [0.3, 0.6, 0.1],
            [0.1, 0.1, 0.8]
        ])
        y_true = np.array([
            [0, 1, 0],
            [0, 0, 1]
        ])

        grads = self.loss_fn.backward(y_pred, y_true)

        # The expected gradient is: -y_true / y_pred / n_samples
        expected = -y_true / y_pred
        expected /= len(y_pred)  # batch size normalization

        np.testing.assert_array_almost_equal(grads, expected)

    def test_shape_consistency(self):
        y_pred = np.array([
            [0.25, 0.25, 0.5],
            [0.1, 0.8, 0.1]
        ])
        y_true = np.array([
            [0, 0, 1],
            [0, 1, 0]
        ])
        grads = self.loss_fn.backward(y_pred, y_true)

        self.assertEqual(grads.shape, y_pred.shape)

    def test_perfect_prediction_loss(self):
        y_pred = np.array([
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0]
        ])
        y_true = np.array([
            [1, 0, 0],
            [0, 1, 0],
            [0, 0, 1]
        ])

        losses = self.loss_fn.forward(y_pred, y_true)
        # Perfect prediction, log(1) = 0 → loss should be zero
        np.testing.assert_array_almost_equal(losses, np.zeros(3))

    def test_numerical_stability(self):
        # Probabilities extremely close to 0 for true class
        y_pred = np.array([
            [1e-15, 1.0 - 1e-15],
        ])
        y_true = np.array([
            [1, 0]
        ])
        loss = self.loss_fn.forward(y_pred, y_true)
        self.assertTrue(np.isfinite(loss).all())


if __name__ == '__main__':
    unittest.main()
