import numpy as np
import pytest
from nnf.losses.categorical_cross_entropy import CategoricalCrossEntropy

@pytest.fixture
def loss_fn():
    return CategoricalCrossEntropy()

def test_forward_loss(loss_fn):
    y_pred = np.array([
        [0.7, 0.2, 0.1],
        [0.1, 0.8, 0.1],
        [0.2, 0.2, 0.6]
    ])
    y_true = np.array([
        [1, 0, 0],
        [0, 1, 0],
        [0, 0, 1]
    ])
    expected_losses = -np.log(np.array([0.7, 0.8, 0.6]))
    expected_mean_loss = np.mean(expected_losses)

    losses = loss_fn.forward(y_pred, y_true)
    mean_loss = np.mean(losses)

    np.testing.assert_array_almost_equal(losses, expected_losses)
    assert abs(mean_loss - expected_mean_loss) < 1e-6

def test_backward_gradient(loss_fn):
    y_pred = np.array([
        [0.3, 0.6, 0.1],
        [0.1, 0.1, 0.8]
    ])
    y_true = np.array([
        [0, 1, 0],
        [0, 0, 1]
    ])

    grads = loss_fn.backward(y_pred, y_true)
    expected = -y_true / y_pred
    expected /= len(y_pred)

    np.testing.assert_array_almost_equal(grads, expected)

def test_shape_consistency(loss_fn):
    y_pred = np.array([
        [0.25, 0.25, 0.5],
        [0.1, 0.8, 0.1]
    ])
    y_true = np.array([
        [0, 0, 1],
        [0, 1, 0]
    ])
    grads = loss_fn.backward(y_pred, y_true)

    assert grads.shape == y_pred.shape

def test_perfect_prediction_loss(loss_fn):
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

    losses = loss_fn.forward(y_pred, y_true)
    np.testing.assert_array_almost_equal(losses, np.zeros(3))

def test_numerical_stability(loss_fn):
    y_pred = np.array([
        [1e-15, 1.0 - 1e-15],
    ])
    y_true = np.array([
        [1, 0]
    ])
    loss = loss_fn.forward(y_pred, y_true)
    assert np.isfinite(loss).all()
