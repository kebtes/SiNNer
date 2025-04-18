import numpy as np
import pytest
from nnf.losses.binary_cross_entropy import BinaryCrossEntropy


def test_initialization():
    bce = BinaryCrossEntropy()
    assert bce.output is None
    assert bce.dinputs is None


def test_forward_perfect_prediction():
    bce = BinaryCrossEntropy()
    y_true = np.array([[1], [0], [1], [0]])
    y_pred = np.array([[1], [0], [1], [0]])
    
    # Due to clipping, perfect prediction will not give exactly 0
    loss = bce.forward(y_pred, y_true)
    assert np.isclose(loss, 0, atol=1e-6)


def test_forward_worst_prediction():
    bce = BinaryCrossEntropy()
    y_true = np.array([[1], [0], [1], [0]])
    y_pred = np.array([[0], [1], [0], [1]])
    
    # Due to clipping, these will not be exactly 0 and 1
    y_pred = np.clip(y_pred, 1e-7, 1 - 1e-7)
    
    loss = bce.forward(y_pred, y_true)
    # Loss should be very high for worst predictions
    assert loss > 10


def test_forward_medium_prediction():
    bce = BinaryCrossEntropy()
    y_true = np.array([[1], [0]])
    y_pred = np.array([[0.7], [0.3]])
    
    expected_loss = -(1 * np.log(0.7) + (1 - 1) * np.log(1 - 0.7) + 
                      0 * np.log(0.3) + (1 - 0) * np.log(1 - 0.3)) / 2
    
    loss = bce.forward(y_pred, y_true)
    assert np.isclose(loss, expected_loss)


def test_backward():
    bce = BinaryCrossEntropy()
    y_true = np.array([[1], [0]])
    y_pred = np.array([[0.7], [0.3]])
    
    bce.forward(y_pred, y_true)  # Call forward first to simulate normal usage
    gradients = bce.backward(y_pred, y_true)
    
    # Calculate expected gradients manually
    samples = len(y_pred)
    expected_gradients = -(y_true / y_pred - (1 - y_true) / (1 - y_pred)) / samples
    
    assert np.allclose(gradients, expected_gradients)


def test_backward_shape():
    bce = BinaryCrossEntropy()
    batch_size = 32
    feature_size = 1
    
    y_true = np.random.randint(0, 2, size=(batch_size, feature_size))
    y_pred = np.random.random(size=(batch_size, feature_size))
    
    gradients = bce.backward(y_pred, y_true)
    
    assert gradients.shape == y_pred.shape


def test_output_range():
    bce = BinaryCrossEntropy()
    y_true = np.array([[1], [0], [1], [0]])
    
    # Test with various prediction probabilities
    for _ in range(10):
        y_pred = np.random.random(size=(4, 1))
        loss = bce.forward(y_pred, y_true)
        
        # BCE loss should always be positive
        assert loss >= 0