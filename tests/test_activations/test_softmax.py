import numpy as np
import pytest
from nnf.activations.softmax import Softmax

@pytest.fixture
def softmax():
    return Softmax()

def test_forward_output_range(softmax):
    inputs = np.array([[1.0, 2.0, 3.0],
                       [1.0, 2.0, 3.0]])
    output = softmax.forward(inputs)
    assert np.all(output >= 0) and np.all(output <= 1)
    row_sums = np.sum(output, axis=1)
    assert np.allclose(row_sums, 1.0)

def test_backward_gradient(softmax):
    inputs = np.array([[1.0, 2.0, 3.0],
                       [1.0, 2.0, 3.0]])
    softmax.forward(inputs)
    dvalues = np.array([[0.1, 0.2, 0.3],
                        [0.4, 0.5, 0.6]])
    dinputs = softmax.backward(dvalues)
    assert dinputs.shape == dvalues.shape
    np.testing.assert_array_equal(dinputs, dvalues)

def test_forward_numerical_stability(softmax):
    inputs = np.array([[1000, 1000, 1000],
                       [1000, 1000, 1000]])
    output = softmax.forward(inputs)
    assert np.all(output >= 0) and np.all(output <= 1)
    row_sums = np.sum(output, axis=1)
    assert np.allclose(row_sums, 1.0)

def test_invalid_input(softmax):
    inputs = np.array([[np.nan, 1.0, 2.0],
                       [1.0, np.inf, 3.0]])

    with pytest.raises(ValueError):
        softmax.forward(inputs)
