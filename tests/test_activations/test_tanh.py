import numpy as np
import pytest
from nnf.activations.tanh import Tanh

@pytest.fixture
def tanh():
    return Tanh()

def test_forward_output_range(tanh):
    inputs = np.array([[0.0, 1.0, -1.0],
                       [2.0, -2.0, 0.5]])
    output = tanh.forward(inputs)
    assert np.all(output >= -1) and np.all(output <= 1)

def test_backward_gradient(tanh):
    inputs = np.array([[0.0, 1.0, -1.0],
                       [2.0, -2.0, 0.5]])
    tanh.forward(inputs)
    dvalues = np.array([[0.1, 0.2, 0.3],
                        [0.4, 0.5, 0.6]])
    dinputs = tanh.backward(dvalues)
    expected_dinputs = dvalues * (1 - tanh.output ** 2)

    assert dinputs.shape == dvalues.shape
    np.testing.assert_array_equal(dinputs, expected_dinputs)

def test_forward_numerical_stability(tanh):
    inputs = np.array([[1000, 1000, 1000],
                       [-1000, -1000, -1000]])
    output = tanh.forward(inputs)
    assert np.all(output >= -1) and np.all(output <= 1)
    assert np.isfinite(output).all()

def test_invalid_input(tanh):
    inputs = np.array([[np.nan, 1.0, 2.0],
                       [1.0, np.inf, 3.0]])

    with pytest.raises(ValueError):
        tanh.forward(inputs)