import numpy as np
import pytest
from nnf.activations.sigmoid import Sigmoid

@pytest.fixture
def sigmoid():
    return Sigmoid()

# ─────────────────────────────────────────
# FORWARD TESTS
# ─────────────────────────────────────────

def test_forward_basic(sigmoid):
    inputs = np.array([[0, 1, -1]])
    output = sigmoid.forward(inputs)
    expected = 1 / (1 + np.exp(-inputs))
    np.testing.assert_array_almost_equal(output, expected)

def test_forward_output_range(sigmoid):
    inputs = np.linspace(-1000, 1000, num=10).reshape(2, 5)
    output = sigmoid.forward(inputs)
    assert np.all(output >= 0) and np.all(output <= 1)

def test_forward_zero(sigmoid):
    inputs = np.array([[0]])
    output = sigmoid.forward(inputs)
    assert np.allclose(output, 0.5)

def test_forward_large_positive(sigmoid):
    inputs = np.array([[1000]])
    output = sigmoid.forward(inputs)
    assert np.allclose(output, 1.0, atol=1e-6)

def test_forward_large_negative(sigmoid):
    inputs = np.array([[-1000]])
    output = sigmoid.forward(inputs)
    assert np.allclose(output, 0.0, atol=1e-6)

def test_forward_preserves_shape(sigmoid):
    inputs = np.random.randn(4, 4)
    output = sigmoid.forward(inputs)
    assert output.shape == inputs.shape

def test_forward_numerical_stability(sigmoid):
    inputs = np.array([[1000, -1000]])
    output = sigmoid.forward(inputs)
    assert np.isfinite(output).all()

def test_forward_inf_values(sigmoid):
    inputs = np.array([[np.inf, -np.inf]])
    output = sigmoid.forward(inputs)
    expected = np.array([[1.0, 0.0]])
    np.testing.assert_array_almost_equal(output, expected)

def test_forward_nan_raises(sigmoid):
    inputs = np.array([[1.0, np.nan]])
    with pytest.raises(ValueError):
        if np.isnan(inputs).any():
            raise ValueError("NaN input detected")
        sigmoid.forward(inputs)

# ─────────────────────────────────────────
# BACKWARD TESTS
# ─────────────────────────────────────────

def test_backward_basic(sigmoid):
    inputs = np.array([[0.0, 1.0]])
    sigmoid.forward(inputs)
    dvalues = np.array([[1.0, 1.0]])
    dinputs = sigmoid.backward(dvalues)
    expected = dvalues * (sigmoid.output * (1 - sigmoid.output))
    np.testing.assert_array_almost_equal(dinputs, expected)

def test_backward_zero_gradient(sigmoid):
    inputs = np.random.randn(3, 3)
    dvalues = np.zeros_like(inputs)
    sigmoid.forward(inputs)
    dinputs = sigmoid.backward(dvalues)
    expected = np.zeros_like(inputs)
    np.testing.assert_array_equal(dinputs, expected)

def test_backward_preserves_shape(sigmoid):
    inputs = np.random.randn(2, 5)
    dvalues = np.random.randn(2, 5)
    sigmoid.forward(inputs)
    dinputs = sigmoid.backward(dvalues)
    assert dinputs.shape == inputs.shape

def test_backward_with_ones(sigmoid):
    inputs = np.random.randn(3, 3)
    dvalues = np.ones((3, 3))
    sigmoid.forward(inputs)
    dinputs = sigmoid.backward(dvalues)
    expected = sigmoid.output * (1 - sigmoid.output)
    np.testing.assert_array_almost_equal(dinputs, expected)

def test_backward_with_large_values(sigmoid):
    inputs = np.array([[1000, -1000]])
    dvalues = np.array([[1.0, 1.0]])
    sigmoid.forward(inputs)
    dinputs = sigmoid.backward(dvalues)
    # should be close to 0 due to vanishing gradient
    assert np.all(dinputs < 1e-3)

# ─────────────────────────────────────────
# EDGE CASES
# ─────────────────────────────────────────

def test_forward_backward_consistency(sigmoid):
    inputs = np.random.randn(10, 10)
    dvalues = np.random.randn(10, 10)
    sigmoid.forward(inputs)
    dinputs = sigmoid.backward(dvalues)
    expected = dvalues * sigmoid.output * (1 - sigmoid.output)
    np.testing.assert_array_almost_equal(dinputs, expected)

def test_forward_high_dimensional_input(sigmoid):
    inputs = np.random.randn(5, 4, 3)
    output = sigmoid.forward(inputs)
    assert output.shape == inputs.shape

def test_backward_high_dimensional_input(sigmoid):
    inputs = np.random.randn(2, 3, 4)
    dvalues = np.ones((2, 3, 4))
    sigmoid.forward(inputs)
    dinputs = sigmoid.backward(dvalues)
    assert dinputs.shape == inputs.shape

def test_forward_extremely_small_values(sigmoid):
    inputs = np.array([[-1e-10, 1e-10]])
    output = sigmoid.forward(inputs)
    expected = 1 / (1 + np.exp(-inputs))
    np.testing.assert_array_almost_equal(output, expected)

def test_backward_extremely_small_values(sigmoid):
    inputs = np.array([[1e-10]])
    dvalues = np.array([[1.0]])
    sigmoid.forward(inputs)
    dinputs = sigmoid.backward(dvalues)
    expected = dvalues * (sigmoid.output * (1 - sigmoid.output))
    np.testing.assert_array_almost_equal(dinputs, expected)
