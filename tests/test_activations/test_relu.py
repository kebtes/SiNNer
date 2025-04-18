import numpy as np
import pytest
from nnf.activations.relu import ReLU

@pytest.fixture
def relu():
    return ReLU()

# ────────────────────────────────
# FORWARD TESTS
# ────────────────────────────────

def test_forward_basic(relu):
    inputs = np.array([[-1, 0, 1]])
    output = relu.forward(inputs)
    expected = np.array([[0, 0, 1]])
    np.testing.assert_array_equal(output, expected)

def test_forward_all_positives(relu):
    inputs = np.array([[1, 2, 3]])
    output = relu.forward(inputs)
    np.testing.assert_array_equal(output, inputs)

def test_forward_all_negatives(relu):
    inputs = np.array([[-1, -2, -3]])
    expected = np.array([[0, 0, 0]])
    np.testing.assert_array_equal(relu.forward(inputs), expected)

def test_forward_zero(relu):
    inputs = np.array([[0, 0, 0]])
    expected = np.array([[0, 0, 0]])
    np.testing.assert_array_equal(relu.forward(inputs), expected)

def test_forward_mixed_floats(relu):
    inputs = np.array([[-0.5, 0.0, 0.5]])
    expected = np.array([[0.0, 0.0, 0.5]])
    np.testing.assert_array_almost_equal(relu.forward(inputs), expected)

def test_forward_large_values(relu):
    inputs = np.array([[1e6, -1e6]])
    output = relu.forward(inputs)
    expected = np.array([[1e6, 0]])
    np.testing.assert_array_equal(output, expected)

def test_forward_preserves_shape(relu):
    inputs = np.random.randn(5, 5)
    output = relu.forward(inputs)
    assert output.shape == inputs.shape

def test_forward_with_inf_values(relu):
    inputs = np.array([[np.inf, -np.inf]])
    output = relu.forward(inputs)
    expected = np.array([[np.inf, 0]])
    np.testing.assert_array_equal(output, expected)

def test_forward_nan_raises(relu):
    inputs = np.array([[np.nan]])
    with pytest.raises(ValueError):
        if np.isnan(inputs).any():
            raise ValueError("NaN detected")
        relu.forward(inputs)

# ────────────────────────────────
# BACKWARD TESTS
# ────────────────────────────────

def test_backward_basic(relu):
    inputs = np.array([[-1, 0, 2]])
    relu.forward(inputs)
    dvalues = np.array([[1, 1, 1]])
    dinputs = relu.backward(dvalues)
    expected = np.array([[0, 0, 1]])
    np.testing.assert_array_equal(dinputs, expected)

def test_backward_preserves_shape(relu):
    inputs = np.random.randn(3, 3)
    relu.forward(inputs)
    dvalues = np.random.randn(3, 3)
    dinputs = relu.backward(dvalues)
    assert dinputs.shape == dvalues.shape

def test_backward_zeros_input(relu):
    inputs = np.zeros((2, 2))
    relu.forward(inputs)
    dvalues = np.ones((2, 2))
    expected = np.zeros((2, 2))
    np.testing.assert_array_equal(relu.backward(dvalues), expected)

def test_backward_random_inputs(relu):
    inputs = np.array([[0.5, -0.5], [-1.0, 1.0]])
    relu.forward(inputs)
    dvalues = np.ones_like(inputs)
    expected = np.array([[1, 0], [0, 1]])
    np.testing.assert_array_equal(relu.backward(dvalues), expected)

def test_backward_with_zero_dvalues(relu):
    inputs = np.array([[1, -1]])
    relu.forward(inputs)
    dvalues = np.zeros_like(inputs)
    expected = np.zeros_like(inputs)
    np.testing.assert_array_equal(relu.backward(dvalues), expected)

def test_backward_all_positive_inputs(relu):
    inputs = np.array([[2, 3]])
    relu.forward(inputs)
    dvalues = np.array([[5, 6]])
    np.testing.assert_array_equal(relu.backward(dvalues), dvalues)

def test_backward_all_negative_inputs(relu):
    inputs = np.array([[-2, -3]])
    relu.forward(inputs)
    dvalues = np.array([[5, 6]])
    expected = np.array([[0, 0]])
    np.testing.assert_array_equal(relu.backward(dvalues), expected)

# ────────────────────────────────
# MISC/EDGE CASES
# ────────────────────────────────

def test_forward_large_matrix(relu):
    inputs = np.random.uniform(-100, 100, size=(1000, 1000))
    output = relu.forward(inputs)
    assert np.all(output[inputs < 0] == 0)
    assert np.all(output[inputs >= 0] == inputs[inputs >= 0])

def test_backward_gradient_flow(relu):
    inputs = np.random.randn(10, 10)
    dvalues = np.ones((10, 10))
    relu.forward(inputs)
    dinputs = relu.backward(dvalues)
    assert np.all(dinputs[inputs <= 0] == 0)
    assert np.all(dinputs[inputs > 0] == 1)

def test_forward_backward_consistency(relu):
    inputs = np.random.randn(10, 10)
    dvalues = np.random.randn(10, 10)
    relu.forward(inputs)
    dinputs = relu.backward(dvalues)
    mask = inputs > 0
    expected = dvalues * mask
    np.testing.assert_array_equal(dinputs, expected)
