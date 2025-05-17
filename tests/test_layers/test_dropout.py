import numpy as np
import pytest

from nnf.layers import Dropout

@pytest.fixture
def sample_input():
    return np.ones((4, 4), dtype=np.float32)

@pytest.fixture
def dropout_layer():
    return Dropout()

def test_forward_training_shape_and_scale(dropout_layer, sample_input):
    dropout_layer.training = True
    out = dropout_layer.forward(sample_input)

    assert out.shape == sample_input.shape

    valid_vals = [0.0, 1.0 / (1 - dropout_layer.p)]
    unique_vals = np.unique(out)
    for val in unique_vals:
        assert val in valid_vals

def test_forward_inference_no_change(dropout_layer, sample_input):
    dropout_layer.training = False
    out = dropout_layer.forward(sample_input)

    np.testing.assert_array_equal(out, sample_input)

def test_mask_is_binary(dropout_layer, sample_input):
    dropout_layer.training = True
    _ = dropout_layer.forward(sample_input)
    mask = dropout_layer.mask

    assert mask is not None
    assert mask.shape == sample_input.shape

    unique_vals = np.unique(mask)
    for val in unique_vals:
        assert val in [0.0, 1.0]

def test_backward_gradient_masking(dropout_layer, sample_input):
    dropout_layer.training = True
    _ = dropout_layer.forward(sample_input)

    upstream_grad = np.full_like(sample_input, 2.0)
    dx = dropout_layer.backward(upstream_grad)

    expected = (upstream_grad * dropout_layer.mask) / (1 - dropout_layer.p)
    np.testing.assert_array_almost_equal(dx, expected)

def test_backward_raises_without_forward():
    dropout = Dropout(p=0.5)
    upstream_grad = np.ones((4, 4), dtype=np.float32)

    with pytest.raises(ValueError):
        dropout.backward(upstream_grad)

@pytest.mark.parametrize("invalid_p", [-0.1, 1.0, 1.5])
def test_invalid_p_raises(invalid_p):
    with pytest.raises(ValueError):
        Dropout(p=invalid_p)