import numpy as np
import pytest
from nnf.layers.dense import Dense
from nnf.losses import MSE
from nnf.optimizers.gradient_descent import GradientDescent
from nnf.models import Model
from nnf.activations import ReLU, Sigmoid

@pytest.fixture
def mock_data():
    X = np.random.randn(100, 3)  
    y = np.random.randn(100, 1)  
    return X, y

@pytest.fixture
def simple_model():
    model = Model(
        Dense(3, 5),
        ReLU(),
        Dense(5, 1),
        Sigmoid()
    )
    loss = MSE()  
    optimizer = GradientDescent(learning_rate=0.01)
    model.set(loss, optimizer)
    return model

def test_train_and_predict(mock_data, simple_model):
    X, y = mock_data
    model = simple_model

    model.train(X, y, epochs=1, batch_size=32)

    predictions = model.predict(X)

    assert predictions.shape == (X.shape[0], 1), f"Expected prediction shape: {(X.shape[0], 1)}, but got: {predictions.shape}"

    initial_loss = model.loss.calculate(model.forward(X), y)
    final_loss = model.loss.calculate(predictions, y)
    assert final_loss <= initial_loss, "Model did not reduce the loss during training"

# def test_model_summary(simple_model):
#     model = simple_model

#     # Capture the output of the summary
#     from io import StringIO
#     import sys

#     # Redirect stdout to capture print output
#     captured_output = StringIO()
#     sys.stdout = captured_output

#     # Call the summary method
#     model.summary()

#     # Check if the summary includes expected information
#     assert "Total Layers: 2" in captured_output.getvalue(), "Model summary does not include total layers"
#     assert "Total parameters" in captured_output.getvalue(), "Model summary does not include total parameters"
    
#     # Reset redirect.
#     sys.stdout = sys.__stdout__
