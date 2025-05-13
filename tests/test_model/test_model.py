import numpy as np
import pytest
import tempfile
import os
import json

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
    model.name = "TestModel"
    model.clip_value = 1.0
    model.shuffle = False
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

def test_get_model_attrs(simple_model):
    attrs = simple_model.get_model_attrs()

    assert isinstance(attrs, dict)
    assert attrs["name"] == "TestModel"
    assert attrs["loss"] == "MSE"
    assert isinstance(attrs["optimizer"], dict)
    assert attrs["clip_value"] == 1.0
    assert attrs["shuffle"] is False

def test_set_model_attrs_preserves_values(simple_model):
    attrs = simple_model.get_model_attrs()
    model = Model()
    model.set_model_attrs(attrs)
    
    assert model.name == "TestModel"
    assert model.loss.name == "MSE"
    assert isinstance(model.loss, MSE)
    assert model.optimizer.get_params() == attrs["optimizer"]
    assert model.clip_value == 1.0
    assert model.shuffle is False

def test_save_and_load_model(simple_model):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".json") as temp_file:
        path = temp_file.name

    try:
        simple_model.save_model(file_path=path)
        assert os.path.exists(path), "Model file was not created"

        with open(path) as f:
            saved = json.load(f)
            assert isinstance(saved, list) and "model" in saved[0]

        loaded_model = Model.load_model(path)

        assert isinstance(loaded_model, Model)
        assert loaded_model.name == simple_model.name
        assert loaded_model.loss.name == simple_model.loss.name
        assert loaded_model.optimizer.get_params() == simple_model.optimizer.get_params()
        assert loaded_model.clip_value == simple_model.clip_value
        assert loaded_model.shuffle == simple_model.shuffle
        assert len(loaded_model.layers) == len(simple_model.layers)
    finally:
        os.remove(path)

def test_default_model_path_creates_file(simple_model):
    # Force no file_path
    simple_model.name = None

    simple_model.save_model()  # Should trigger default path creation

    saved_dir = "saved_models"
    files = os.listdir(saved_dir)
    model_files = [f for f in files if f.startswith("model_") and f.endswith(".json")]

    assert len(model_files) > 0, "Default model file was not created"

    for f in model_files:
        os.remove(os.path.join(saved_dir, f))

def test_load_model_file_not_found():
    with pytest.raises(FileNotFoundError):
        Model.load_model("non_existent_model.json")

def test_load_model_invalid_json(tmp_path):
    bad_file = tmp_path / "invalid.json"
    bad_file.write_text("{not valid json}")

    with pytest.raises(json.JSONDecodeError):
        Model.load_model(str(bad_file))