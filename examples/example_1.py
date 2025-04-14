"""
Binary Classification of MNIST Digits 0 and 1 Using a Custom Neural Network

This example demonstrates how to build and train a simple feedforward neural network
from scratch to classify handwritten digits using the MNIST dataset. Specifically, the
network is trained to distinguish between digits `0` and `1` (binary classification).

Key Steps:
----------
1. **Loading the Data**:
    - The script uses the MNIST dataset in CSV format (from `resources/mnist/`).
    - It loads both training and testing data using pandas.

2. **Preprocessing**:
    - The labels (`y_train` and `y_test`) are extracted as the first column.
    - The pixel values (`X_train`, `X_test`) are normalized to the range [0, 1].
    - Only examples labeled `0` or `1` are retained, turning this into a binary classification 
        problem.

3. **Model Architecture**:
    - The neural network has the following architecture:
        Input (784 nodes) →
        Dense(128) + ReLU →
        Dense(64) + ReLU  →
        Dense(32) + ReLU  →
        Dense(1) + Sigmoid
    - ReLU (Rectified Linear Unit) is used for hidden layers for non-linearity.
    - Sigmoid is used in the output layer to produce probabilities between 0 and 1.

4. **Training**:
    - The model uses Binary Cross Entropy as the loss function.
    - Gradient Descent is used as the optimizer with a learning rate of 0.01.
    - The model is trained for 500 epochs.

5. **Evaluation**:
    - After training, predictions are made on the test set.
    - Each prediction is thresholded at 0.5 to convert probabilities into binary labels.
    - The accuracy is computed and printed.

Then run the script using:
    $ python -m examples.example_1

"""

from pathlib import Path

import numpy as np
import pandas as pd

from nnf.layers import Dense
from nnf.activations import ReLU, Sigmoid
from nnf.losses import BinaryCrossEntropy
from nnf.optimizers import GradientDescent
from nnf.models import Model

np.random.seed(42)

def main():
    base_path = Path(__file__).resolve().parent.parent

    train_path = base_path / "resources" / "mnist" / "mnist_train.csv"
    test_path = base_path / "resources" / "mnist" / "mnist_test.csv"

    # Read CSVs
    df_train = pd.read_csv(train_path)
    df_test = pd.read_csv(test_path)

    # training sets
    X_train = df_train.iloc[:, 1:].values.astype(np.float32)
    y_train = df_train.iloc[:, 0].values.reshape(-1, 1)

    # test sets
    y_test = df_test.iloc[:, 0].values.reshape(-1, 1)
    X_test = df_test.iloc[:, 1:].values.astype(np.float32)

    # normalizations
    X_test /= 255.0
    X_train /= 255.0

    # only take 0 and 1 since this is a binary classification
    mask = (y_train == 0) | (y_train == 1)
    X_train = X_train[mask.flatten()]
    y_train = y_train[mask.flatten()]

    mask = (y_test == 0) | (y_test == 1)
    X_test = X_test[mask.flatten()]
    y_test = y_test[mask.flatten()]

    model = Model(
        Dense(X_train.shape[1], 128),
        ReLU(),
        Dense(128, 64),
        ReLU(),
        Dense(64, 32),
        ReLU(),
        Dense(32, 1),
        Sigmoid(),
    )

    model.set(
        loss=BinaryCrossEntropy(),
        optimizer=GradientDescent(learning_rate=0.01, decay=None),
    )

    model.train(X_train, y_train, epochs=100)

    print("\nMaking predictions:")
    predictions = model.forward(X_test)

    correct_pred = 0
    for pred, act in zip(predictions, y_test):
        predicted_label = int(pred[0] > 0.5)
        print(f"True: {int(act[0])}, Predicted: {predicted_label} ({pred[0]:.4f})")

        if predicted_label == act[0]:
            correct_pred += 1

    print(f"Accuracy: {correct_pred / len(predictions):.4f}")


if __name__ == "__main__":
    main()
