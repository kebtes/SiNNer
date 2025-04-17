"""
Multi-Class Classification of MNIST Digits (0-9) Using a Custom Neural Network

This example demonstrates how to build and train a simple feedforward neural network
from scratch to classify handwritten digits using the MNIST dataset. Specifically, the
network is trained to distinguish between digits `0` through `9` (multi-class classification).

Key Steps:
----------
1. **Loading the Data**:
    - The script uses the MNIST dataset in CSV format (from `resources/mnist/`).
    - It loads both training and testing data using pandas.

2. **Preprocessing**:
    - The labels (`y_train` and `y_test`) are extracted as the first column.
    - The pixel values (`X_train`, `X_test`) are normalized to the range [0, 1].
    - One-hot encoding is applied to the labels for multi-class classification.

3. **Model Architecture**:
    - The neural network has the following architecture:
        Input (784 nodes) →
        Dense(128) + ReLU →
        Dense(64) + ReLU →
        Dense(32) + ReLU →
        Dense(10) + Softmax
    - ReLU (Rectified Linear Unit) is used for hidden layers for non-linearity.
    - Softmax is used in the output layer to produce probabilities for each of the 10 classes.

4. **Training**:
    - The model uses Categorical Cross Entropy as the loss function.
    - Gradient Descent is used as the optimizer with a learning rate of 0.01.
    - The model is trained for 100 epochs.

5. **Evaluation**:
    - After training, predictions are made on the test set.
    - The predicted class is chosen based on the highest probability from the output of the softmax.
    - The accuracy is computed and printed.

Then run the script using:
    $ python -m examples.example_2
"""

from pathlib import Path

import numpy as np
import pandas as pd

from nnf.layers import Dense
from nnf.activations import ReLU, Softmax
from nnf.losses import CategoricalCrossEntropy
from nnf.optimizers import GradientDescent, Momentum
from nnf.models import Model

np.random.seed(42)

def one_hot_encode(labels, num_classes=10):
    """
    One-hot encode the labels for multi-class classification.
    """

    one_hot = np.zeros((labels.size, num_classes))
    one_hot[np.arange(labels.size), labels] = 1
    return one_hot

def main():
    base_path = Path(__file__).resolve().parent.parent

    train_path = base_path / "resources" / "mnist" / "mnist_train.csv"
    test_path = base_path / "resources" / "mnist" / "mnist_test.csv"

    # Read CSVs
    df_train = pd.read_csv(train_path)[:500]
    df_test = pd.read_csv(test_path)[:500]

    # training sets
    X_train = df_train.iloc[:, 1:].values.astype(np.float32)
    y_train = df_train.iloc[:, 0].values.reshape(-1, 1)

    # test sets
    y_test = df_test.iloc[:, 0].values.reshape(-1, 1)
    X_test = df_test.iloc[:, 1:].values.astype(np.float32)

    # normalizations
    X_test /= 255.0
    X_train /= 255.0

    # One-hot encode labels
    # the arrays are two dimenstional so we need to flatten them out
    # before we pass them to the method
    y_train = one_hot_encode(y_train.flatten(), num_classes=10)
    y_test = one_hot_encode(y_test.flatten(), num_classes=10)


    # Build the model
    model = Model(
        Dense(X_train.shape[1], 64),
        ReLU(),
        Dense(64, 32),
        ReLU(),
        Dense(32, 10),  # 10 output neurons for digits 0-9
        Softmax(),
    )

    print(y_train)
    
    # Set loss and optimizer
    model.set(
        loss=CategoricalCrossEntropy(),
        optimizer=Momentum(learning_rate=0.00001, decay=None),
    )

    # Train the model
    model.train(X_train, y_train, epochs=1000, batch_size=12)

    print("\nMaking predictions:")

    predictions = model.forward(X_test)
    
    correct_pred = 0
    for pred, act in zip(predictions, y_test):
        # take the index of the highest probability

        predicted_label = np.argmax(pred)  
        # one-hot encoded, so we use argmax to get the true label

        true_label = np.argmax(act)  
        if predicted_label == true_label:
            correct_pred += 1

    print(f"Accuracy: {correct_pred / len(predictions):.4f}")


if __name__ == "__main__":
    main()
