"""
Binary Classification of XOR Problem Using a Custom Neural Network

This example demonstrates how to build and train a simple feedforward neural network
from scratch to solve the XOR problem, a classic binary classification task.

Key Steps:
----------
1. **Defining the Problem**:
    - The XOR problem involves classifying pairs of binary input values (0 or 1) into one of two possible outputs.
    - The goal is to map the following input-output pairs:
        - (0, 0) -> 0
        - (0, 1) -> 1
        - (1, 0) -> 1
        - (1, 1) -> 0

2. **Model Architecture**:
    - The neural network has the following architecture:
        Input (2 nodes) → 
        Dense(4) + ReLU → 
        Dense(1) + Sigmoid
    - ReLU (Rectified Linear Unit) is used for hidden layers to introduce non-linearity.
    - Sigmoid is used in the output layer to produce probabilities between 0 and 1 for binary classification.

3. **Training**:
    - The model uses Binary Cross Entropy as the loss function.
    - Gradient Descent is used as the optimizer with a learning rate of 0.1.
    - The model is trained for 10,000 epochs to ensure convergence on the XOR problem.

4. **Evaluation**:
    - After training, predictions are made on the same XOR dataset (since it's a small problem).
    - Each prediction is thresholded at 0.5 to classify as either 0 or 1.
    - The accuracy is computed and printed.

Then run the script using:
    $ python -m examples.example_xor
"""

from pathlib import Path
import numpy as np

from nnf.layers import Dense
from nnf.activations import ReLU, Sigmoid
from nnf.losses import BinaryCrossEntropy
from nnf.optimizers import GradientDescent
from nnf.models import Model

np.random.seed(42)

def main():
    # XOR input and output
    X = np.array([[0, 0],
                  [0, 1],
                  [1, 0],
                  [1, 1]])
    
    y = np.array([[0],
                  [1],
                  [1],
                  [0]])
    
    model = Model(
        Dense(X.shape[1], 4),    
        ReLU(),                  
        Dense(4, 1),             
        Sigmoid(),               
    )
    
    model.set(
        loss=BinaryCrossEntropy(),  
        optimizer=GradientDescent(learning_rate=0.1, decay=0.001),
    )
    
    model.train(X, y, epochs=10000, batch_size=32)
    
    predictions = model.forward(X)

    # Calculate accuracy
    correct_pred = 0
    for pred, act in zip(predictions, y):
        predicted_label = 1 if pred >= 0.5 else 0  
        true_label = act
        if predicted_label == true_label:
            correct_pred += 1
    
    accuracy = correct_pred / len(y)
    print(f"\nAccuracy: {accuracy * 100:.2f}%")

if __name__ == "__main__":
    main()
