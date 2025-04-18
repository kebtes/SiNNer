from pathlib import Path
import numpy as np
import pandas as pd

from nnf.layers import Dense
from nnf.activations import ReLU, Softmax
from nnf.losses import CategoricalCrossEntropy
from nnf.optimizers import GradientDescent
from nnf.models import Model

np.random.seed(42)

def one_hot_encode(labels, num_classes=3):
    """
    One-hot encode the labels for multi-class classification.
    """
    one_hot = np.zeros((labels.size, num_classes))
    one_hot[np.arange(labels.size), labels] = 1
    return one_hot

def main():
    base_path = Path(__file__).resolve().parent.parent

    # Load Iris dataset
    data_set = base_path / "resources" / "iris" / "iris.csv"
    df = pd.read_csv(data_set)
    
    # Create a mapping for the string labels to integers
    label_mapping = {
        'Iris-setosa': 0,
        'Iris-versicolor': 1,
        'Iris-virginica': 2
    }
    
    df['numeric_label'] = df.iloc[:, -1].map(label_mapping)
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    # 80% training, 20% testing
    train_size = int(len(df) * 0.8)
    df_train = df[:train_size]
    df_test = df[train_size:]

    # Training sets
    X_train = df_train.iloc[:, :-2].values.astype(np.float32)  
    y_train = df_train['numeric_label'].values  
    y_train = one_hot_encode(y_train, 3)  

    # Test sets
    X_test = df_test.iloc[:, :-2].values.astype(np.float32)
    y_test = df_test['numeric_label'].values
    y_test = one_hot_encode(y_test, 3)  

    mean = np.mean(X_train, axis=0)
    std = np.std(X_train, axis=0)
    
    # Apply normalization to both sets using training statistics
    X_train = (X_train - mean) / (std + 1e-8)  
    X_test = (X_test - mean) / (std + 1e-8)

    model = Model(
        Dense(X_train.shape[1], 64),  
        ReLU(),
        Dense(64, 32),  
        ReLU(),
        Dense(32, 3),  
        Softmax(),
    )

    model.set(
        loss=CategoricalCrossEntropy(),
        optimizer=GradientDescent(learning_rate=0.1),  
    )

    model.train(X_train, y_train, epochs=500)

    print("\nMaking predictions:")

    predictions = model.forward(X_test)
    
    correct_pred = 0
    for pred, act in zip(predictions, y_test):
        predicted_label = np.argmax(pred)  
        true_label = np.argmax(act)  
        if predicted_label == true_label:
            correct_pred += 1

    accuracy = correct_pred / len(predictions)
    print(f"Accuracy: {accuracy:.4f}")
    
    # # Print detailed predictions for the first few samples
    # print("\nDetailed predictions for first 5 samples:")
    # for i in range(min(5, len(predictions))):
    #     pred = predictions[i]
    #     act = y_test[i]
    #     predicted_label = np.argmax(pred)
    #     true_label = np.argmax(act)
        
    #     # Convert back to flower names for clarity
    #     flower_names = {0: 'Iris-setosa', 1: 'Iris-versicolor', 2: 'Iris-virginica'}
    #     pred_name = flower_names[predicted_label]
    #     true_name = flower_names[true_label]
        
    #     print(f"Sample {i+1}:")
    #     print(f"  Prediction probabilities: {pred}")
    #     print(f"  Predicted: {pred_name} (class {predicted_label})")
    #     print(f"  Actual: {true_name} (class {true_label})")
    #     print(f"  Correct: {predicted_label == true_label}")
    #     print()

if __name__ == "__main__":
    main()