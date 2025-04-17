import numpy as np
from nnf.losses.base import Loss


class CategoricalCrossEntropy(Loss):
    """
    ----------------------------------------
    CATEGORICAL CROSS ENTROPY LOSS FUNCTION.
    ----------------------------------------

    COMMONLY USED FOR MULTI-CLASS CLASSIFICATION PROBLEMS WHERE LABELS 
    ARE ONE-HOT ENCODED.

    FORMULA: L = -Σ y_true * log(y_pred)
    """

    def forward(self, y_pred, y_true):
        """
        Calculates the categorical cross-entropy loss.
        """

        # Avoid log(0) by clipping predictions
        y_pred_clipped = np.clip(y_pred, 1e-7, 1 - 1e-7)

        # Calculate loss for each sample (assuming one-hot encoded y_true)
        sample_losses = -np.sum(y_true * np.log(y_pred_clipped), axis=1)
        return sample_losses

    def backward(self, dvalues, y_true):
        """
        Computes the gradient of the loss w.r.t. the predicted values.

        Derivative of Cross Entropy Loss (w.r.t. Softmax Output):
        Since only one y_i is 1, the derivative is:
            ∂L/∂ŷᵢ = {
                -1 / ŷᵢ, if yᵢ = 1
                 0       if yᵢ = 0
            }
        Or in one-shot for all classes:
            ∂L/∂ŷ = -y / ŷ
        """

        samples = len(dvalues)
        y_pred_clipped = np.clip(dvalues, 1e-7, 1 - 1e-7)

        self.dinputs = -y_true / y_pred_clipped
        self.dinputs = self.dinputs / samples

        return self.dinputs
