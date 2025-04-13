import numpy as np
from nnf.losses.base import Loss

class MSE(Loss):
    """
    MEAN SQUARED ERROR LOSS FUNCTION.
    
    MSE IS COMMONLY USED FOR REGRESSION PROBLEMS. IT MEASURES THE AVERAGE
    SQUARED DIFFERENCE BETWEEN THE PREDICTED VALUES AND THE TARGET VALUES.
    
    FORMULA: MSE = (1/n) * Σ(y_pred - y_true)²
    
    """

    def forward(self, y_pred, y_true):
        """
        The function calculates the mean squared error between predicted and true values.
        """

        # calc the squared difference between the prediction and target values
        # return the loss per sample
        return np.mean((y_pred - y_true) ** 2, axis=1)

    def backward(self, dvalues, y_true):
        """
        The `backward` function calculates the gradient of the loss with respect to the inputs.
        """

        samples = len(dvalues) # number of samples
        outputs = len(dvalues[0]) # number fo outputs in every sample

        # gradient on values
        # 2 * (output - target) / outputs (outputs is for normalization)        # 
        self.dinputs = 2 * (dvalues - y_true) / outputs

        # normalize the gradient 
        # we want it to reflect for the entire batch
        self.dinputs = self.dinputs / samples

        return self.dinputs