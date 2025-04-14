import numpy as np

class Loss:
    """
    --------------------------------
    BASE CLASS FOR THE LOSS FUNCTION
    --------------------------------
    """

    def __init__(self):
        self.name = self.__class__.__name__

    def calculate(self, output, y):
        """
        The function calculates the mean loss across all samples in a batch based on the output and
        target values.
        """

        # loss calculated for each individual sample in the batch
        sample_losses = self.forward(output, y)

        # mean loss accross all the samples in the batch
        data_loss = np.mean(sample_losses)
        return data_loss
    
    def forward(self, y_pred, y_true):
        """
        Forward pass to calculate the loss

        """

        raise NotImplementedError
    
    def backward(self, dvalues, y_true):
        """
        Backward pass to calculate gradients
        
        """

        raise NotImplementedError