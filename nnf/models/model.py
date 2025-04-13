import numpy as np
import math
from typing import List
from nnf.layers.base import Layer
from nnf.losses.base import Loss
from nnf.optimizers.base import Optimizer

class Model:
    """
    --------------------------------------------------------------
    NEURAL NETWORK MODEL THAT COMBINES LAYERS AND HANDLES TRAINING.
    --------------------------------------------------------------
    """

    def __init__(self, *layers: Layer):
        self.layers     : List[Layer] = list(layers)
        self.loss       : Loss = None
        self.optimizer  : Optimizer = None

    def set(self, loss : Loss, optimzer: Optimizer):
        """
        Set the loss function and optimizer for the model.

        """
        self.loss = loss
        self.optimizer = optimzer

    def forward(self, X):
        for layer in self.layers:
            X = layer.forward(X)
        
        return X
    
    def backward(self, output, y):
        dinputs = self.loss.backward(output, y)

        self.layers[-1].backward(dinputs)
    
        for i in reversed(range(len(self.layers) -1)):
            self.layers[i].backward(self.layers[i + 1].dinputs)

    def train(self, X, y, *, epochs=1, batch_size : int=None):
        """
        Train the model using the provided data.
        
        """
        
        # set the batch_size to the size of input X if not set
        if batch_size is None:
            batch_size = len(X)

        # calc the number of steps
        steps = math.ceil(len(X) / batch_size)

        for epoch in range(1, epochs + 1):
            # reset accumulated loss value
            loss = 0 

            for step in range(steps):
                # calc the starting and ending of the batch
                batch_start = step * batch_size
                batch_end = min(batch_start + batch_size, len(X))

                # get the batch of the data
                batch_X = X[batch_start:batch_end]
                batch_y = y[batch_start:batch_end]
                
                # make a forward pass
                output = self.forward(batch_X)

                # calc the loss
                data_loss = self.loss.calculate(output, batch_y)
                loss += data_loss

                self.backward(output, batch_y)
                
                # update params for all trainable layers
                for layer in self.layers:
                    if hasattr(layer, "weights"):
                        self.optimizer.update_params(layer)
                
                # optimizer post-update (incremement iterations)
                self.optimizer.post_update_params() 

            print(f"Epoch: {epoch}, Loss: {loss}")
        
    def predict(self, X):
        """
        Make predictions using the trained model.
        
        """
        
        return self.forward(X)