from nnf.layers.base import Layer

class Optimizer:
    """
    ------------------------
    BASE CLASS FOR OPTMIZERS
    ------------------------
    """
    
    def __init__(self, learning_rate : int = 0.1, decay = 0.0):
        self.learning_rate = learning_rate
        self.current_learning_rate = learning_rate
        self.decay = decay
        self.iterations = 0

    
    def pre_update_params(self):
        """
        The function `pre_update_params` adjusts the current learning rate based on a decay factor and
        the number of iterations.
        """
        if self.decay:
            self.current_learning_rate = self.learning_rate * (1.0 / (1.0 + self.decay * self.iterations))
            
    def update_params(self, layer : Layer):
        """
        Update layer params based on their gradients.
        
        """
        raise NotImplementedError("Subclasses must implement update_params()")
    
    def post_update_params(self):
        """
        Method called after updating params.

        """
        self.iterations += 1