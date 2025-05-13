from nnf.layers.base import Layer

class Activation(Layer):
    """
    -----------------------------------
    BASE CLASS FOR ACTIVATION FUNCTIONS
    -----------------------------------
    """
    def __init__(self):
        super().__init__()
        
    def forward(self, inputs):
        raise NotImplementedError
    
    def backward(self, dvalues):
        raise NotImplementedError
    
    def get_params(self):
        return super().get_params()