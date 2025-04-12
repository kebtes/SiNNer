class Activation:
    """
    -----------------------------------
    BASE CLASS FOR ACTIVATION FUNCTIONS
    -----------------------------------
    """

    def forward(self, inputs):
        raise NotImplementedError
    
    def backward(self, dvalues):
        raise NotImplementedError