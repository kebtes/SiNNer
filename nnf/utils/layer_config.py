from nnf.activations import LeakyReLU
from nnf.activations import ReLU
from nnf.activations import Sigmoid
from nnf.activations import Softmax
from nnf.activations import Tanh

from nnf.layers import Dense

from nnf.losses import MSE
from nnf.losses import BinaryCrossEntropy
from nnf.losses import CategoricalCrossEntropy

from nnf.optimizers import GradientDescent
from nnf.optimizers import Momentum

LAYER_CLASSES = {
    "Dense"                     : Dense,
    "ReLU"                      : ReLU,
    "LeakyRelU"                 : LeakyReLU,
    "Sigmoid"                   : Sigmoid,
    "Softmax"                   : Softmax,
    "Tanh"                      : Tanh,
    "GradientDescent"           : GradientDescent,
    "Momentum"                  : Momentum,
    "MSE"                       : MSE,
    "BinaryCrossEntropy"        : BinaryCrossEntropy,
    "CategoricalCrossEntropy"   : CategoricalCrossEntropy
}