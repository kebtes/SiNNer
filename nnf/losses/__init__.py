from nnf.losses.base import Loss
from nnf.losses.mse import MSE
from nnf.losses.binary_cross_entropy import BinaryCrossEntropy
from nnf.losses.categorical_cross_entropy import CategoricalCrossEntropy

__all__ = ['MSE', 'BinaryCrossEntropy', 'CategoricalCrossEntropy']