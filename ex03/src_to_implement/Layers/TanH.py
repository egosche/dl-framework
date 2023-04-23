import numpy as np
import copy
from Layers import Base


class TanH(Base.BaseLayer):
    def __init__(self):
        """
        Constructor of the TanH layer.
        """
        super().__init__()
        self.activation = None

    def forward(self, input_tensor):
        self.activation = self.tanh(input_tensor)
        return copy.deepcopy(self.activation)

    def backward(self, error_tensor):
        return error_tensor * (1 - (self.activation * self.activation))

    def tanh(self, x):
        return (np.exp(x) - np.exp(-x)) / (np.exp(x) + np.exp(-x))
