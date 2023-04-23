import numpy as np
import copy
from Layers import Base


class Sigmoid(Base.BaseLayer):
    def __init__(self):
        """
        Constructor of the Sigmoid layer.
        """
        super().__init__()
        self.activation = None

    def forward(self, input_tensor):
        self.activation = 1 / (1 + np.exp(-input_tensor))
        return copy.deepcopy(self.activation)

    def backward(self, error_tensor):
        return error_tensor * (self.activation * (1 - self.activation))
