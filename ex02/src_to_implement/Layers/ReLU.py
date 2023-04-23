from Layers import Base
import numpy as np


class ReLU(Base.BaseLayer):
    def __init__(self):
        """
        Constructor for the ReLU layer.
        """
        super().__init__()
        self.input_tensor = None

    def forward(self, input_tensor):
        """
        The forward pass applies the ReLU function element-wise on the input tensor.
        :param input_tensor: input on which the ReLU function will get applied
        :return: input tensor for the next layer
        """
        self.input_tensor = input_tensor
        forward_tensor = np.array([rectified_recursive(x) for x in input_tensor])
        return forward_tensor

    def backward(self, error_tensor):
        """
        The backward pass will pass through all values of the error tensor where the value of input tensor was <= 0.
        :param error_tensor: error tensor for current layer
        :return: error tensor for the previous layer
        """
        error_tensor[self.input_tensor <= 0] = 0
        return error_tensor

def rectified_recursive(x):
    if len(x.shape) == 0:
        return max(0, x)
    else:
        result = np.array([rectified_recursive(arr) for arr in x])
    return result

