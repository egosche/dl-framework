import numpy as np


class Constant:
    def __init__(self, constant=0.1):
        """
        Constructor for the constant initializer.
        :param constant: constant as initial weights (default: 0.1)
        """
        self.constant = constant

    def initialize(self, weights_shape, fan_in, fan_out):
        """
        Initialize weights for a given shape with a constant value.
        :param weights_shape: shape of the weights
        :param fan_in: input dimension of the weights (FCN) or [# input channels * kernel height * kernel width] (CNN)
        :param fan_out:output dimension of the weights (FCN) or [# output channels * kernel height * kernel width] (CNN)
        :return: initialized weight tensor
        """
        return np.ones(weights_shape) * self.constant


class UniformRandom:
    def __init__(self):
        """
        Constructor for the uniform random initializer.
        """
        pass

    def initialize(self, weights_shape, fan_in, fan_out):
        """
        Initialize weights for a given shape with uniform random values.
        :param weights_shape: shape of the weights
        :param fan_in: input dimension of the weights (FCN) or [# input channels * kernel height * kernel width] (CNN)
        :param fan_out:output dimension of the weights (FCN) or [# output channels * kernel height * kernel width] (CNN)
        :return: initialized weight tensor
        """
        return np.random.uniform(size=weights_shape)


class Xavier:
    def __init__(self):
        """
        Constructor for the Xavier initializer.
        """
        pass

    def initialize(self, weights_shape, fan_in, fan_out):
        """
        Initialize weights for a given shape with the Xavier/Glorot method.
        :param weights_shape: shape of the weights
        :param fan_in: input dimension of the weights (FCN) or [# input channels * kernel height * kernel width] (CNN)
        :param fan_out:output dimension of the weights (FCN) or [# output channels * kernel height * kernel width] (CNN)
        :return: initialized weight tensor
        """
        sigma = np.sqrt(2 / (fan_out + fan_in))
        return np.random.normal(loc=0, scale=sigma, size=weights_shape)     # Zero-mean Gaussian


class He:
    def __init__(self):
        """
        Constructor for the He initializer.
        """
        pass

    def initialize(self, weights_shape, fan_in, fan_out):
        """
        Initialize weights for a given shape with the He method.
        :param weights_shape: shape of the weights
        :param fan_in: input dimension of the weights (FCN) or [# input channels * kernel height * kernel width] (CNN)
        :param fan_out:output dimension of the weights (FCN) or [# output channels * kernel height * kernel width] (CNN)
        :return: initialized weight tensor
        """
        sigma = np.sqrt(2 / fan_in)
        return np.random.normal(loc=0, scale=sigma, size=weights_shape)     # Zero-mean Gaussian
