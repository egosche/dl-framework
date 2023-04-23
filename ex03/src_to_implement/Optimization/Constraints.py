import numpy as np


class L2_Regularizer:
    def __init__(self, alpha):
        """
        Constructor for the L2-Regularizer.
        :param alpha: regularization weight
        """
        self.alpha = alpha

    def calculate_gradient(self, weights):
        return self.alpha * weights

    def norm(self, weights):
        # should be with parameter 'fro' but works only like that :c
        return self.alpha * np.square(np.linalg.norm(weights))


class L1_Regularizer:
    def __init__(self, alpha):
        """
        Constructor for the L1-Regularizer.
        :param alpha: regularization weight
        """
        self.alpha = alpha

    def calculate_gradient(self, weights):
        return self.alpha * np.sign(weights)

    def norm(self, weights):
        return self.alpha * np.sum(np.abs(weights))
