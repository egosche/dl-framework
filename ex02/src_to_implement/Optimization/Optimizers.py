import numpy as np
import math


class Sgd:
    def __init__(self, learning_rate):
        """
        Constructor of the SGD optimizer.
        :param learning_rate: sets the learning rate of the optimizer
        """
        self.learning_rate = learning_rate

    def calculate_update(self, weight_tensor, gradient_tensor):
        """
        Calculates the update of a given weight tensor w.r.t. to the gradient tensor.
        :param weight_tensor: weight tensor to be updated
        :param gradient_tensor: gradient tensor for the given weight tensor
        :return: updated weight tensor
        """
        return weight_tensor - self.learning_rate * gradient_tensor


class SgdWithMomentum:
    def __init__(self, learning_rate, momentum_rate):
        self.learning_rate = learning_rate
        self.momentum_rate = momentum_rate
        self.change = 0  # Stores the direction we're going. Basically w_(t) - w_(t-1)

    def calculate_update(self, weight_tensor, gradient_tensor):
        self.change = self.momentum_rate * self.change - self.learning_rate * gradient_tensor
        return weight_tensor + self.change


class Adam:
    def __init__(self, learning_rate, mu, rho):
        self.learning_rate = learning_rate
        self.mu = mu
        self.rho = rho
        self.v_k = 0
        self.r_k = 0
        self.epsilon = 1e-8
        self.iter = 1

    def calculate_update(self, weight_tensor, gradient_tensor):
        self.v_k = self.mu * self.v_k + (1 - self.mu) * gradient_tensor
        self.r_k = self.rho * self.r_k + (1 - self.rho) * (gradient_tensor ** 2)

        # Bias correction
        v_k_corr = self.v_k / (1 - self.mu ** self.iter)
        r_k_corr = self.r_k / (1 - self.rho ** self.iter)
        self.iter += 1

        return weight_tensor - self.learning_rate * (v_k_corr / (np.sqrt(r_k_corr) + self.epsilon))
