import numpy as np

from src_to_implement.Layers import Base


class FullyConnected(Base.BaseLayer):
    def __init__(self, input_size, output_size):
        """
        Constructor for a fully connected layer.
        :param input_size: number of inputs
        :param output_size: number of outputs
        """
        super().__init__()
        self.trainable = True
        self.input_size = input_size
        self.output_size = output_size
        self.input_tensor = None
        self._optimizer = None
        self.weights = np.random.uniform(0, 1, size=(input_size + 1, output_size))  # plus 1 for the bias
        self._gradient_weights = None

    def forward(self, input_tensor):
        """
        The forward pass adds a column of ones for the bias to the input and performs a matrix mul with the weights.
        :param input_tensor: input which will get multiplied with the weights
        :return: input tensor for the next layer
        """
        input_tensor = np.c_[input_tensor, np.ones(len(input_tensor))]
        self.input_tensor = input_tensor  # create a copy for backward pass
        # On slides, the order is reversed. It's weights * input_tensor (not good).
        return np.matmul(input_tensor, self.weights)

    def backward(self, error_tensor):
        """
        The backward pass calculates the gradient w.r.t. to the weights to perform an update of the weights.
        :param error_tensor: error tensor for current layer
        :return: error tensor for the previous layer
        """
        # Note: on slides, the order is reversed. It's error_tensor * input_tensor.T
        self._gradient_weights = np.matmul(self.input_tensor.T, error_tensor)

        # Get unupdated weights without the weights for the bias
        unupdated_weights = np.delete(self.weights, len(self.weights) - 1, axis=0)

        # Update weights
        if self._optimizer is not None:
            self.weights = self._optimizer.calculate_update(self.weights, self._gradient_weights)

        # Return error_tensor for the previous layer
        return np.matmul(error_tensor, unupdated_weights.T)

    @property
    def optimizer(self):
        return self._optimizer

    @optimizer.setter
    def optimizer(self, optimizer):
        self._optimizer = optimizer

    @property
    def gradient_weights(self):
        return self._gradient_weights

    @gradient_weights.setter
    def gradient_weights(self, gradient_weights):
        self._gradient_weights = gradient_weights
