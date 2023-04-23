from Layers import Base
import numpy as np


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

    def initialize(self, weights_initializer, bias_initializer):
        """
        Reinitialize weights for the fully connected layer.
        :param weights_initializer: initializer object for the weights
        :param bias_initializer: initializer object for the bias
        :return:
        """
        # Access first to second last row (weights)
        self.weights[0:-1] = weights_initializer.initialize((self.input_size, self.output_size), self.input_size,
                                                            self.output_size)

        # Access last row (bias)
        self.weights[-1] = bias_initializer.initialize((1, self.output_size), self.input_size, self.output_size)

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

    def forward(self, input_tensor):
        """
        The forward pass adds a column of ones for the bias to the input and performs a matrix mul with the weights.
        :param input_tensor: input which will get multiplied with the weights
        :return: input tensor for the next layer
        """
        self.input_tensor = input_tensor  # create a copy for backward pass

        if len(self.input_tensor) >= 2:
            # Add ones for bias
            self.input_tensor = np.ones((len(input_tensor), self.input_size + 1))
            self.input_tensor[:, :-1] = input_tensor
            # Calculate forward pass
            return np.matmul(self.input_tensor, self.weights)
        else:
            # Add ones for bias
            self.input_tensor = np.c_[input_tensor, np.ones(len(input_tensor))]
            # Calculate forward pass
            return np.array(np.matmul(np.asmatrix(self.input_tensor), self.weights))[0]

    def backward(self, error_tensor):
        """
        The backward pass calculates the gradient w.r.t. to the weights to perform an update of the weights.
        :param error_tensor: error tensor for current layer
        :return: error tensor for the previous layer
        """
        if (np.asmatrix(self.input_tensor).shape[0]) >= 2:
            self.gradient_weights = np.matmul(self.input_tensor.T, error_tensor)
        else:
            self.gradient_weights = np.matmul(np.asmatrix(self.input_tensor).T, np.asmatrix(error_tensor))

        # Get unupdated weights without the weights for the bias
        unupdated_weights = np.delete(self.weights, len(self.weights) - 1, axis=0)

        # Update weights
        if self._optimizer is not None:
            self.weights = self._optimizer.calculate_update(self.weights, self.gradient_weights)

        # Return error_tensor for the previous layer
        return np.matmul(error_tensor, unupdated_weights.T)
