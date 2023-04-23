import numpy as np
from Layers import Base, Helpers
import copy


class BatchNormalization(Base.BaseLayer):
    def __init__(self, channels):
        super().__init__()
        self.trainable = True
        self.testing_phase = False

        self.input_tensor = None
        self.input_tensor_norm = None
        self.error_tensor = None
        self.num_of_channels = channels

        self.weights = np.ones(self.num_of_channels)
        self.bias = np.zeros(self.num_of_channels)
        self.alpha = 0.8
        self.mean = None
        self.variance = None
        self.moving_avg_mean = None
        self.moving_avg_variance = None

        self._gradient_bias = None
        self._gradient_weights = None
        self._bias_optimizer = None
        self._weights_optimizer = None

    def initialize(self, weights_initializer=None, bias_initializer=None):
        self.weights = np.ones(self.weights.shape)
        self.bias = np.zeros(self.bias.shape)

    @property
    def gradient_weights(self):
        return self._gradient_weights

    @gradient_weights.setter
    def gradient_weights(self, weights):
        self._gradient_weights = weights

    @property
    def gradient_bias(self):
        return self._gradient_bias

    @gradient_bias.setter
    def gradient_bias(self, bias):
        self._gradient_bias = bias

    @property
    def optimizer(self):
        return self._weights_optimizer, self._bias_optimizer

    @optimizer.setter
    def optimizer(self, optimizer):
        self._weights_optimizer = copy.deepcopy(optimizer)
        self._bias_optimizer = copy.deepcopy(optimizer)

    def forward(self, input_tensor):
        self.input_tensor = input_tensor

        if len(self.input_tensor.shape) == 4:
            input_tensor = self.reformat(input_tensor)

        if not self.testing_phase:
            self.mean = np.mean(input_tensor, axis=0)
            self.variance = np.var(input_tensor, axis=0)

            # Calculate normalized input tensor
            self.input_tensor_norm = (input_tensor - self.mean) / np.sqrt(self.variance + np.finfo(float).eps)

            # Perform forward pass
            output = self.weights * self.input_tensor_norm + self.bias

            # Reformat the input tensor again if necessary (images)
            if len(self.input_tensor.shape) == 4:
                output = self.reformat(output)

            # Set the moving average mean and variance in the first iteration to mean and variance of first batch
            if self.moving_avg_mean is None:
                self.moving_avg_mean = self.mean
                self.moving_avg_variance = self.variance
            else:
                self.moving_avg_mean = self.alpha * self.moving_avg_mean + (1 - self.alpha) * self.mean
                self.moving_avg_variance = self.alpha * self.moving_avg_variance + (1 - self.alpha) * self.variance
        else:
            # Use moving average values in testing phase
            input_tensor_norm = (input_tensor - self.moving_avg_mean) / np.sqrt(self.moving_avg_variance +
                                                                            np.finfo(float).eps)
            output = self.weights * input_tensor_norm + self.bias

            # Reformat the input tensor again if necessary (images)
            if len(self.input_tensor.shape) == 4:
                output = self.reformat(output)

        return output

    def backward(self, error_tensor):
        # Reformat the error tensor if necessary (images)
        if len(self.input_tensor.shape) == 4:
            self.error_tensor = self.reformat(error_tensor)
            input_tensor = self.reformat(self.input_tensor)
        else:
            self.error_tensor = error_tensor
            input_tensor = self.input_tensor

        # Calculate the gradient w.r.t to weights and bias
        self.gradient_weights = np.sum(self.error_tensor * self.input_tensor_norm, axis=0)
        self.gradient_bias = np.sum(self.error_tensor, axis=0)

        if self._weights_optimizer:     # checking one optimizer is enough here
            self.bias = self._bias_optimizer.calculate_update(self.bias, self.gradient_bias)
            self.weights = self._weights_optimizer.calculate_update(self.weights, self.gradient_weights)

        # Calculate the gradient w.r.t. to input
        output = Helpers.compute_bn_gradients(self.error_tensor, input_tensor, self.weights, self.mean, self.variance)
        if len(error_tensor.shape) == 4:
            output = self.reformat(output)

        return output

    def reformat(self, tensor):
        if len(tensor.shape) == 4:
            batch, channel, height, width = tensor.shape
            reformatted_tensor = np.reshape(tensor, (batch, channel, height * width))
            reformatted_tensor = np.transpose(reformatted_tensor, axes=(0, 2, 1))
            reformatted_tensor = np.reshape(reformatted_tensor, ((batch * height * width), channel))
        else:
            batch, channel, height, width = self.input_tensor.shape
            reformatted_tensor = np.reshape(tensor, (batch, height * width, channel))
            reformatted_tensor = np.transpose(reformatted_tensor, axes=(0, 2, 1))
            reformatted_tensor = np.reshape(reformatted_tensor, (batch, channel, height, width))

        return reformatted_tensor
