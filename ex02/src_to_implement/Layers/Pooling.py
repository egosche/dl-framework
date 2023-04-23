import numpy as np

from Layers import Base


class Pooling(Base.BaseLayer):
    def __init__(self, stride_shape, pooling_shape):
        """
        Constructor for the Pooling layer.
        :param stride_shape: controls amount of downsampling
        :param pooling_shape:
        """
        super().__init__()
        self.trainable = False
        self.input_tensor = None
        self.location = None
        self.stride_shape = stride_shape
        self.pooling_shape = pooling_shape

    def forward(self, input_tensor):
        self.input_tensor = input_tensor
        self.location = np.zeros(shape=(*input_tensor.shape, 4))     # add a dimension of len 4 for max value location
        input_channels = input_tensor.shape[1]
        batch_size = input_tensor.shape[0]

        output_height = int((self.input_tensor.shape[2] - self.pooling_shape[0]) / self.stride_shape[0]) + 1
        output_width = int((self.input_tensor.shape[3] - self.pooling_shape[1]) / self.stride_shape[1]) + 1

        output_tensor = np.zeros(shape=(batch_size, input_channels, output_height, output_width))

        # loop over every single value in every input channel in every batch
        for batch in range(batch_size):
            for input_channel in range(input_channels):
                y_output = 0
                for y_input in range(0, input_tensor.shape[2], self.stride_shape[0]):
                    x_output = 0
                    for x_input in range(0, input_tensor.shape[3], self.stride_shape[1]):
                        # check if pooling kernel is inside the input tensor - otherwise IndexError
                        try:
                            # define pooling kernel on input tensor using a.o. current x, y and demanded pooling shape
                            kernel = input_tensor[
                                     batch,
                                     input_channel,
                                     y_input: y_input + self.pooling_shape[0],
                                     x_input: x_input + self.pooling_shape[1]]

                            # write the max value of the current pooling area into the output_tensor
                            output_tensor[batch, input_channel, y_output, x_output] = np.max(kernel)

                            # store maxima locations
                            # unravel_index returns coordinates of max value inside the kernel
                            location = np.unravel_index(np.argmax(kernel), kernel.shape)
                            # max value is in the same batch and input channel
                            self.location[batch, input_channel, y_output, x_output, 0] = batch
                            self.location[batch, input_channel, y_output, x_output, 1] = input_channel
                            # use the max value coordinates in the kernel as the residual to current (y, x) position
                            self.location[batch, input_channel, y_output, x_output, 2] = y_input + location[0]
                            self.location[batch, input_channel, y_output, x_output, 3] = x_input + location[1]
                        except IndexError:
                            continue

                        x_output += 1
                    y_output += 1
        return output_tensor

    def backward(self, error_tensor):
        output_tensor = np.zeros(shape=self.input_tensor.shape)

        # loop over every single value in every error channel in every batch
        for batch in range(error_tensor.shape[0]):
            for error_channel in range(error_tensor.shape[1]):
                for y in range(error_tensor.shape[2]):
                    for x in range(error_tensor.shape[3]):
                        # get location of max value for current iteration and convert return value to integer tuple
                        location = tuple(map(int, self.location[batch, error_channel, y, x]))

                        # In cases where the stride is smaller than the kernel size the error might be
                        # routed multiple times to the same location and therefore has to be summed
                        # up
                        output_tensor[location] += error_tensor[batch, error_channel, y, x]

        return output_tensor
