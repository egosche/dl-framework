import copy
import numpy as np
from scipy import signal

from Layers import Base


class Conv(Base.BaseLayer):
    def __init__(self, stride_shape, convolution_shape, num_kernels):
        super().__init__()
        self.trainable = True
        self.stride_shape = stride_shape  # allows for different strides in the spatial dimensions
        self.convolution_shape = convolution_shape  # determines whether this object provides a 1D or a 2D conv layer
        self.num_kernels = num_kernels

        # Initialize the parameters of this layer uniformly random in the range [0; 1)
        self.weights = np.random.uniform(0, 1, size=(self.num_kernels, *self.convolution_shape))
        self.bias = np.random.uniform(0, 1, size=self.num_kernels)

        self._gradient_weights = np.zeros(self.weights.shape)
        self._gradient_bias = np.zeros(self.bias.shape)
        self.input_tensor = None
        self._weights_optimizer = None
        self._bias_optimizer = None

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
        self.input_tensor = input_tensor  # create a copy for backward pass
        feature_maps = []  # feature maps that form the output of the forward pass
        for image in self.input_tensor:  # running for every element in the batch
            image_feature_map = []  # feature maps in the current image
            # In the loop below, we convolve the image by every kernel, and stack the result into image_feature_map.
            for num_kernel in range(self.num_kernels):
                # Do a convolution of current image (or 1D signal) with current kernel
                conv_image = signal.correlate(image, self.weights[num_kernel], 'same')

                # extract the valid (middle) channel (number of input channels // 2)
                # correlate function adds padding also z direction (channels). We need only the middle channel.
                conv_image = conv_image[self.convolution_shape[0] // 2]

                # element-wise addition of bias which belongs to the current kernel
                conv_image += self.bias[num_kernel]

                # stride
                if len(self.stride_shape) == 1:
                    # In case of 1D signals
                    conv_image = conv_image[::self.stride_shape[0]]
                else:
                    # In case of images
                    conv_image = conv_image[::self.stride_shape[0], ::self.stride_shape[1]]

                image_feature_map.append(conv_image)
            feature_maps.append(image_feature_map)
            output = np.array(feature_maps)

        return output

    def backward(self, error_tensor):
        error_n_minus_one = []
        error_n_minus_one_in_batch = []

        # -------------------------------------------------------------------------------------------------------------
        # Do upsampling
        # -------------------------------------------------------------------------------------------------------------
        if len(self.stride_shape) == 1 and self.stride_shape[0] != 1:
            # 1D signals
            error_tensor_upsampled = np.zeros(
                (error_tensor.shape[0], self.num_kernels, self.input_tensor.shape[2]))

            for x in range(error_tensor.shape[2]):
                error_tensor_upsampled[:, :, x * self.stride_shape[0]] = error_tensor[:, :, x]
            error_tensor = error_tensor_upsampled
        elif self.stride_shape != (1, 1):
            # 2D signals
            error_tensor_upsampled = np.zeros(
                (error_tensor.shape[0], self.num_kernels, self.input_tensor.shape[2], self.input_tensor.shape[3]))

            for y in range(error_tensor.shape[2]):
                for x in range(error_tensor.shape[3]):
                    error_tensor_upsampled[:, :, y * self.stride_shape[0], x * self.stride_shape[1]] = \
                        error_tensor[:, :, y, x]
            error_tensor = error_tensor_upsampled

        # -------------------------------------------------------------------------------------------------------------
        # Create padded version of the input tensor with half the kernels’ width
        # -------------------------------------------------------------------------------------------------------------
        hwx = self.convolution_shape[1] // 2
        if len(self.convolution_shape) == 3:
            hwy = self.convolution_shape[2] // 2

        # Check if convolution shapes are even. If so, we'll pad one more at one of the sides
        x_extra = 0
        y_extra = 0
        if self.convolution_shape[1] % 2 == 0:
            x_extra = -1
        if len(self.convolution_shape) == 3 and self.convolution_shape[2] % 2 == 0:
            y_extra = -1

        # Padding the input tensor with half widths
        if len(self.input_tensor.shape) == 4:
            padded_input = np.pad(self.input_tensor, ((0, 0), (0, 0), (hwx + x_extra, hwx), (hwy + y_extra, hwy)))
        elif len(self.input_tensor.shape) == 3:
            padded_input = np.pad(self.input_tensor, ((0, 0), (0, 0), (hwx, hwx)))
        else:
            print("Something is wrong with the input tensor shapes!")

        # -------------------------------------------------------------------------------------------------------------
        # Calculate gradient w.r.t. to lower layer
        # -------------------------------------------------------------------------------------------------------------
        # We stack every kernel via axis 1, creating backward kernels
        backward_kernels = np.stack(self.weights, axis=1)
        # Filters need to be flipped (rotated 180 degrees)
        backward_kernels = backward_kernels[::, ::-1]

        # Loop over all kernels, convolve with error_tensor to get each channel ol E_(n-1) (Gradient wrt input)
        # Error tensor is also a batch! So loop over them.
        for et_count, single_err_tensor in enumerate(error_tensor):
            for bkernel in backward_kernels:
                conv_channel = signal.convolve(single_err_tensor, bkernel, 'same')
                conv_channel = conv_channel[backward_kernels.shape[1] // 2]
                error_n_minus_one.append(conv_channel)
            error_n_minus_one_in_batch.append(error_n_minus_one)
            error_n_minus_one = []
        error_n_minus_one_in_batch = np.array(error_n_minus_one_in_batch)

        # -------------------------------------------------------------------------------------------------------------
        # Calculate gradient w.r.t. to weights
        # -------------------------------------------------------------------------------------------------------------
        if len(self.convolution_shape) == 2:
            # 1D signals
            self.gradient_weights = np.zeros((self.num_kernels,
                                              self.convolution_shape[0],
                                              padded_input.shape[2] - error_tensor.shape[2] + 1))
            for kernel in range(self.num_kernels):
                for channel in range(self.convolution_shape[0]):
                    gradient = signal.correlate(padded_input[:, channel], error_tensor[:, kernel], mode='valid')
                    self.gradient_weights[kernel, channel] += gradient[0]
        else:
            # 2D signals
            self.gradient_weights = np.zeros((self.num_kernels,
                                              self.convolution_shape[0],
                                              padded_input.shape[2] - error_tensor.shape[2] + 1,
                                              padded_input.shape[3] - error_tensor.shape[3] + 1))
            for kernel in range(self.num_kernels):
                for channel in range(self.convolution_shape[0]):
                    gradient = signal.correlate(padded_input[:, channel], error_tensor[:, kernel], mode='valid')
                    self.gradient_weights[kernel, channel] += gradient[0]

        # -------------------------------------------------------------------------------------------------------------
        # Calculate gradient w.r.t. to bias
        # -------------------------------------------------------------------------------------------------------------
        # Since every kernel has its own bias, we have to sum kernel-wise over all batches
        if len(self.convolution_shape) == 2:
            # 1D signals
            for kernel in range(self.num_kernels):
                self.gradient_bias[kernel] = np.sum(error_tensor[:, kernel, :])
        else:
            # 2D signals
            for kernel in range(self.num_kernels):
                self.gradient_bias[kernel] = np.sum(error_tensor[:, kernel, :, :])

        # -------------------------------------------------------------------------------------------------------------
        # Update weights and bias
        # -------------------------------------------------------------------------------------------------------------
        if self._weights_optimizer is not None:     # checking one optimizer is enough here
            self.weights = self._weights_optimizer.calculate_update(self.weights, self.gradient_weights)
            self.bias = self._bias_optimizer.calculate_update(self.bias, self.gradient_bias)

        return error_n_minus_one_in_batch

    def initialize(self, weights_initializer, bias_initializer):
        # Get input and output dimensions
        if len(self.convolution_shape) == 2:
            # 1D signals
            fan_in = self.weights.shape[1] * self.weights.shape[2]
            fan_out = self.num_kernels * self.weights.shape[2]
        else:
            # 2D signals
            fan_in = self.weights.shape[1] * self.weights.shape[2] * self.weights.shape[3]
            fan_out = self.num_kernels * self.weights.shape[2] * self.weights.shape[3]

        self.weights = weights_initializer.initialize(self.weights.shape, fan_in, fan_out)
        self.bias = bias_initializer.initialize(self.bias.shape, fan_in, fan_out)
