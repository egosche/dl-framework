from Layers import Base, FullyConnected, TanH, Sigmoid
import numpy as np
import copy


class RNN(Base.BaseLayer):
    def __init__(self, input_size, hidden_size, output_size):
        # input_size = dimension of the input vector
        # hidden_size = dimension of the hidden state
        super().__init__()

        self.trainable = True
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.hidden_state = np.zeros(hidden_size)
        self._memorize = False

        self.fc_hidden = FullyConnected.FullyConnected(self.input_size + self.hidden_size, self.hidden_size)
        self.fc_output = FullyConnected.FullyConnected(self.hidden_size, self.output_size)
        self.input_tensor = None
        self.hidden_values = []
        self._weights = self.fc_hidden.weights
        self._weights_output = self.fc_output.weights
        self._gradient_weights = np.zeros(self.weights.shape)

        self._hidden_optimizer = None
        self._output_optimizer = None

    @property
    def memorize(self):
        return self._memorize

    @memorize.setter
    def memorize(self, memorize):
        self._memorize = memorize

    @property
    def weights(self):
        return self.fc_hidden.weights

    @weights.setter
    def weights(self, weights):
        self.fc_hidden.weights = weights

    @property
    def gradient_weights(self):
        return self.gradient_w_hidden

    @gradient_weights.setter
    def gradient_weights(self, weights):
        self.fc_hidden.gradient_weights(weights)

    @property
    def optimizer(self):
        return self._hidden_optimizer

    @optimizer.setter
    def optimizer(self, optimizer):
        self._hidden_optimizer = copy.deepcopy(optimizer)
        self._output_optimizer = copy.deepcopy(optimizer)

    def initialize(self, weights_initializer, bias_initializer):
        self.fc_hidden.initialize(weights_initializer, bias_initializer)
        self.fc_output.initialize(weights_initializer, bias_initializer)

    def calculate_regularization_loss(self):
        return self._hidden_optimizer.regularizer.norm(self.fc_hidden.weights) + \
               self._output_optimizer.regularizer.norm(self.fc_output.weights)

    def forward(self, input_tensor):
        output = np.zeros((input_tensor.shape[0], self.output_size))
        self.input_tensor = input_tensor

        self.hidden_values = np.zeros((input_tensor.shape[0], self.hidden_size))
        self.input_hidden = np.zeros((input_tensor.shape[0], self.input_size + self.hidden_size + 1))
        self.input_out = np.zeros((input_tensor.shape[0], self.hidden_size + 1))
        self.sigmoids = np.zeros((input_tensor.shape[0]), dtype=Sigmoid.Sigmoid)
        self.tanhs = np.empty((input_tensor.shape[0]), dtype=TanH.TanH)

        if not self.memorize:
            self.hidden_state = np.zeros(self.hidden_size)

        for time in range(input_tensor.shape[0]):
            # Concatenate input with hidden state
            input_hidden_concatenated = np.concatenate([input_tensor[time], self.hidden_state])
            # Propagate through hidden fc
            input_hidden_concatenated = self.fc_hidden.forward(np.asmatrix(input_hidden_concatenated))

            self.input_hidden[time] = self.fc_hidden.input_tensor

            # Apply TanH
            tanh = TanH.TanH()
            self.hidden_state = tanh.forward(input_hidden_concatenated)

            self.hidden_values[time] = self.hidden_state

            # Propagate through output fc
            o_t = self.hidden_state
            o_t = self.fc_output.forward(np.asmatrix(o_t))
            self.input_out[time] = self.fc_output.input_tensor

            # Apply Sigmoid
            sigmoid = Sigmoid.Sigmoid()
            output[time] = sigmoid.forward(o_t)

            self.tanhs[time] = tanh
            self.sigmoids[time] = sigmoid

        return output

    def backward(self, error_tensor):
        error = np.zeros((error_tensor.shape[0], self.input_size))
        self.gradient_w_hidden = np.zeros((self.input_size + self.hidden_size + 1, self.hidden_size))
        gradient_w_out = np.zeros(self.fc_output.weights.shape)

        for time in reversed(range(error_tensor.shape[0])):
            # Gradient sigmoid
            grad_ot = self.sigmoids[time].backward(error_tensor[time])
            # Gradient output fc layer
            self.fc_output.input_tensor = self.input_out[time]
            grad_ut = self.fc_output.backward(grad_ot)
            gradient_w_out += self.fc_output.gradient_weights

            # Copy
            if time == error_tensor.shape[0] - 1:
                self.gradient_hidden = np.zeros(self.hidden_size)
            un_copy = grad_ut + self.gradient_hidden

            # Gradient tanH
            grad_tanh = self.tanhs[time].backward(un_copy)

            # Gradient fc hidden
            self.fc_hidden.input_tensor = (self.input_hidden[time])
            grad_hidden = self.fc_hidden.backward(grad_tanh)
            # Get gradient w.r.t to weights and input by splitting
            error[time] = grad_hidden[0:self.input_size]
            self.gradient_hidden = grad_hidden[self.input_size:]

            self.gradient_w_hidden += self.fc_hidden.gradient_weights

        # Use optimizer
        if self._output_optimizer is not None:
            self.fc_hidden.weights = self._hidden_optimizer.calculate_update(self.fc_hidden.weights,
                                                                             self.gradient_w_hidden)
            self.fc_output.weights = self._output_optimizer.calculate_update(self.fc_output.weights,
                                                                             gradient_w_out)

        return error
