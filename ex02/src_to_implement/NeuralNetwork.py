import copy


class NeuralNetwork:
    def __init__(self, optimizer, weights_initializer, bias_initializer):
        """
        Constructor of the Neural Network.
        :param optimizer: defines the optimizer
        """
        self.optimizer = optimizer
        self.loss = []  # contains the loss value for each iteration after calling train
        self.layers = []  # holds the architecture
        self.data_layer = None  # provide input data and labels
        self.loss_layer = None  # referring to the special layer providing loss and prediction
        self.label_tensor = None  # holds label_tensor for the backward pass
        self.weights_initializer = weights_initializer  # holds the initializer for the weights
        self.bias_initializer = bias_initializer    # holds the initializer for the bias

    def forward(self):
        """
        In the forward pass the input data will be propagated through every layer of the neural network. At the end the
        loss will be calculated.
        :return: loss
        """
        input_tensor, self.label_tensor = self.data_layer.next()
        for layer in self.layers:
            # Note: input_tensor is input for the next layer but also output of current layer
            input_tensor = layer.forward(input_tensor)
        output = self.loss_layer.forward(input_tensor, self.label_tensor)

        return output

    def backward(self):
        """
        In the backward pass the weights in each layer (starting backwards) will be updated using the loss.
        :return:
        """
        error = self.loss_layer.backward(self.label_tensor)
        for layer in self.layers[::-1]:  # reverse layer list
            error = layer.backward(error)   # use error to update weights in backward method of each layer

    def append_layer(self, layer):
        """
        Appends a layer to the neural network.
        :param layer: layer to be appended
        :return:
        """
        if layer.trainable:
            layer._optimizer = copy.deepcopy(self.optimizer)    # create independent copy of optimizer object
            layer.initialize(self.weights_initializer, self.bias_initializer)   # initializes trainable layers
        self.layers.append(layer)

    def train(self, iterations):
        """
        Starting the training routine.
        :param iterations: number of iterations
        :return:
        """
        for i in range(iterations):
            output = self.forward()
            self.loss.append(output)
            self.backward()

    def test(self, input_tensor):
        """
        Tests the accuracy of the neural network given an input tensor.
        :param input_tensor: arbitrary input tensor
        :return: result of the last layer (e.g. estimated class probabilities)
        """
        for layer in self.layers:
            input_tensor = layer.forward(input_tensor)
        output = input_tensor   # just to give it a proper name (it's no longer an input)

        return output
