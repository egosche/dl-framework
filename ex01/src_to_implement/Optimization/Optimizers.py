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
