import numpy as np
from Layers import Base


class SoftMax(Base.BaseLayer):
    def __init__(self):
        """
        Constructor of the SoftMax layer.
        """
        super().__init__()
        self.max_items = None
        self.pred = None

    def forward(self, input_tensor):
        """
        The forward pass the SoftMax activation function is used to transform the logits (the output of the network)
        into a probability distribution. (Formula: yk = exp(xk)/sum_over_batches(exp(xj)))
        :param input_tensor: input on which the SoftMax function will get applied
        :return: prediction for the class probabilities for each row representing an element of the batch
        """
        input_tensor -= np.max(input_tensor)    # increase numerical stability (xk = xk − max (x))
        expo = np.exp(input_tensor)
        denom = np.sum(expo, axis=1, keepdims=True)
        self.pred = expo / denom

        return self.pred


    def backward(self, error_tensor):
        """
        The backward pass the SoftMax activation function. (Formula: En-1 = y * (En - sum(Enjyj)))
        :param error_tensor: error tensor for current layer
        :return: error tensor for the previous layer
        """
        enjyj = error_tensor * self.pred
        sum_of_enjyj = np.sum(enjyj, axis=1, keepdims=True)
        error_tensor = self.pred * (error_tensor - sum_of_enjyj)

        return error_tensor
