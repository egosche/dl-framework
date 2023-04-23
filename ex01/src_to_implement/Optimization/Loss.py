import numpy as np


class CrossEntropyLoss:
    def __init__(self):
        """
        Constructor for the Cross Entropy Loss.
        """
        self.prediction_tensor = None

    def forward(self, prediction_tensor, label_tensor):
        """
        The forward pass of the Cross Entropy Loss.
        :param prediction_tensor: tensor with predictions
        :param label_tensor: label tensor for the given inputs
        :return: loss
        """
        self.prediction_tensor = prediction_tensor
        loss = 0

        for count, arr in enumerate(prediction_tensor):
            yk_hat = np.dot(arr, label_tensor[count])  # this gives us the yk hat we should use in the log
            loss += (-1 * np.log(yk_hat + np.finfo(float).eps))

        return loss

    def backward(self, label_tensor):
        """
        The backward pass of the Cross Entropy Loss.
        :param label_tensor: label tensor for the given inputs
        :return:
        """
        return -1 * (label_tensor / np.array(self.prediction_tensor + np.finfo(float).eps))
