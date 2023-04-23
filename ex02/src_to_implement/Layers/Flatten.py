import numpy as np

from Layers import Base


class Flatten(Base.BaseLayer):
    def __init__(self):
        super().__init__()
        self.trainable = False
        self.shape = None

    def forward(self, input_tensor):
        self.shape = input_tensor[0].shape  # Store this so that we'll know the output shape when doing backward()
        batches = len(input_tensor)
        flatten_outputs = []

        for batch in range(0, batches):
            flatten_outputs.append(np.reshape(input_tensor[batch], -1))
        flatten_outputs = np.array(flatten_outputs)
        return flatten_outputs

    def backward(self, error_tensor):
        batches = len(error_tensor)
        return error_tensor.reshape(batches, *self.shape)
