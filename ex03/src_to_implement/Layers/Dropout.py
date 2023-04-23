import numpy as np
from Layers import Base


class Dropout(Base.BaseLayer):
    def __init__(self, probability):
        super().__init__()
        self.probability = probability
        self.drop_pattern = None

    def forward(self, input_tensor):
        if self.testing_phase:
            return input_tensor
        else:
            # create binary mask
            self.drop_pattern = np.random.random(input_tensor.shape) > (1 - self.probability)
            return input_tensor * self.drop_pattern * (1 / self.probability)

    def backward(self, error_tensor):
        if self.testing_phase:
            return error_tensor
        else:
            return self.drop_pattern * error_tensor * (1 / self.probability)

