import numpy as np

from nn import Module


class L2Loss(Module):
    def __init__(self):
        pass

    def initialize(self, rng):
        pass

    def forward(self, input, target) -> np.float32:
        self.input = input
        self.target = target
        diff = input.reshape(input.shape[0], -1) - target.reshape(target.shape[0], -1)
        output = np.sum(diff ** 2, axis=1)
        self.n = output.shape[0]
        output = np.sum(output) / self.n
        return output

    def backward(self, delta):
        return 2 * (self.input - self.target) * delta / self.n


class SoftmaxWithLogitsLoss(Module):
    def __init__(self):
        pass

    def initialize(self, rng):
        pass

    def forward(self, input, target) -> np.float32:
        """
        Forward pass of the softmax cross-entropy loss.
        Hint: store the input and target or other necessary intermediate values for the backward pass.
        Args:
            input: n x n_class matrix with a d-dimensional feature for each of the n images
            target: n x n_class vector for each of the images
        Returns:
            loss: scalar, the average negative log likelihood loss over the n images
        """
        pass

    def backward(self, delta):
        """
        Backward pass of the softmax cross-entropy loss.
        Hint: use the stored input and target.
        Args:
            delta: scalar, the upstream gradient.
        Returns:
            gradient: n x n_class, gradient with respect to the input.
        """
        pass
