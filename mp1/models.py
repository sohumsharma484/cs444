import numpy as np
import logging
from utils import compute_accuracy

class NearestNeighbor(object):
    def __init__(self, data, labels, k):
        """
        Args:
            data: n x d matrix with a d-dimensional feature for each of the n
            points
            labels: n x 1 vector with the label for each of the n points
            k: number of nearest neighbors to use for prediction
        """
        self.k = k
        self.data = data
        self.labels = labels

    def train(self):
        """
        Trains the model and stores in class variables whatever is necessary to
        make predictions later.
        """
        # BEGIN YOUR CODE
        pass
        # END YOUR CODE

    def predict(self, x):
        """
        Args:
            x: n x d matrix with a d-dimensional feature for each of the n
            points
        Returns:
            y: n vector with the predicted label for each of the n points
        """
        # BEGIN YOUR CODE
        pass
        # END YOUR CODE

    def get_nearest_neighbors(self, x, k):
        """
        Args:
            x: n x d matrix with a d-dimensional feature for each of the n
            points
            k: number of nearest neighbors to return
        Returns:
            top_imgs: n x k x d vector containing the nearest neighbors in the
            training data, top_imgs must be sorted by the distance to the
            corresponding point in x.
        """
        # BEGIN YOUR CODE
        pass
        # END YOUR CODE


class LinearClassifier(object):
    def __init__(self, data, labels, epochs=10, lr=1e-3, reg_wt=3e-5, writer=None):
        self.data = data
        self.labels = labels
        self.epochs = epochs
        self.lr = lr
        self.reg_wt = reg_wt
        self.rng = np.random.RandomState(1234)
        std = 1. / np.sqrt(data.shape[1])
        self.w = self.rng.uniform(-std, std, size=(self.data.shape[1], 10))
        self.writer = writer

    def compute_loss_and_gradient(self):
        """
        Computes total loss and gradient of total loss with respect to weights
        w.  You may want to use the `data, w, labels, reg_wt` attributes in
        this function.
        
        Returns:
            data_loss, reg_loss, total_loss: 3 scalars that represent the
                losses $L_d$, $L_r$ and $L$ as defined in the README.
            grad_w: d x 10. The gradient of the total loss (including
            the regularization term), wrt the weight.
        """
        # BEGIN YOUR CODE
        pass
        # END YOUR CODE

    def train(self):
        """Train the linear classifier using gradient descent"""
        for i in range(self.epochs):
            # BEGIN YOUR CODE
            # You may want to call the `compute_loss_and_gradient` method.
            # You can also print the total loss and accuracy on the training
            # data here for debugging.
            pass
            # END YOUR CODE

    def predict(self, x):
        """
        Args:
            x: n x d matrix with a d-dimensional feature for each of the n
            points
        Returns:
            y: n vector with the predicted label for each of the n points
        """
        pass
