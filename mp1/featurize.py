import numpy as np


def pool(x, pool_size=7):
    """
    Implement the pooling featurizer.
    Args:
        x: n x d matrix with a d-dimensional feature for each of the n points
        pool_size: size of the pooling window
    Returns:
        feat: n x dd matrix with the pooled features for each of the n points
            dd = d // pool_size ** 2
    """
    # BEGIN YOUR CODE
    pass
    # END YOUR CODE


def hog(x, pool_size=7, angle_bins=18):
    """
    Implement the Histogram of Gradient featurizer.
    Args:
        x: n x d matrix with a d-dimensional feature for each of the n points
        pool_size: size of the pooling window
        angle_bins: number of bins to use for the angle histogram
            For example, if angle_bins=18, then you should split the gradient
            orientation into 18 equal bins between 0 and 360, each one spanning
            20 degrees.
    Returns:
        feat: n x dd matrix with the HOG features for each of the n points
            dd = d // pool_size ** 2 * angle_bins
    """
    x = x.reshape(-1, 28, 28)
    # BEGIN YOUR CODE
    pass
    # END YOUR CODE


def featurize(x, type='raw', pool_size=7, angle_bins=18):
    if type == 'raw':
        x = x.reshape(x.shape[0], -1) - 0.5
    elif type == 'pool':
        x = pool(x, pool_size=pool_size)
    elif type == 'hog':
        x = hog(x, pool_size=pool_size, angle_bins=angle_bins)
    return x
