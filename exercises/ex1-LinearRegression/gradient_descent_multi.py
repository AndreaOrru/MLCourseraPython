import numpy as np

from compute_cost_multi import compute_cost_multi


def gradient_descent_multi(X, y, theta, alpha, iterations=1000):
    """
    Perform a gradient descent.

    Args:
        X (numpy.array): A two dimensional matrix of data where
            each column corresponds to a parameter. The leftmost
            column should consist only of 1's.
        y (numpy.array): A vector of known results from the data in `X`
        theta (numpy.array): A vector of initial theta values
        alpha (float): The alpha value controlling the rate of descent
        iterations (int): The number of times to iterate the descent

    Returns:
        tuple (theta, J_history): Where `theta` is a `numpy.array` vector
            of the optimized values for theta from the descent,
            and J_history is a `numpy.array` of error values across
            each iteration of the descent.
    """
    # Make me!
    pass
