import numpy as np


def compute_cost_multi(X, y, theta):
    m = y.size

    error = np.dot(X, theta) - y
    error_squared = np.dot(error.T, error)

    return error_squared / (2 * m)
