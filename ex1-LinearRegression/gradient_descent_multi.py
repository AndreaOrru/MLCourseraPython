import numpy as np

from compute_cost_multi import compute_cost_multi


def gradient_descent_multi(X, y, theta, alpha, iterations):
    m = y.size
    J_history = np.zeros((iterations, 1))

    for i in range(iterations):
        error = np.dot(X, theta) - y
        theta -= ((alpha / m) * np.dot(error.T, X)).T
        J_history[i] = compute_cost_multi(X, y, theta)

    return theta, J_history
