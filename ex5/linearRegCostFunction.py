'''
Compute cost and gradient for regularized linear regression
with multiple variables
'''

import numpy as np


def linearRegCostFunction(X, y, theta, lambda_):
    # Initialize some useful values
    # number of training examples
    m = y.size
    theta = theta.reshape(-1, 1)

    # You need to return the following variables correctly
    J = 0
    grad = np.zeros(theta.shape)

    # ====================== YOUR CODE HERE ======================
    # Instructions: Compute the cost and gradient of regularized linear
    #               regression for a particular choice of theta.
    #
    #               You should set J to the cost and grad to the gradient.

    h = X.dot(theta)
    J = 1 / (2 * m) * ((h - y)**2).sum()
    J += lambda_ / (2 * m) * (theta[1:, :]**2).sum()
    grad = 1 / m * X.T.dot(h - y)
    grad += lambda_ / m * np.vstack((0, theta[1:, :]))

    # ============================================================

    return J, grad.ravel()
