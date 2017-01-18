'''
Compute cost and gradient for logistic regression with regularization
'''

import numpy as np
from sigmoid import sigmoid


def costFunctionReg(theta, X, y, lambda_):
    '''
    computes the cost of using theta as the parameter for
    regularized logistic regression and the gradient of
    the cost w.r.t. to the parameters.
    '''

    # Initialize some useful values
    m = y.size
    theta = theta.reshape(-1, 1)

    # You need to return the following variables correctly
    J = 0
    grad = np.zeros(theta.shape)

    # ====================== YOUR CODE HERE ======================
    # Instructions: Compute the cost of a particular choice of theta.
    #               You should set J to the cost.
    #               Compute the partial derivatives and set grad to the partial
    #               derivatives of the cost w.r.t. each parameter in theta

    z = X.dot(theta)
    h = sigmoid(z)
    J = 1 / m * (-y.T.dot(np.log(h)) - (1 - y).T.dot(np.log(1 - h))).sum()
    J += lambda_ / (2 * m) * (theta[1:, :]**2).sum() / (2 * m)
    grad = 1 / m * X.T.dot(h - y)
    grad[1:] += lambda_ / m * theta[1:, :]

    # =============================================================

    return J, grad.ravel()


def costFunc(theta, X, y, lambda_):
    return costFunctionReg(theta, X, y, lambda_)[0]


def gradFunc(theta, X, y, lambda_):
    return costFunctionReg(theta, X, y, lambda_)[1]
