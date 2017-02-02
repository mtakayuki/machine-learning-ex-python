'''
Compute cost and gradient for logistic regression
'''

import numpy as np
from sigmoid import sigmoid


def costFunction(theta, X, y):
    '''
    computes the cost of using theta as the parameter for logistic
    regression and the gradient of the cost w.r.t. to the parameters.
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
    #
    # Note: grad should have the same dimensions as theta

    z = X.dot(theta)
    h = sigmoid(z)
    J = 1 / m * (-y.T.dot(np.log(h)) - (1 - y).T.dot(np.log(1 - h))).sum()
    grad = 1 / m * X.T.dot(h - y)

    # =============================================================

    return J, grad.ravel()


def costFunc(theta, X, y):
    return costFunction(theta, X, y)[0]


def gradFunc(theta, X, y):
    return costFunction(theta, X, y)[1]
