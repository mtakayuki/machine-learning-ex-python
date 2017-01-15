'''
Performs gradient descent to learn theta
'''

import numpy as np
from computeCost import computeCost


def gradientDescent(X, y, theta, alpha, num_iters):
    '''
    updates theta by taking num_iters gradient steps with learning rate alpha
    '''

    # Initialize some useful values
    m = y.size  # number of training examples
    J_history = np.zeros((num_iters, 1))

    y = y.reshape(m, 1)
    theta = theta.reshape(theta.size, 1)

    for iter in range(0, num_iters):
        # ====================== YOUR CODE HERE ======================
        # Instructions: Perform a single gradient step on the parameter vector
        #               theta.
        #
        # Hint: While debugging, it can be useful to print out the values
        #       of the cost function (computeCost) and gradient here.

        theta = theta - alpha / m * X.T.dot(X.dot(theta) - y)

        # ============================================================

        J_history[iter][0] = computeCost(X, y, theta)

    return theta, J_history
