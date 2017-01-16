'''
Performs gradient descent to learn theta
'''

import numpy as np
from computeCostMulti import computeCostMulti


def gradientDescentMulti(X, y, theta, alpha, num_iters):
    '''
    updates theta by taking num_iters gradient steps with learning rate alpha
    '''

    # Initialize some useful values
    m = y.size
    J_history = np.zeros(num_iters)

    y = y.reshape(m, 1)
    theta = theta.reshape(theta.size, 1)

    for iter in range(num_iters):
        # ====================== YOUR CODE HERE ======================
        # Instructions: Perform a single gradient step on the parameter vector
        #               theta.
        #
        # Hint: While debugging, it can be useful to print out the values
        #       of the cost function (computeCostMulti) and gradient here.

        theta = theta - alpha / m * X.T.dot(X.dot(theta) - y)

        # ============================================================

        # Save the cost J in every iteration
        J_history[iter] = computeCostMulti(X, y, theta)

    return theta, J_history
