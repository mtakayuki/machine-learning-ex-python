'''
Computes the closed-form solution to linear regression
'''

import numpy as np


def normalEqn(X, y):
    '''
    computes the closed-form solution to linear regression using
    the normal equations.
    '''

    theta = np.zeros((np.size(X, 1), 1))

    # ====================== YOUR CODE HERE ======================
    # Instructions: Complete the code to compute the closed form solution
    #               to linear regression and put the result in theta.

    theta = np.linalg.pinv(X.T.dot(X)).dot(X.T).dot(y)

    # ============================================================

    return theta
