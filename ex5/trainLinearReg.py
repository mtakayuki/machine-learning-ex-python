'''
Trains linear regression given a dataset (X, y) and a
regularization parameter lambda
'''

import numpy as np
from scipy.optimize import minimize
from linearRegCostFunction import linearRegCostFunction


def trainLinearReg(X, y, lambda_):
    '''
    trains linear regression using the dataset (X, y) and regularization
    parameter lambda. Returns the trained parameters theta.
    '''

    # Initialize Theta
    initial_theta = np.zeros(X.shape[1])

    # Create "short hand" for the cost function to be minimized
    def costFunction(t):
        J, _ = linearRegCostFunction(X, y, t, lambda_)
        return J

    def gradFunction(t):
        _, grad = linearRegCostFunction(X, y, t, lambda_)
        return grad

    # Now, costFunction is a function that takes in only one argument
    options = {'maxiter': 200}

    # Minimize using fmincg
    res = minimize(costFunction, initial_theta, jac=gradFunction,
                   method='CG', options=options)

    return res.x
