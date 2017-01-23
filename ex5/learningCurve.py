'''
Generates the train and cross validation set errors needed to plot
a learning curve
'''

import numpy as np
from linearRegCostFunction import linearRegCostFunction
from trainLinearReg import trainLinearReg


def learningCurve(X, y, Xval, yval, l):
    # Number of training examples
    m = X.shape[0]

    # You need to return these values correctly
    error_train = np.zeros(m)
    error_val = np.zeros(m)

    # ====================== YOUR CODE HERE ======================
    for i in range(m):
        X_part = X[:i+1, :]
        y_part = y[:i+1, :]
        theta = trainLinearReg(X_part, y_part, l)
        error_train[i], _ = linearRegCostFunction(X_part, y_part, theta, l)
        error_val[i], _ = linearRegCostFunction(Xval, yval, theta, l)
    # ============================================================

    return error_train, error_val
