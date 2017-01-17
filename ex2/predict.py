'''
Predict whether the label is 0 or 1 using learned logistic
regression parameters theta
'''

import numpy as np


def predict(theta, X):
    '''
    computes the predictions for X using a threshold at 0.5
    (i.e., if sigmoid(theta'*x) >= 0.5, predict 1)
    '''

    # Number of training examples
    m = X.shape[0]

    # You need to return the following variables correctly
    p = np.zeros((m, 1))

    theta = theta.reshape(-1, 1)

    # ====================== YOUR CODE HERE ======================
    p = X.dot(theta) >= 0
    # ============================================================

    return p
