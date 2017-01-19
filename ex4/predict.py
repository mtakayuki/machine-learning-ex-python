'''
Predict the label of an input given a trained neural network
'''

import numpy as np
from sigmoid import sigmoid


def predict(Theta1, Theta2, X):
    '''
    outputs the predicted label of X given the trained weights of
    a neural network (Theta1, Theta2)
    '''

    # Useful values
    m = X.shape[0]
    # num_labels = Theta2.shape[0]

    # You need to return the following variables correctly
    p = np.zeros((m, 1))

    X = np.c_[np.ones(m), X]
    z2 = X.dot(Theta1.T)
    a2 = sigmoid(z2)

    a2 = np.c_[np.ones(m), a2]
    z3 = a2.dot(Theta2.T)
    a3 = sigmoid(z3)

    p = np.argmax(a3, axis=1) + 1

    return p.reshape(-1, 1)
