'''
Run principal component analysis on the dataset X
'''

import numpy as np
import scipy as sp


def pca(X):
    '''
    computes eigenvectors of the covariance matrix of X Returns
    the eigenvectors U, the eigenvalues (on diagonal) in S
    '''

    # Usefule values
    m, n = X.shape

    # You need to return the following variables correctly.
    U = np.zeros(n)
    S = np.zeros(n)

    # ====================== YOUR CODE HERE ======================
    # Instructions: You should first compute the covariance matrix. Then, you
    #               should use the "svd" function to compute the eigenvectors
    #               and eigenvalues of the covariance matrix.
    #
    # Note: When computing the covariance matrix, remember to divide by m (the
    #       number of examples).

    Sigma = 1 / m * X.T.dot(X)
    U, S, V = sp.linalg.svd(Sigma)

    # ============================================================

    return U, S
