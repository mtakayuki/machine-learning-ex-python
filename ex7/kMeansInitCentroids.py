'''
This function initializes K centroids that are to be used in
K-Means on the dataset X
'''

import numpy as np


def kMeansInitCentroids(X, K):
    '''
    returns K initial centroids to be used with the K-Means on the dataset X
    '''

    # You should return this values correctly
    centroids = np.zeros((K, X.shape[1]))

    # ====================== YOUR CODE HERE ======================
    # Instructions: You should set centroids to randomly chosen examples from
    #               the dataset X

    # Initialize the centroids to be random examples
    # Randomly reorder the indices of examples
    randidx = np.random.permutation(X.shape[0])
    # Take the first K examples as centroids
    centroids = X[randidx[0:K], :]

    # ============================================================

    return centroids
