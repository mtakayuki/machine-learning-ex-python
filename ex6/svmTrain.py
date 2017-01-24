'''
Trains an SVM classifier using a simplified version of the SMO algorithm.
'''

import numpy as np
from functools import partial
from sklearn.svm import SVC


def svmTrain(X, Y, C, kernelFunction='rbf', tol=1e-3, max_pass=-1):
    '''
    trains an SVM classifier and returns trained model. X is the matrix of
    training examples.  Each row is a training example, and the jth column
    holds the jth feature. Y is a column matrix containing 1 for positive
    examples and 0 for negative examples.  C is the standard SVM
    regularization parameter. tol is a tolerance value used for determining
    equality of floating point numbers. max_passes controls the number of
    iterations over the dataset (without changes to alpha) before the
    algorithm quits.

    Note: This is a simplified version of the SMO algorithm for training
          SVMs. In practice, if you want to train an SVM classifier, we
          recommend using an optimized package such as:

              LIBSVM   (http://www.csie.ntu.edu.tw/~cjlin/libsvm/)
              SVMLight (http://svmlight.joachims.org/)
    '''

    if callable(kernelFunction):
        kernelFunction = partial(_proxy_kernel, K=kernelFunction)

    svc = SVC(C=C, kernel=kernelFunction, tol=tol, max_iter=max_pass)
    svc.fit(X, Y.ravel())
    return svc


def _proxy_kernel(X, Y, K):
    gram_matrix = np.zeros((X.shape[0], Y.shape[0]))
    for i, x in enumerate(X):
        for j, y in enumerate(Y):
            gram_matrix[i, j] = K(x, y)
    return gram_matrix
