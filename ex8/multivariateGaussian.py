'''
Computes the probability density function of the multivariate
gaussian distribution.
'''

from numpy import pi, diag, exp
from numpy.linalg import det, pinv


def multivariateGaussian(X, mu, Sigma2):
    '''
    Computes the probability density function of the examples X
    under the multivariate gaussian distribution with parameters
    mu and Sigma2. If Sigma2 is a matrix, it is treated as the
    covariance matrix. If Sigma2 is a vector, it is treated as
    the \sigma^2 values of the variances in each dimension
    (a diagonal covariance matrix)
    '''

    k = mu.size

    if len(Sigma2.shape) == 1 or Sigma2.shape[0] == 1 or Sigma2.shape[1] == 1:
        Sigma2 = diag(Sigma2)

    X = X - mu
    p = (2 * pi)**(-k / 2) * det(Sigma2)**(-0.5) * \
        exp(-0.5 * (X.dot(pinv(Sigma2)) * X).sum(axis=1))

    return p
