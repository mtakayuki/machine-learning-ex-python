'''
Visualize the dataset and its estimated distribution.
'''

import numpy as np
import matplotlib.pyplot as plt
from multivariateGaussian import multivariateGaussian


def visualizeFit(X, mu, sigma2):
    '''
    This visualization shows you the probability density function of
    the Gaussian distribution. Each example has a location (x1, x2)
    that depends on its feature values.
    '''

    x1 = np.linspace(0, 35, 71)
    x2 = np.linspace(0, 35, 71)
    X1, X2 = np.meshgrid(x1, x2)
    Z = multivariateGaussian(np.c_[X1.ravel(), X2.ravel()], mu, sigma2)
    Z = Z.reshape(X1.shape)

    plt.figure()
    plt.plot(X[:, 0], X[:, 1], 'bx')
    # Do not plot if there are infinities
    if np.isinf(Z).sum() == 0:
        plt.contour(X1, X2, Z, 10.0**np.arange(-20, 0, 3))
