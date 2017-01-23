'''
lots a learned polynomial regression fit over an existing figure.
Also works with linear regression.
'''

import numpy as np
import matplotlib.pyplot as plt
from polyFeatures import polyFeatures


def plotFit(min_x, max_x, mu, sigma, theta, p):
    '''
    plots the learned polynomial fit with power p and
    feature normalization (mu, sigma).
    '''

    theta = theta.reshape(-1, 1)

    # We plot a range slightly bigger than the min and max values to get
    # an idea of how the fit will vary outside the range of the data points
    x = np.arange(min_x - 15, max_x + 25.05, 0.05)
    x = x.reshape(x.size, 1)

    # Map the X values
    X_poly = (polyFeatures(x, p) - mu) / sigma

    # Add ones
    X_poly = np.c_[np.ones(x.shape[0]), X_poly]

    # Plot
    plt.plot(x, X_poly.dot(theta), '--', linewidth=2)
