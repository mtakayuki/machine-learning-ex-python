'''
plots a non-linear decision boundary learned by the SVM
'''

import numpy as np
import matplotlib.pyplot as plt
from plotData import plotData


def visualizeBoundary(X, y, model):
    '''
    plots a non-linear decision boundary learned by the SVM
    and overlays the data on it
    '''

    # Plot the training data on top of the boundary
    plotData(X, y)

    # Make classification predictions over a grid of values
    x1plot = np.linspace(X[:, 0].min(), X[:, 0].max(), 100)
    x2plot = np.linspace(X[:, 1].min(), X[:, 1].max(), 100)
    X1, X2 = np.meshgrid(x1plot, x2plot)
    Z = model.predict(np.c_[X1.ravel(), X2.ravel()])
    Z = Z.reshape(X1.shape)

    # Plot the SVM boundary
    plt.contour(X1, X2, Z, [0.5])
    plt.show(block=False)
