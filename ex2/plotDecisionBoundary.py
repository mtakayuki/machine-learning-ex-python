'''
Plots the data points X and y into a new figure with
the decision boundary defined by theta
'''

import matplotlib.pyplot as plt
import numpy as np
from plotData import plotData


def plotDecisionBoundary(theta, X, y):
    '''
    plots the data points with + for the positive examples and o
    for the negative examples. X is assumed to be a either
    1) Mx3 matrix, where the first column is an all-ones column for the
       intercept.
    2) MxN, N>3 matrix, where the first column is all-ones
    '''

    # Plot Data
    plt.figure()
    plotData(X[:, 1:], y)

    if X.shape[1] <= 3:
        # Only need 2 points to define a line, so choose two endpoints
        plot_x = np.array((X[:, 1].min() - 2, X[:, 1].max() + 2))

        # Calculate the decision boundary line
        plot_y = -(theta[0] + theta[1] * plot_x) / theta[2]

        # Plot, and adjust axes for better viewing
        plt.plot(plot_x, plot_y)

        # Legend, specific for the exercise
        plt.legend(('Admitted', 'Not admitted', 'Decision Boundary'))
        plt.axis((30, 100, 30, 100))
