'''
Plots the data points X and y into a new figure
'''

import matplotlib.pyplot as plt


def plotData(X, y):
    '''
    plots the data points with + for the positive examples and o
    for the negative examples. X is assumed to be a Mx2 matrix.
    '''

    # Find Indices of Positive and Negative Examples
    pos = (y == 1).ravel()
    neg = (y == 0).ravel()

    # Plot Examples
    plt.figure()
    plt.plot(X[pos, 0], X[pos, 1], 'k+', linewidth=1, markersize=7)
    plt.plot(X[neg, 0], X[neg, 1], 'ko', markerfacecolor='y', markersize=7)
    plt.show(block=False)
