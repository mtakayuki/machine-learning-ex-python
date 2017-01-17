'''
Plots the data points X and y into a new figure
'''

import matplotlib.pyplot as plt


def plotData(X, y):
    '''
    plots the data points with + for the positive examples and o
    for the negative examples. X is assumed to be a Mx2 matrix.
    '''

    # ====================== YOUR CODE HERE ======================
    # Instructions: Plot the positive and negative examples on a
    #               2D plot, using the option 'k+' for the positive
    #               examples and 'ko' for the negative examples.

    # Find indices of positive and negative examples
    pos = (y == 1)[:, 0]
    neg = (y == 0)[:, 0]

    # Plot examples
    plt.plot(X[pos, 0], X[pos, 1], 'r+', linewidth=2, markersize=7)
    plt.plot(X[neg, 0], X[neg, 1], 'ko', markerfacecolor='y', markersize=7)
