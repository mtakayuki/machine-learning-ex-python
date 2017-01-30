'''
plots data points in X, coloring them so that those with the same
index assignments in idx have the same color
'''

import matplotlib.pyplot as plt
import matplotlib.cm as cm


def plotDataPoints(X, idx, K):
    '''
    plots data points in X, coloring them so that those with the same
    index assignments in idx have the same color
    '''

    # Plot the data
    plt.scatter(X[:, 0], X[:, 1], s=15, c=idx, cmap=cm.hsv)
