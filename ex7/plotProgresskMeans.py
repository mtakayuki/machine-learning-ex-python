'''
a helper function that displays the progress of k-Means as it is running.
It is intended for use only with 2D data.
'''

import matplotlib.pyplot as plt
from plotDataPoints import plotDataPoints
from drawLine import drawLine


def plotProgresskMeans(X, centroids, previous, idx, K, i):
    '''
    plots the data points with colors assigned to each centroid.
    With the previous centroids, it also plots a line between
    the previous locations and current locations of the centroids.
    '''

    # Plot the examples
    plotDataPoints(X, idx, K)

    # Plot the centroids as black x's
    plt.plot(centroids[:, 0], centroids[:, 1], 'x', mec='k', ms=10, lw=3)

    # Plot the history of the centroids with lines
    for j in range(centroids.shape[0]):
        drawLine(centroids[j, :], previous[j, :])

    # Title
    plt.title('Iteration number %d' % (i+1))
    plt.show(block=False)
