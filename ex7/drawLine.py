'''
Draws a line from point p1 to point p2
'''

import matplotlib.pyplot as plt


def drawLine(p1, p2, *args, **kwargs):
    '''
    Draws a line from point p1 to point p2 and holds the current figure
    '''

    plt.plot((p1[0], p2[0]), (p1[1], p2[1]), *args, **kwargs)
