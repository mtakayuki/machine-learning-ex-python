'''
Find the best threshold (epsilon) to use for selecting outliers
'''

import numpy as np


def selectThreshold(yval, pval):
    '''
    finds the best threshold to use for selecting outliers based on
    the results from a validation set (pval) and the ground truth (yval).
    '''

    bestEpsilon = 0
    bestF1 = 0
    F1 = 0

    yval = yval.ravel()

    for epsilon in np.linspace(pval.min(), pval.max(), 1001):
        # ====================== YOUR CODE HERE ======================
        # Instructions: Compute the F1 score of choosing epsilon as the
        #               threshold and place the value in F1. The code at the
        #               end of the loop will compare the F1 score for this
        #               choice of epsilon and set it to be the best epsilon if
        #               it is better than the current choice of epsilon.
        #
        # Note: You can use predictions = (pval < epsilon) to get a binary vector
        #       of 0's and 1's of the outlier predictions

        predictions = pval < epsilon
        tp = ((yval == True) & (predictions == True)).sum()
        fp = ((yval == False) & (predictions == True)).sum()
        fn = ((yval == True) & (predictions == False)).sum()

        prec = tp / (tp + fp) if tp + fp > 0 else 0
        rec = tp / (tp + fn) if tp + fn > 0 else 0

        F1 = 2 * prec * rec / (prec + rec) if prec + rec > 0 else 0

        # ============================================================

        if F1 > bestF1:
            bestF1 = F1
            bestEpsilon = epsilon

    return bestEpsilon, bestF1
