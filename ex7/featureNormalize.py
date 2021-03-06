'''
Normalizes the features in X
'''


def featureNormalize(X):
    '''
    Returns a normalized version of X where the mean value of each
    feature is 0 and the standard deviation is 1. This is often a
    good preprocessing step to do when working with learning algorithms.
    '''

    mu = X.mean(axis=0)
    X_norm = X - mu

    sigma = X_norm.std(axis=0, ddof=1)
    X_norm /= sigma

    return X_norm, mu, sigma
