'''
Compute cost for linear regression
'''


def computeCost(X, y, theta):
    '''
    computes the cost of using theta as the parameter for linear
    regression to fit the data points in X and y
    '''

    # Initialize some useful values
    m = y.size
    y = y.reshape(m, 1)
    theta = theta.reshape(theta.size, 1)

    # You need to return the following variables correctly
    J = 0

    # ====================== YOUR CODE HERE ======================
    # Instructions: Compute the cost of a particular choice of theta
    #               You should set J to the cost.

    J = 1 / (2 * m) * ((X.dot(theta) - y)**2).sum()

    # ============================================================

    return J
