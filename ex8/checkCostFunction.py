'''
Creates a collaborative filering problem to check your cost
function and gradients
'''

import numpy as np
from cofiCostFunc import cofiCostFunc
from computeNumericalGradient import computeNumericalGradient


def checkCostFunction(lambda_=0):
    '''
    Creates a collaborative filering problem to check your
    cost function and gradients, it will output the analytical
    gradients produced by your code and the numerical gradients
    (computed using computeNumericalGradient). These two gradient
    computations should result in very similar values.
    '''

    # Create small problem
    X_t = np.random.rand(4, 3)
    Theta_t = np.random.rand(5, 3)

    # Zap out most entries
    Y = X_t.dot(Theta_t.T)
    Y[np.random.rand(*Y.shape) > 0.5] = 0
    R = np.zeros(Y.shape)
    R[Y != 0] = 1

    # Run Gradient Checking
    X = np.random.randn(*X_t.shape)
    Theta = np.random.randn(*Theta_t.shape)
    num_users = Y.shape[1]
    num_movies = Y.shape[0]
    num_features = Theta_t.shape[1]

    def costFunc(t):
        J, _ = cofiCostFunc(t, Y, R, num_users, num_movies,
                            num_features, lambda_)
        return J

    param = np.r_[X.ravel(), Theta.ravel()]

    numgrad = computeNumericalGradient(costFunc, param)
    cost, grad = cofiCostFunc(param, Y, R, num_users, num_movies,
                              num_features, lambda_)

    print(np.c_[numgrad, grad])
    print('The above two columns you get should be very similar.\n',
          '(Left-Your Numerical Gradient, Right-Analytical Gradient)\n')

    diff = np.linalg.norm(numgrad - grad) / np.linalg.norm(numgrad + grad)
    print('If your backpropagation implementation is correct, then \n',
          'the relative difference will be small (less than 1e-9). \n',
          '\nRelative Difference: %g' % diff)
