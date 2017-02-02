'''
trains multiple logistic regression classifiers and returns all
the classifiers in a matrix all_theta, where the i-th row of all_theta
corresponds to the classifier for label i
'''

import numpy as np
from scipy.optimize import minimize

from lrCostFunction import costFunc, gradFunc


def oneVsAll(X, y, num_labels, lambda_):
    '''
    trains num_labels logisitc regression classifiers and
    returns each of these classifiers in a matrix all_theta,
    where the i-th row of all_theta corresponds to the classifier
    for label i
    '''

    # Some useful variables
    m, n = X.shape

    # You need to return the following variables correctly
    all_theta = np.zeros((num_labels, n + 1))

    # Add ones to the X data matrix
    X = np.c_[np.ones(m), X]

    # ====================== YOUR CODE HERE ======================
    # Instructions: You should complete the following code to train num_labels
    #               logistic regression classifiers with regularization
    #               parameter lambda.
    #
    # Hint: theta(:) will return a column vector.
    #
    # Hint: You can use y == c to obtain a vector of 1's and 0's that tell use
    #       whether the ground truth is true/false for this class.
    #
    # Note: For this assignment, we recommend using fmincg to optimize the cost
    #       function. It is okay to use a for-loop (for c = 1:num_labels) to
    #       loop over the different classes.
    #
    #       fmincg works similarly to fminunc, but is more efficient when we
    #       are dealing with large number of parameters.
    #
    # Example Code for fmincg:
    #
    #     % Set Initial theta
    #     initial_theta = zeros(n + 1, 1);
    #
    #     % Set options for fminunc
    #     options = optimset('GradObj', 'on', 'MaxIter', 50);
    #
    #     % Run fmincg to obtain the optimal theta
    #     % This function will return theta and the cost
    #     [theta] = ...
    #         fmincg (@(t)(lrCostFunction(t, X, (y == c), lambda)), ...
    #                 initial_theta, options);

    for i in range(num_labels):
        # Set initial theta
        initial_theta = np.zeros(n + 1)

        # Set options for fminunc
        options = {'maxiter': 50}

        # Run fmincg to obtain the optional theta
        # This function will return theta and the cost
        res = minimize(costFunc, initial_theta,
                       args=(X, (y == i).astype('int'), lambda_),
                       jac=gradFunc, options=options)
        all_theta[i] = res.x

    # ============================================================

    return all_theta
