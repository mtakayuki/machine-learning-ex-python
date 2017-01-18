## Machine Learning Online Class - Exercise 2: Logistic Regression
#
#  Instructions
#  ------------
#
#  This file contains code that helps you get started on the second part
#  of the exercise which covers regularization with logistic regression.
#
#  You will need to complete the following functions in this exericse:
#
#     sigmoid.m
#     costFunction.m
#     predict.m
#     costFunctionReg.m
#
#  For this exercise, you will not need to change any code in this file,
#  or any other files other than those mentioned above.

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import minimize

from plotData import plotData
from mapFeature import mapFeature
from costFunctionReg import costFunctionReg, costFunc, gradFunc
from plotDecisionBoundary import plotDecisionBoundary
from predict import predict
from pause import pause


## Initialization
plt.close('all')


## Load Data
#  The first two columns contains the X values and the third column
#  contains the label (y).
data = np.genfromtxt('ex2data2.txt', delimiter=',')
X = data[:, 0:2]
y = data[:, 2:3]

plotData(X, y)

# Put some labels
# Labels and Legend
plt.xlabel('Microchip Test 1')
plt.ylabel('Microchip Test 2')

# Specified in plot order
plt.legend(('y = 1', 'y = 0'))

plt.show(block=False)


## =========== Part 1: Regularized Logistic Regression ============
#  In this part, you are given a dataset with data points that are not
#  linearly separable. However, you would still like to use logistic
#  regression to classify the data points.
#
#  To do so, you introduce more features to use -- in particular, you add
#  polynomial features to our data matrix (similar to polynomial
#  regression).

# Add Polynomial Features

# Note that mapFeature also adds a column of ones for us, so the intercept
# term is handled
X = mapFeature(X[:, 0:1], X[:, 1:2])

# Initialize fitting parameters
initial_theta = np.zeros(X.shape[1])

# Set regularization parameter lambda to 1
l = 1

# Compute and display initial cost and gradient for regularized logistic
# regression
cost, grad = costFunctionReg(initial_theta, X, y, l)

print('Cost at initial theta (zeros): %f' % cost)

pause()


## ============= Part 2: Regularization and Accuracies =============
#  Optional Exercise:
#  In this part, you will get to try different values of lambda and
#  see how regularization affects the decision coundart
#
#  Try the following values of lambda (0, 1, 10, 100).
#
#  How does the decision boundary change when you vary lambda? How does
#  the training set accuracy vary?

# Initialize fitting parameters
initial_theta = np.zeros(X.shape[1])

# Set regularization parameter lambda to 1 (you should vary this)
lambda_ = 1

# Set Options
options = {'maxiter': 400}

# Optimize
res = minimize(costFunc, initial_theta, args=(X, y, lambda_),
               jac=gradFunc, options=options)
theta = res.x
cost = res.fun

# Plot Boundary
plotDecisionBoundary(theta, X, y)
plt.title('lambda = %g' % lambda_)

# Labels and Legend
plt.xlabel('Microchip Test 1')
plt.ylabel('Microchip Test 2')

plt.legend(('y = 1', 'y = 0', 'Decision boundary'))
plt.show(block=False)

# Compute accuracy on our training set
p = predict(theta, X)

print('Train Accuracy: %f' % ((p == y).mean() * 100))

pause()
