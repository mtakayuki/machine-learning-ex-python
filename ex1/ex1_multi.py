## Machine Learning Online Class
#  Exercise 1: Linear regression with multiple variables
#
#  Instructions
#  ------------
#
#  This file contains code that helps you get started on the
#  linear regression exercise.
#
#  You will need to complete the following functions in this
#  exericse:
#
#     warmUpExercise.m
#     plotData.m
#     gradientDescent.m
#     computeCost.m
#     gradientDescentMulti.m
#     computeCostMulti.m
#     featureNormalize.m
#     normalEqn.m
#
#  For this part of the exercise, you will need to change some
#  parts of the code below for various experiments (e.g., changing
#  learning rates).

import numpy as np
import matplotlib.pyplot as plt

from featureNormalize import featureNormalize
from gradientDescentMulti import gradientDescentMulti
from normalEqn import normalEqn
from pause import pause


## ================ Part 1: Feature Normalization ================
# Clear and Close Figures
plt.close('all')

print('Loading data ...')

# Load Data
data = np.genfromtxt('ex1data2.txt', delimiter=',')
X = data[:, 0:2]
y = data[:, 2:3]
m = y.size

# Print out some data points
print('First 10 examples from the dataset:')
for i in range(10):
    print(' x = [%.0f %.0f], y = %.0f' % (X[i][0], X[i][1], y[i]))

pause()

# Scale features and set them to zero mean

print('Normalizing Features ...')

X, mu, sigma = featureNormalize(X)

# Add intercept term to X
X = np.c_[np.ones(m), X]


## ================ Part 2: Gradient Descent ================

# ====================== YOUR CODE HERE ======================
# Instructions: We have provided you with the following starter
#               code that runs gradient descent with a particular
#               learning rate (alpha).
#
#               Your task is to first make sure that your functions -
#               computeCost and gradientDescent already work with
#               this starter code and support multiple variables.
#
#               After that, try running gradient descent with
#               different values of alpha and see which one gives
#               you the best result.
#
#               Finally, you should complete the code at the end
#               to predict the price of a 1650 sq-ft, 3 br house.
#
# Hint: By using the 'hold on' command, you can plot multiple
#       graphs on the same figure.
#
# Hint: At prediction, make sure you do the same feature normalization.

print('Running gradient descent ...')

# Choose some alpha value
alpha = 0.1
num_iters = 400

# Init Theta and Run Gradient Descent
theta = np.zeros(3)
theta, J_history = gradientDescentMulti(X, y, theta, alpha, num_iters)

# Plot the convergence graph
plt.plot(range(J_history.size), J_history, '-b', linewidth=2)
plt.xlabel('Number of iterations')
plt.ylabel('Cost J')
plt.show(block=False)

# Display gradient descent's result
print('Theta computed from gradient descent:')
for t in theta:
    print(' %f ' % t)
print()

# Estimate the price of a 1650 sq-ft, 3 br house
# ====================== YOUR CODE HERE ======================
price = np.hstack(([1], ([1650, 3] - mu) / sigma)).dot(theta)
# ============================================================

print('Predicted price of a 1650 sq-ft, 3 br house (using gradient descent):',
      '\n $%f\n' % price)

pause()


## ================ Part 3: Normal Equations ================
data = np.genfromtxt('ex1data2.txt', delimiter=',')
X = data[:, 0:2]
y = data[:, 2:3]
m = len(y)

# Add intercept term to X
X = np.hstack((np.ones((m, 1)), X))

# Calculate the parameters from the normal equation

theta = normalEqn(X, y)

# Display normal equation's result
print('Theta computed from the normal equations: ')
for t in theta:
    print(' %f ' % t)
print()

# ====================== YOUR CODE HERE ======================
price = np.array([1, 1650, 3]).dot(theta)
# ============================================================
print('Predicted price of a 1650 sq-ft, 3 br house (using normal equations):',
      '\n $%f\n' % price)

pause()
