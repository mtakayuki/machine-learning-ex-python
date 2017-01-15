## Machine Learning Online Class - Exercise 1: Linear Regression

#  Instructions
#  ------------
#
#  This file contains code that helps you get started on the
#  linear exercise. You will need to complete the following functions
#  in this exericse:
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
#  For this exercise, you will not need to change any code in this file,
#  or any other files other than those mentioned above.
#
# x refers to the population size in 10,000s
# y refers to the profit in $10,000s

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D

from warmUpExecise import warmUpExecise
from plotData import plotData
from computeCost import computeCost
from gradientDescent import gradientDescent
from pause import pause


## ==================== Part 1: Basic Function ====================
# Complete warmUpExercise.py
print('Running warmUpExercise ...')
print('5x5 Identity Matrix:\n')
print(warmUpExecise(), '\n')

pause()


## ======================= Part 2: Plotting =======================
print('Plotting Data ...')
data = np.genfromtxt('ex1data1.txt', delimiter=',')
X = data[:, 0:1]
y = data[:, 1:2]
m = y.size  # number of training examples

# Plot Data
# Note: You have to complete the code in plotData.m
plotData(X, y)

pause()


## =================== Part 3: Gradient descent ===================
print('Running Gradient Descent ...')

X = np.c_[np.ones(m), data[:, 0:1]]  # Add a column of ones to x
theta = np.zeros(2)  # initialize fitting parameters

# Some gradient descent settings
iterations = 1500
alpha = 0.01

# compute and display initial cost
J = computeCost(X, y, theta)
print(J, '\n')

# run gradient descent
theta, _ = gradientDescent(X, y, theta, alpha, iterations)

# print theta to screen
print('Theta found by gradient descent: %f %f' % (theta[0], theta[1]))

# Plot the linear fit
plt.plot(X[:, 1], X.dot(theta), '-')
plt.legend(('Training data', 'Linear regression'))
plt.show(block=False)

# Predict values for population sizes of 35,000 and 70,000
predict1 = np.array((1, 3.5)).dot(theta)
print('For population = 35,000, we predict a profit of %f' %
      (predict1 * 10000))
predict2 = np.array((1, 7)).dot(theta)
print('For population = 70,000, we predict a profit of %f' %
      (predict2 * 10000))

pause()


## ============= Part 4: Visualizing J(theta_0, theta_1) =============
print('Visualizing J(theta_0, theta_1) ...')

# Grid over which we will calculate J
theta0_vals = np.linspace(-10, 10, 100)
theta1_vals = np.linspace(-1, 4, 100)

# initialize J_vals to a matrix of 0's
J_vals = np.zeros((theta0_vals.size, theta1_vals.size))

# Fill out J_vals
for i in range(0, len(theta0_vals)):
    for j in range(0, len(theta1_vals)):
        t = np.array((theta0_vals[i], theta1_vals[j]))
        J_vals[i][j] = computeCost(X, y, t)

# Because of the way meshgrids work in the surf command, we need to
# transpose J_vals before calling surf, or else the axes will be flipped
J_vals = J_vals.T
# Surface plot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
Theta0_vals, Theta1_vals = np.meshgrid(theta0_vals, theta1_vals)
ax.plot_surface(Theta0_vals, Theta1_vals, J_vals,
                rstride=2, cstride=2, cmap=cm.plasma)
ax.set_xlabel('$\\theta_0$')
ax.set_ylabel('$\\theta_1$')
fig.show()

# Contour plot
plt.figure()
# Plot J_vals as 15 contours spaced logarithmically between 0.01 and 100
plt.contour(Theta0_vals, Theta1_vals, J_vals, np.logspace(-2, 3, 20))
plt.xlabel('$\\theta_0$')
plt.ylabel('$\\theta_1$')
plt.plot(theta[0], theta[1], 'rx', markersize=10, linewidth=2)
plt.show(block=False)

pause()
