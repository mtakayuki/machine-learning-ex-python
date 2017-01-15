import numpy as np


def pause():
    input('Program paused. Press enter to continue.')


## ==================== Part 1: Basic Function ====================
# Complete warmUpExercise.py
from warmUpExecise import warmUpExecise

print('Running warmUpExercise ... ')
print('5x5 Identity Matrix: \n')
print(warmUpExecise(), '\n')

pause()

## ======================= Part 2: Plotting =======================
from plotData import plotData

print('Plotting Data ...')
data = np.genfromtxt('ex1data1.txt', delimiter=',')
X = data[:, 0:1]
y = data[:, 1:2]
m = len(y)

plotData(X, y)

pause()

## =================== Part 3: Gradient descent ===================
import matplotlib.pyplot as plt
from computeCost import computeCost
from gradientDescent import gradientDescent

print('Running Gradient Descent ...')

X = np.hstack((np.ones((m, 1)), data[:, 0:1]))
theta = np.zeros((2, 1))

# Some gradient descent settings
iterations = 1500
alpha = 0.01

# compute and display initial cost
print(computeCost(X, y, theta), '\n')

# run gradient descent
(theta, _) = gradientDescent(X, y, theta, alpha, iterations)

# print theta to screen
print('Theta found by gradient descent: %f %f' % (theta[0], theta[1]))

# Plot the linear fit
plt.plot(X[:, 1], X.dot(theta), '-')
plt.legend(('Training data', 'Linear regression'))
plt.show(block=False)

# Predict values for population sizes of 35,000 and 70,000
predict1 = np.array((1, 3.5)).dot(theta)
print('For population = 35,000, we predict a profit of %f' % (predict1*10000))
predict2 = np.array((1, 7)).dot(theta)
print('For population = 70,000, we predict a profit of %f' % (predict2*10000))

pause()

## ============= Part 4: Visualizing J(theta_0, theta_1) =============
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D

print('Visualizing J(theta_0, theta_1) ...')

# Grid over which we will calculate J
theta0_vals = np.linspace(-10, 10, 100)
theta1_vals = np.linspace(-1, 4, 100)

# initialize J_vals to a matrix of 0's
J_vals = np.zeros((len(theta0_vals), len(theta1_vals)))

# Fill out J_vals
for i in range(0, len(theta0_vals)):
    for j in range(0, len(theta1_vals)):
        t = np.array([[theta0_vals[i]], [theta1_vals[j]]])
        J_vals[i][j] = computeCost(X, y, t)

# Because of the way meshgrids work in the surf command, we need to 
# transpose J_vals before calling surf, or else the axes will be flipped
J_vals = J_vals.T
# Surface plot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
(Theta0_vals, Theta1_vals) = np.meshgrid(theta0_vals, theta1_vals)
ax.plot_surface(Theta0_vals, Theta1_vals, J_vals, rstride=2, cstride=2, cmap=cm.plasma)
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
