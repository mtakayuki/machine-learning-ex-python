## Machine Learning Online Class - Exercise 2: Logistic Regression
#
#  Instructions
#  ------------
#
#  This file contains code that helps you get started on the logistic
#  regression exercise. You will need to complete the following functions
#  in this exericse:
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
from plotData import plotData
from costFunction import costFunction, costFunc, gradFunc
from scipy.optimize import minimize
from plotDecisionBoundary import plotDecisionBoundary
from sigmoid import sigmoid
from predict import predict
from pause import pause


## Initialize
plt.close('all')


## Load Data
#  The first two columns contains the exam scores and the third column
#  contains the label.
data = np.genfromtxt('ex2data1.txt', delimiter=',')
X = data[:, 0:2]
y = data[:, 2:3]


## ==================== Part 1: Plotting ====================
#  We start the exercise by first plotting the data to understand the
#  the problem we are working with.
print('Plotting data with + indicating (y = 1) examples and o ',
      'indicating (y = 0) examples.')

plotData(X, y)

# Put some labels
# Labels and Legend
plt.xlabel('Exam 1 score')
plt.ylabel('Exam 2 score')

# Specified in plot order
plt.legend(('Admitted', 'Not admitted'))
plt.show(block=False)

pause()


## ============ Part 2: Compute Cost and Gradient ============
#  In this part of the exercise, you will implement the cost and gradient
#  for logistic regression. You neeed to complete the code in
#  costFunction.m

#  Setup the data matrix appropriately, and add ones for the intercept term
m, n = X.shape

# Add intercept term to x and X_test
X = np.c_[np.ones(m), X]

# Initialize fitting parameters
initial_theta = np.zeros(n + 1)

# Compute and display initial cost and gradient
cost, grad = costFunction(initial_theta, X, y)

print('Cost at initial theta (zeros): %f' % cost)
print('Gradient at initial theta (zeros): ')
print(grad)

pause()


## ============= Part 3: Optimizing using fminunc  =============

#  In this exercise, you will use a built-in function (fminunc) to find the
#  optimal parameters theta.

# Set options for fminunc
options = {'maxiter': 400}

#  Run fminunc to obtain the optimal theta
#  This function will return theta and the cost
res = minimize(costFunc, initial_theta, args=(X, y),
               jac=gradFunc, options=options)
theta = res.x
cost = res.fun

# Print theta to screen
print('Cost at theta found by fminunc: %f' % cost)
print('theta: ')
print(theta)

# Plot Boundary
plotDecisionBoundary(theta, X, y)

# Put some labels
# Labels and Legend
plt.xlabel('Exam 1 score')
plt.ylabel('Exam 2 score')

plt.show(block=False)

pause()


## ============== Part 4: Predict and Accuracies ==============
#  After learning the parameters, you'll like to use it to predict the outcomes
#  on unseen data. In this part, you will use the logistic regression model
#  to predict the probability that a student with score 45 on exam 1 and
#  score 85 on exam 2 will be admitted.
#
#  Furthermore, you will compute the training and test set accuracies of
#  our model.
#
#  Your task is to complete the code in predict.m

#  Predict probability for a student with score 45 on exam 1
#  and score 85 on exam 2
prob = sigmoid(np.array((1, 45, 85)).dot(theta.reshape(-1, 1)))
print('For a student with scores 45 and 85, we predict an admission ',
      'probability of %f\n' % prob)

# Compute accuracy on our training set
p = predict(theta, X)
print('Train Accuracy: %f' % ((p == y).mean() * 100))

pause()
