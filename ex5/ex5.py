## Machine Learning Online Class
#  Exercise 5 | Regularized Linear Regression and Bias-eariance
#
#  Instructions
#  ------------
#
#  This file contains code that helps you get started on the
#  exercise. You will need to complete the following functions:
#
#     linearRegCostFunction.m
#     learningCurve.m
#     validationCurve.m
#
#  For this exercise, you will not need to change any code in this file,
#  or any other files other than those mentioned above.

import numpy as np
import scipy.io
import matplotlib.pyplot as plt

from linearRegCostFunction import linearRegCostFunction
from trainLinearReg import trainLinearReg
from learningCurve import learningCurve
from polyFeatures import polyFeatures
from featureNormalize import featureNormalize
from plotFit import plotFit
from validationCurve import validationCurve
from pause import pause


## Initialization
plt.close('all')


## =========== Part 1: Loading and Visualizing Data =============
#  We start the exercise by first loading and visualizing the dataset.
#  The following code will load the dataset into your environment and plot
#  the data.

# Load Training Data
print('Loading and Visualizing Data ...')

# Load from ex5data1:
# You will have X, y, Xval, yval, Xtest, ytest in your environment
data = scipy.io.loadmat('ex5data1.mat')
X = data['X']
y = data['y']
Xval = data['Xval']
yval = data['yval']
Xtest = data['Xtest']
ytest = data['ytest']

# m = Number of examples
m = X.shape[0]

# Plot training data
plt.plot(X, y, 'rx', markersize=10, linewidth=1.5)
plt.xlabel('Change in water level (x)')
plt.ylabel('Water flowing out of the dam (y)')
plt.show(block=False)

pause()


## =========== Part 2: Regularized Linear Regression Cost =============
#  You should now implement the cost function for regularized linear
#  regression.
theta = np.ones(2)

J, _ = linearRegCostFunction(np.c_[np.ones(m), X], y, theta, 1)

print('Cost at theta = [1 ; 1]: %f ' % J,
      '\n(this value should be about 303.993192)')

pause()


## =========== Part 3: Regularized Linear Regression Gradient =============
#  You should now implement the gradient for regularized linear
#  regression.
theta = np.ones(2)
J, grad = linearRegCostFunction(np.c_[np.ones(m), X], y, theta, 1)

print('Gradient at theta = [1 ; 1]:  [%f; %f]' % (grad[0], grad[1]),
      '\n(this value should be about [-15.303016; 598.250744])')

pause()


## =========== Part 4: Train Linear Regression =============
#  Once you have implemented the cost and gradient correctly, the
#  trainLinearReg function will use your cost function to train
#  regularized linear regression.
#
#  Write Up Note: The data is non-linear, so this will not give a great
#                 fit.

#  Train linear regression with lambda = 0
lambda_ = 0
theta = trainLinearReg(np.c_[np.ones(m), X], y, lambda_)

#  Plot fit over the data
plt.plot(X, y, 'rx', markersize=10, linewidth=1.5)
plt.xlabel('Change in water level (x)')
plt.ylabel('Water flowing out of the dam (y)')
plt.plot(X, np.c_[np.ones(m), X].dot(theta), '--', linewidth=2)
plt.show(block=False)

pause()


## =========== Part 5: Learning Curve for Linear Regression =============
#  Next, you should implement the learningCurve function.
#
#  Write Up Note: Since the model is underfitting the data, we expect to
#                 see a graph with "high bias" -- slide 8 in ML-advice.pdf
lambda_ = 0
error_train, error_val = \
        learningCurve(np.c_[np.ones(m), X], y,
                      np.c_[np.ones(Xval.shape[0]), Xval], yval,
                      lambda_)

plt.figure()
plt.plot(range(1, m + 1), error_train, range(1, m + 1), error_val)
plt.title('Learning curve for linear regression')
plt.legend(('Train', 'Cross Validation'))
plt.xlabel('Number of training examples')
plt.ylabel('Error')
plt.axis((0, 13, 0, 150))
plt.show(block=False)

print('# Training Examples\tTrain Error\tCross Validation Error')
for i in range(m):
    print('  \t%d\t\t%f\t%f' % (i + 1, error_train[i], error_val[i]))

pause()


## =========== Part 6: Feature Mapping for Polynomial Regression =============
#  One solution to this is to use polynomial regression. You should now
#  complete polyFeatures to map each example into its powers
p = 8

# Map X onto Polynomial Features and Normalize
X_poly = polyFeatures(X, p)
X_poly, mu, sigma = featureNormalize(X_poly)
X_poly = np.c_[np.ones(m), X_poly]

# Map X_poly_test and normalize (using mu and sigma)
X_poly_test = (polyFeatures(Xtest, p) - mu) / sigma
X_poly_test = np.c_[np.ones(X_poly_test.shape[0]), X_poly_test]

# Map X_poly_val and normalize (using mu and sigma)
X_poly_val = (polyFeatures(Xval, p) - mu) / sigma
X_poly_val = np.c_[np.ones(X_poly_val.shape[0]), X_poly_val]

print('Normalized Training Example 1:')
for i in range(X_poly.shape[1]):
    print('  %f' % X_poly[0, i])

pause()


## =========== Part 7: Learning Curve for Polynomial Regression =============
#  Now, you will get to experiment with polynomial regression with multiple
#  values of lambda. The code below runs polynomial regression with
#  lambda = 0. You should try running the code with different values of
#  lambda to see how the fit and learning curve change.
lambda_ = 0
theta = trainLinearReg(X_poly, y, lambda_)

# Plot training data and fit
plt.figure()
plt.plot(X, y, 'rx', markersize=10, linewidth=1.5)
plotFit(X.min(), X.max(), mu, sigma, theta, p)
plt.xlabel('Change in water level (x)')
plt.ylabel('Water flowing out of the dam (y)')
plt.title('Polynomial Regression Fit (lambda = %f)' % lambda_)
plt.show(block=False)

plt.figure()
error_train, error_val = learningCurve(X_poly, y, X_poly_val, yval, lambda_)
plt.plot(range(1, m + 1), error_train, range(1, m + 1), error_val)
plt.title('Polynomial Regression Learning Curve (lambda = %f)' % lambda_)
plt.xlabel('Number of training examples')
plt.ylabel('Error')
plt.axis((0, 13, 0, 100))
plt.legend(('Train', 'Cross Validation'))
plt.show(block=False)

print('Polynomial Regression (lambda = %f)\n' % lambda_)
print('# Training Examples\tTrain Error\tCross Validation Error')
for i in range(m):
    print('  \t%d\t\t%f\t%f' % (i, error_train[i], error_val[i]))

pause()


## =========== Part 8: Validation for Selecting Lambda =============
#  You will now implement validationCurve to test various values of
#  lambda on a validation set. You will then use this to select the
#  "best" lambda value.
lambda_vec, error_train, error_val = \
        validationCurve(X_poly, y, X_poly_val, yval)

plt.figure()
plt.plot(lambda_vec, error_train, lambda_vec, error_val)
plt.legend(('Train', 'Cross Validation'))
plt.xlabel('lambda')
plt.ylabel('Error')
plt.show(block=False)

print('lambda\t\tTrain Error\tValidation Error')
for i in range(len(lambda_vec)):
    print(' %f\t%f\t%f' % (lambda_vec[i], error_train[i], error_val[i]))

pause()
