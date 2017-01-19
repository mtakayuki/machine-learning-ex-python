## Machine Learning Online Class - Exercise 3 | Part 2: Neural Networks

#  Instructions
#  ------------
#
#  This file contains code that helps you get started on the
#  linear exercise. You will need to complete the following functions
#  in this exericse:
#
#     lrCostFunction.m (logistic regression cost function)
#     oneVsAll.m
#     predictOneVsAll.m
#     predict.m
#
#  For this exercise, you will not need to change any code in this file,
#  or any other files other than those mentioned above.

import numpy as np
import scipy.io
import matplotlib.pyplot as plt

from displayData import displayData
from predict import predict
from pause import pause


## Initialization
plt.close('all')

## Setup the parameters you will use for this exercise
# 20x20 Input Images of Digits
input_layer_size = 400
# 25 hidden units
hidden_layer_size = 25
# 10 labels, from 1 to 10 (note that we have mapped "0" to label 10)
num_labels = 10


## =========== Part 1: Loading and Visualizing Data =============
#  We start the exercise by first loading and visualizing the dataset.
#  You will be working with a dataset that contains handwritten digits.

# Load Training Data
data = scipy.io.loadmat('ex3data1.mat')
X = data['X']
y = data['y']
m = X.shape[0]

# Randomly select 100 data points to display
rand_indices = np.random.permutation(m)
sel = X[rand_indices[0:100], :]

displayData(sel)

pause()


## ================ Part 2: Loading Pameters ================
# In this part of the exercise, we load some pre-initialized
# neural network parameters.
print('\nLoading Saved Neural Network Parameters ...')
param = scipy.io.loadmat('ex3weights.mat')
Theta1 = param['Theta1']
Theta2 = param['Theta2']


## ================= Part 3: Implement Predict =================
#  After training the neural network, we would like to use it to predict
#  the labels. You will now implement the "predict" function to use the
#  neural network to predict the labels of the training set. This lets
#  you compute the training set accuracy.
pred = predict(Theta1, Theta2, X)

print('\nTraining Set Accuracy: %f' % (np.mean(pred == y) * 100))

pause()

#  To give you an idea of the network's output, you can also run
#  through the examples one at the a time to see what it is predicting.

#  Randomly permute examples
rp = np.random.permutation(m)

for i in rp:
    # Display
    print('\nDisplaying Example Image')
    displayData(X[i, :].reshape(1, -1))

    pred = predict(Theta1, Theta2, X[i, :].reshape(1, -1))
    print('\nNeural Network Prediction: %d (digit %d)' % (pred, pred % 10))

    pause()
