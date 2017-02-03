## Machine Learning Online Class
#  Exercise 7 | Principle Component Analysis and K-Means Clustering
#
#  Instructions
#  ------------
#
#  This file contains code that helps you get started on the
#  exercise. You will need to complete the following functions:
#
#     pca.m
#     projectData.m
#     recoverData.m
#     computeCentroids.m
#     findClosestCentroids.m
#     kMeansInitCentroids.m
#
#  For this exercise, you will not need to change any code in this file,
#  or any other files other than those mentioned above.

import numpy as np
import scipy.io
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from scipy.misc import imread
from mpl_toolkits.mplot3d import Axes3D
from pause import pause
from featureNormalize import featureNormalize
from pca import pca
from drawLine import drawLine
from projectData import projectData
from recoverData import recoverData
from displayData import displayData
from kMeansInitCentroids import kMeansInitCentroids
from runkMeans import runkMeans
from plotDataPoints import plotDataPoints


## Initialization
## ================== Part 1: Load Example Dataset  ===================
#  We start this exercise by using a small dataset that is easily to
#  visualize
print('Visualizing example dataset for PCA.\n')

#  The following command loads the dataset. You should now have the
#  variable X in your environment
data = scipy.io.loadmat('ex7data1.mat')
X = data['X']

# Visualize the example dataset
plt.plot(X[:, 0], X[:, 1], 'bo')
plt.axis((0.5, 6.5, 2, 8))
plt.axis('square')
plt.show(block=False)

pause()


## =============== Part 2: Principal Component Analysis ===============
#  You should now implement PCA, a dimension reduction technique. You
#  should complete the code in pca.m
print('\nRunning PCA on example dataset.\n')

#  Before running PCA, it is important to first normalize X
X_norm, mu, sigma = featureNormalize(X)

# Run PCA
U, S = pca(X_norm)

#  Compute mu, the mean of the each feature

#  Draw the eigenvectors centered at mean of data. These lines show the
#  directions of maximum variations in the dataset.
drawLine(mu, mu + 1.5 * S[0] * U[:, 0], '-k', linewidth=2)
drawLine(mu, mu + 1.5 * S[1] * U[:, 1], '-k', linewidth=2)
plt.show(block=False)

print('Top eigenvector:')
print(' U(:,1) = %f %f' % (U[0, 0], U[1, 0]))
print('\n(you should expect to see -0.707107 -0.707107)')

pause()


## =================== Part 3: Dimension Reduction ===================
#  You should now implement the projection step to map the data onto the
#  first k eigenvectors. The code will then plot the data in this reduced
#  dimensional space.  This will show you what the data looks like when
#  using only the corresponding eigenvectors to reconstruct it.
#
#  You should complete the code in projectData.m
print('\nDimension reduction on example dataset.\n')

# Plot the normalized dataset (returned from pca)
plt.figure()
plt.plot(X_norm[:, 0], X_norm[:, 1], 'bo')
plt.axis((-4, 3, -4, 3))
plt.axis('square')
plt.show(block=False)

#  Project the data onto K = 1 dimension
K = 1
Z = projectData(X_norm, U, K)
print('Projection of the first example: %f' % Z[0])
print('\n(this value should be about 1.481274)\n')

X_rec = recoverData(Z, U, K)
print('Approximation of the first example: %f %f' % (X_rec[0, 0], X_rec[0, 1]))
print('\n(this value should be about  -1.047419 -1.047419)\n')

#  Draw lines connecting the projected points to the original points
plt.plot(X_rec[:, 0], X_rec[:, 1], 'ro')
for i in range(X_norm.shape[0]):
    drawLine(X_norm[i, :], X_rec[i, :], '--k', linewidth=1)
plt.show(block=False)

pause()


## =============== Part 4: Loading and Visualizing Face Data =============
#  We start the exercise by first loading and visualizing the dataset.
#  The following code will load the dataset into your environment
#
print('\nLoading face dataset.\n')

# Load Face dataset
data = scipy.io.loadmat('ex7faces.mat')
X = data['X']

# Display the first 100 faces in the dataset
plt.figure()
displayData(X[0:100, :])
plt.show(block=False)

pause()


## =========== Part 5: PCA on Face Data: Eigenfaces  ===================
#  Run PCA and visualize the eigenvectors which are in this case eigenfaces
#  We display the first 36 eigenfaces.
print('\nRunning PCA on face dataset.\n',
      '(this mght take a minute or two ...)\n')

#  Before running PCA, it is important to first normalize X by subtracting
#  the mean value from each feature
X_norm, mu, sigma = featureNormalize(X)

#  Run PCA
U, S = pca(X_norm)

#  Visualize the top 36 eigenvectors found
plt.figure()
displayData(U[:, 0:36].T)
plt.show(block=False)

pause()


## ============= Part 6: Dimension Reduction for Faces =================
#  Project images to the eigen space using the top k eigenvectors
#  If you are applying a machine learning algorithm
print('\nDimension reduction for face dataset.\n')

K = 100
Z = projectData(X_norm, U, K)

print('The projected data Z has a size of: ', Z.shape)

pause()


## ==== Part 7: Visualization of Faces after PCA Dimension Reduction ====
#  Project images to the eigen space using the top K eigen vectors and
#  visualize only using those K dimensions
#  Compare to the original input, which is also displayed
print('\nVisualizing the projected (reduced dimension) faces.\n')

K = 100
X_rec = recoverData(Z, U, K)

# Display normalized data
plt.figure()
plt.subplot(121)
displayData(X_norm[0:100, :])
plt.title('Original faces')
# plt.axis('square')

plt.subplot(122)
displayData(X_rec[0:100, :])
plt.title('Recovered faces')
# plt.axis('square')

plt.show(block=False)

pause()


## === Part 8(a): Optional (ungraded) Exercise: PCA for Visualization ===
#  One useful application of PCA is to use it to visualize high-dimensional
#  data. In the last K-Means exercise you ran K-Means on 3-dimensional
#  pixel colors of an image. We first visualize this output in 3D, and then
#  apply PCA to obtain a visualization in 2D.
plt.close('all')

# Re-load the image from the previous exercise and run K-Means on it
# For this to work, you need to complete the K-Means assignment first
A = imread('bird_small.png')
A = A / 255
img_size = A.shape
X = A.reshape(img_size[0] * img_size[1], 3)
K = 16
max_iters = 10
initial_centroids = kMeansInitCentroids(X, K)
centroids, idx = runkMeans(X, initial_centroids, max_iters)

#  Sample 1000 random indexes (since working with all the data is
#  too expensive. If you have a fast computer, you may increase this.
sel = np.random.permutation(X.shape[0])[0:1000]

# Setup Color Palette
colors = idx[sel]

# Visualize the data and centroid memberships in 3D
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter3D(X[sel, 0], X[sel, 1], X[sel, 2], s=10, c=colors,
             linewidths=0, cmap=cm.hsv)
plt.title('Pixel dataset plotted in 3D. Color shows centroid memberships')
plt.show(block=False)

pause()

## === Part 8(b): Optional (ungraded) Exercise: PCA for Visualization ===
# Use PCA to project this cloud to 2D for visualization

# Subtract the mean to use PCA
X_norm, mu, sigma = featureNormalize(X)

# PCA and project the data to 2D
U, S = pca(X_norm)
Z = projectData(X_norm, U, 2)

# Plot in 2D
plt.figure()
plotDataPoints(Z[sel, :], idx[sel], K)
plt.title('Pixel dataset plotted in 2D, ' +
          'using PCA for dimensionality reduction')
plt.show(block=False)

pause()
