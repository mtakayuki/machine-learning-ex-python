import numpy as np
from computeCost import computeCost

def gradientDescent(X, y, theta, alpha, num_iters):
    m = len(y)
    J_history = np.zeros((num_iters, 1))

    for iter in range(0, num_iters):
        # ====================== YOUR CODE HERE ======================
        # ============================================================

        J_history[iter][0] = computeCost(X, y, theta)

    return (theta, J_history)
