'''
returns a linear kernel between x1 and x2
'''


def linearKernel(x1, x2):
    '''
    returns a linear kernel between x1 and x2 and returns the value in sim
    '''

    # Ensure that x1 and x2 are column vectors
    x1 = x1.reshape(-1, 1)
    x2 = x2.reshape(-1, 1)

    # Compute the kernel
    return x1.T.dot(x2)
