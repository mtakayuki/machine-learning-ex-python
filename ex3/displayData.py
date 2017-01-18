'''
Display 2D data in a nice grid
'''

import numpy as np
import matplotlib.pyplot as plt


def displayData(X, example_width=None):
    '''
    displays 2D data stored in X in a nice grid. It returns
    the figure handle h and the displayed array if requested.
    '''

    # Set example_width automatically if not passed in
    if example_width is None:
        example_width = int(np.round(np.sqrt(X.shape[1])))

    # Gray Image
    plt.set_cmap('gray')

    # Compute rows, cols
    m, n = X.shape
    example_height = int(n / example_width)

    # Compute number of items to display
    display_rows = int(np.floor(np.sqrt(m)))
    display_cols = int(np.ceil(m / display_rows))

    # Between images padding
    pad = 1

    # Setup blank display
    display_array = -np.ones((pad + display_rows * (example_height + pad),
                              pad + display_cols * (example_width + pad)))

    # Copy each example into a patch on the display array
    curr_ex = 0
    for j in range(display_rows):
        for i in range(display_cols):
            if curr_ex > m:
                break

            # Copy the patch
            # Get the max value of the patch
            max_val = np.max(np.abs(X[curr_ex, :]))
            rows = pad + j * (example_height + pad) + \
                np.array(range(example_height + 1))
            cols = pad + i * (example_width + pad) + \
                np.array(range(example_width + 1))
            display_array[min(rows):max(rows), min(cols):max(cols)] = \
                X[curr_ex, :].reshape(example_height, example_width) / max_val
            curr_ex = curr_ex + 1
        if curr_ex > m:
            break

    # Display Image
    plt.imshow(display_array.T)
    plt.axis('off')
    plt.show(block=False)