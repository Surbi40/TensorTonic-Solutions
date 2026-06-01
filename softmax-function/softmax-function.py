import numpy as np

def softmax(x):
    """
    Compute softmax for 1D or 2D arrays (row-wise for 2D).
    """

    # Convert input to numpy array (safety)
    x = np.array(x)

    # 1D case
    if x.ndim == 1:
        x_max = np.max(x)
        exp_x = np.exp(x - x_max)
        return exp_x / np.sum(exp_x)

    # 2D case (row-wise softmax)
    elif x.ndim == 2:
        x_max = np.max(x, axis=1, keepdims=True)  # row-wise max
        exp_x = np.exp(x - x_max)
        return exp_x / np.sum(exp_x, axis=1, keepdims=True)

    else:
        raise ValueError("Input must be 1D or 2D array")