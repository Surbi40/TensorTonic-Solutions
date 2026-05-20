import numpy as np

def relu(x):
    """
    Implement ReLU activation function.
    """
    x = np.array(x, dtype=float)
    return np.maximum(0,x)