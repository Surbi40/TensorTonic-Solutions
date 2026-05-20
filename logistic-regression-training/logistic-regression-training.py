import numpy as np

def _sigmoid(z):
    """Numerically stable sigmoid implementation."""
    return np.where(z >= 0, 1/(1+np.exp(-z)), np.exp(z)/(1+np.exp(z)))

def train_logistic_regression(X, y, lr=0.1, steps=1000):
    """
    Train logistic regression via gradient descent.
    Return (w, b).
    """
    # Convert inputs to numpy array
    X= np.array(X, dtype=float)
    y = np.array(y,dtype= float)

    #Number of samples(N) and features(D)
    N,D =X.shape

    #initialize weights and bias
    w=np.zeros(D)
    b=0.0
    #Gradient descent Loop 
    for _ in range(steps):

        #linear transformation 
        z = X @ w + b
        #predicted probabilities
        p=_sigmoid(z)

        #compute gradient
        dw = (1/N) * (X.T @ (p-y))
        db = (1/N) * np.sum(p-y)

        #update parameters
        w -= lr * dw
        b-= lr *db
    return w, float(b)
    