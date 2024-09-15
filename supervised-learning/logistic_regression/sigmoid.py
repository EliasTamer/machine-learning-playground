from numpy import np

# to make sure that the predictions of our classification model to be between 0 and 1
def sigmoid(z):
    """
    Compute the sigmoid of z
    Args:
        z (ndarray): A scalar, numpy array of any size.
    Returns:
        g (ndarray): sigmoid(z), with the same shape as z 
    """
    g = 1/(1+np.exp(-z))
    return g