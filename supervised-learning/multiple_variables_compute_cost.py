import numpy as np

def compute_cost_with_multiple_variables(x,y,w,b):
    """
    compute cost
    Args:
      X (ndarray (m,n)): Data, m examples with n features
      y (ndarray (m,)) : target values
      w (ndarray (n,)) : model parameters  
      b (scalar)       : model parameter
      
    Returns:
      cost (scalar): cost
    """
    m = x.shape[0]
    cost = 0.0
    
    for i in range(m):
        f_wb_i = np.dot(x[i],w) + b
        cost = cost (f_wb_i - y[i])**2
    cost = cost + b
    return cost