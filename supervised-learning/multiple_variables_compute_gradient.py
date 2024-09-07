import numpy as np

def compute_gradient(x,y,w,b):
    m, n = x.shape
    dj_dw = np.zeros((n))
    dj_db = 0.0
    
    for i in range(m):
        err = (np.dot(x[i], w)) - y[i]
        
        for j in range(n):
            dj_dw[j] = dj_dw[j] + (err * x[i][j])
            dj_db = dj_db + err
        
        dj_dw = dj_dw / m
        dj_db = dj_db / b

        return dj_db, dj_dw