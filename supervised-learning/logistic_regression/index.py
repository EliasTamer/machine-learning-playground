import numpy as np
from compute_cost import compute_cost_logistic

X_train = np.array([[0.5, 1.5], [1,1], [1.5, 0.5], [3, 0.5], [2, 2], [1, 2.5]])
y_train = np.array([0, 0, 0, 1, 1, 1])    

w_tmp = np.array([1,1])
b_tmp = -3

print(compute_cost_logistic(X_train, y_train,w_tmp,b_tmp))


# trying with a different w and b values

