import math, copy
import numpy as np
from gradient_descent import gradient_descent


x_train = np.array([1.0, 2.0])
y_train = np.array([300.0, 500.0])

w_init = 0
b_init = 0
iterations = 10000
tmp_alpha = 1.0e-2

# gradient_descent(x_train, y_train, w_init, b_init, tmp_alpha, iterations)

a = np.array([[1,2,4], [2,4,6]]) 

print(a.shape)