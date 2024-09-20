import numpy as np
from gradient_descent import gradient_descent
from sklearn.linear_model import LogisticRegression


X_train = np.array([[0.5, 1.5], [1,1], [1.5, 0.5], [3, 0.5], [2, 2], [1, 2.5]])
y_train = np.array([0, 0, 0, 1, 1, 1])

w_tmp  = np.zeros_like(X_train[0])
b_tmp  = 0.
alph = 0.1
iters = 10000

w_out, b_out, _ = gradient_descent(X_train, y_train, w_tmp, b_tmp, alph, iters) 
print(f"\nupdated parameters: w:{w_out}, b:{b_out}")


# with scikit

lr_model = LogisticRegression()

# fit the model with the training data
lr_model.fit(X_train,y_train)

# make a prediction

y_pred = lr_model.predict(X_train)
print("Prediction on training set:", y_pred)