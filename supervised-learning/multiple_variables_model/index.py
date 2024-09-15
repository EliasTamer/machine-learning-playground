import numpy as np
from sklearn.linear_model import SGDRegressor
from sklearn.preprocessing import StandardScaler


X_train = np.array([[2104, 5, 1, 45], [1416, 3, 2, 40], [852, 2, 1, 35]])
y_train = np.array([460, 232, 178])


scaler = StandardScaler()

# scale/normalize the data (substitute to z score normalize algorithm)
x_norm = scaler.fit_transform(X_train)

# create and fit the regression model
sgdr = SGDRegressor(max_iter=1000)
sgdr.fit(x_norm, y_train)

# view params
b_norm = sgdr.intercept_
w_norm = sgdr.coef_

# making a prediciton on my model
y_predicted = sgdr.predict(w_norm)