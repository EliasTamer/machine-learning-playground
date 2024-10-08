import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.linear_model import LinearRegression

x = np.array([[1651.  ], [1691.82], [1732.63], [1773.45],[1814.27], [1855.08], [1895.9 ],
 [1936.71],
 [1977.53],
 [2018.35],
 [2059.16],
 [2099.98],
 [2140.8 ],
 [2181.61],
 [2222.43],
 [2263.24],
 [2304.06],
 [2344.88],
 [2385.69],
 [2426.51],
 [2467.33],
 [2508.14],
 [2548.96],
 [2589.78],
 [2630.59],
 [2671.41],
 [2712.22],
 [2753.04],
 [2793.86],
 [2834.67],
 [2875.49],
 [2916.31],
 [2957.12],
 [2997.94],
 [3038.76],
 [3079.57],
 [3120.39],
 [3161.2 ],
 [3202.02],
 [3242.84],
 [3283.65],
 [3324.47],
 [3365.29],
 [3406.1 ],
 [3446.92],
 [3487.73],
 [3528.55],
 [3569.37],
 [3610.18],
 [3651.  ]])

y = np.array([[432.65],
 [454.94],
 [471.53],
 [482.51],
 [468.36],
 [482.15],
 [540.02],
 [534.58],
 [558.35],
 [566.42],
 [581.4 ], 
 [596.46],
 [596.71],
 [619.45],
 [616.58],
 [653.16],
 [666.52],
 [670.59],
 [669.02],
 [678.91],
 [707.44],
 [710.76],
 [745.19],
 [729.85],
 [743.8 ],
 [738.2 ],
 [772.95],
 [772.22],
 [784.21],
 [776.43],
 [804.78],
 [833.27],
 [825.69],
 [821.05],
 [833.82],
 [833.06],
 [825.7 ],
 [843.58],
 [869.4 ],
 [851.5 ],
 [863.18],
 [853.01],
 [877.16],
 [863.74],
 [874.67],
 [877.74],
 [874.11],
 [882.8 ],
 [910.83],
 [897.42]])


# get 60% of the dataset as the training set. put the remaining 40% in temporary variables: x_ and y_.
x_train, x_, y_train, y_ = train_test_split(x, y, test_size=0.40, random_state=1)

# split the 40% subset above into two: one half for cross validation and the other for the test set
x_cv, x_test, y_cv, y_test = train_test_split(x_, y_, test_size=0.50, random_state=1)

print(f"the shape of the training set (input) is: {x_train.shape}")
print(f"the shape of the training set (target) is: {y_train.shape}\n")
print(f"the shape of the cross validation set (input) is: {x_cv.shape}")
print(f"the shape of the cross validation set (target) is: {y_cv.shape}\n")
print(f"the shape of the test set (input) is: {x_test.shape}")
print(f"the shape of the test set (target) is: {y_test.shape}")


scaler_linear = StandardScaler()

# compute the mean and standard deviation of the training set then transform it
x_train_scaled = scaler_linear.fit_transform(x_train)

print(f"Computed mean of the training set: {scaler_linear.mean_.squeeze():.2f}")
print(f"Computed standard deviation of the training set: {scaler_linear.scale_.squeeze():.2f}")

linear_model = LinearRegression()

# train the model
linear_model.fit(x_train_scaled, y_train)

# feed the scaled training set and get the predictions
yhat = linear_model.predict(x_train_scaled)

# use scikit-learn's utility function and divide by 2
print(f"training MSE (using sklearn function): {mean_squared_error(y_train, yhat) / 2}")

# scale the cross validation set using the mean and standard deviation of the training set
x_cv_scaled = scaler_linear.transform(x_cv)

print(f"Mean used to scale the CV set: {scaler_linear.mean_.squeeze():.2f}")
print(f"Standard deviation used to scale the CV set: {scaler_linear.scale_.squeeze():.2f}")

# feed the scaled cross validation set
y_hat = linear_model.predict(x_cv_scaled)

# use scikit-learn's utility function and divide by 2
print(f"Cross validation MSE (using sklearn function): {mean_squared_error(y_cv, yhat) / 2}")
