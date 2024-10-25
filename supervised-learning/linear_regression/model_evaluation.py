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

# initialize lists to save the errors, models, and feature transforms
train_mses = []
cv_mses = []
models = []
polys = []
scalers = []

# loop over 10 times. Each adding one more degree of polynomial higher than the last.
for degree in range(1,11):
    
    # add polynomial features to the training set
    poly = PolynomialFeatures(degree, include_bias=False)
    X_train_mapped = poly.fit_transform(x_train)
    polys.append(poly)
    
    # scale the training set
    scaler_poly = StandardScaler()
    X_train_mapped_scaled = scaler_poly.fit_transform(X_train_mapped)
    scalers.append(scaler_poly)
    
    # create and train the model
    model = LinearRegression()
    model.fit(X_train_mapped_scaled, y_train )
    models.append(model)
    
    # compute the training MSE
    yhat = model.predict(X_train_mapped_scaled)
    train_mse = mean_squared_error(y_train, yhat) / 2
    train_mses.append(train_mse)
    
    # add polynomial features and scale the cross validation set
    X_cv_mapped = poly.transform(x_cv)
    X_cv_mapped_scaled = scaler_poly.transform(X_cv_mapped)
    
    # compute the cross validation MSE
    yhat = model.predict(X_cv_mapped_scaled)
    cv_mse = mean_squared_error(y_cv, yhat) / 2
    cv_mses.append(cv_mse)
    
# you can decide to use the model with the lowest cv_mse as the one best suited for your application.
degree = np.argmin(cv_mses) + 1
print(f"Lowest CV MSE is found in the model with degree={degree}")