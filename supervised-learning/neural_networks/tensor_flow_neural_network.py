import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input


model = Sequential([
    Input(shape=(2,)),
    Dense(3, activation='sigmoid', name="layer1"),
    Dense(1, activation="sigmoid", name="layer2")
])

# will print the model's layers and parameters
model.summary()

# defines a loss function and specifies a compile optimization.
model.compile(
    loss = tf.keras.losses.BinaryCrossentropy(),
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.01),
)

# we can set the model's weights manually
W1 = np.array([[-8.94,  0.29, 12.89],
               [-0.17, -7.34, 10.79]])
b1 = np.array([-9.87, -9.28,  1.01])

W2 = np.array([[-31.38],
               [-27.86],
               [-32.79]])
b2 = np.array([15.54])

model.get_layer("layer1").set_weights(([W1, b1]))
model.get_layer("layer2").set_weights([W2, b2])

W1, b1 = model.get_layer("layer1").get_weights()
W2, b2 = model.get_layer("layer2").get_weights()


X_test = np.array([[200, 13.9],
                   [200,17]])

# will normalize the values to be predicted
norm_l = tf.keras.layers.Normalization(axis=-1)
X_test_normalized = norm_l(X_test)
predictions = model.predict(X_test_normalized)
print("predictions = \n", predictions)


# to convert the probabilities to a decision, we apply a threshold
yhat = np.zeros_like(predictions)
for i in range(len(predictions)):
    if predictions[i] >= 0.5:
        yhat[i] = 1
    else:
        yhat[i] = 0
print(f"decisions = \n{yhat}")