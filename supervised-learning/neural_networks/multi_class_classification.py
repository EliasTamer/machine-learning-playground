import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

X_train = np.array( [[1.56, 0.85], [-5.34, 1.03], [-4.09, 0.68]] )
y_train = np.array( [2, 0, 0])

model = Sequential (
    [
        Dense(25, activation="relu"),
        Dense(15, activation ="relu"),
        Dense (4, activation="linear")
    ]
)

'''
'from_logits = True' this informs the loss function that the softmax operation should be included in the loss calculation.
this allows for an optimized implementation.
'''

model.compile(
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    optimizer=tf.keras.optimizers.Adam(0.001)
)

model.fit(X_train,y_train, epochs= 10)

predictions = model.predict(X_train)
print(f"two example output vectors:\n {predictions[:2]}")
print("largest value", np.max(predictions), "smallest value", np.min(predictions))

# if the desired output are probabilities, the output should be be processed by a [softmax]
predictions = tf.nn.softmax(predictions).numpy()
print(predictions)
print(f"two example output vectors:\n {predictions[:2]}")
print("largest value", np.max(predictions), "smallest value", np.min(predictions))