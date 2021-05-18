import tensorflow as tf
import numpy as np
import pandas as pd
import sys

seed = int(sys.argv[1])
tf.random.set_seed(seed)

data_train = pd.read_csv("data/train.csv").to_numpy() 
data_test = pd.read_csv("data/test.csv").to_numpy() 
print(data_train.shape)
print(data_test.shape)

x_train = data_train[:,:-1]
y_train = data_train[:, -1]

x_test = data_test[:,:-1]
y_test = data_test[:, -1]

model = tf.keras.models.Sequential([
  # tf.keras.layers.Flatten(),
  tf.keras.layers.Dense(6, activation=tf.nn.relu),
  # tf.keras.layers.Dropout(0.2),
  tf.keras.layers.Dense(1, activation=tf.nn.sigmoid)
])
model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

print("Train:")
model.fit(x_train, y_train, epochs=50)


print("Eval:")
model.evaluate(x_test, y_test)

y_pred = model.predict(x_test) 
np.savetxt('data/pred.csv', y_pred, fmt='%.3f')

model.save('income_model')
