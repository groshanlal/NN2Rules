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
  tf.keras.layers.Dense(6, activation=tf.nn.relu),
  tf.keras.layers.Dense(3, activation=tf.nn.relu),
  tf.keras.layers.Dense(1, activation=tf.nn.sigmoid)
])
model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

print("Train:")
model.fit(x_train, y_train, epochs=50)


print("Eval:")
model.evaluate(x_test, y_test)

y_pred = model.predict(x_train) 
np.savetxt('data/train_pred.csv', y_pred, fmt='%.3f')

y_pred = model.predict(x_test) 
np.savetxt('data/test_pred.csv', y_pred, fmt='%.3f')

model.save('income_model')


y_pred[y_pred < 0.5] = 0.0
y_pred[y_pred >= 0.5] = 1.0 
y_pred = y_pred.reshape(-1)

tp = 0 
fp = 0
tn = 0 
fn = 0
for i in range(len(y_pred)):
  if((y_test[i] == 1) and (y_pred[i] == 1)):
    tp = tp + 1
  if((y_test[i] == 1) and (y_pred[i] == 0)):
    fn = fn + 1
  if((y_test[i] == 0) and (y_pred[i] == 1)):
    fp = fp + 1
  if((y_test[i] == 0) and (y_pred[i] == 0)):
    tn = tn + 1

acc = (tp + tn) / (tp + fp + tn + fn)
precision = tp / (tp + fp)
recall = tp / (tp + fn)
f1 = 2*precision*recall/(precision + recall)

print("Accuracy")
print(acc)
print()
print("Precision")
print(precision)
print()
print("Recall")
print(recall)
print()
print("F1-Score")
print(f1)
print()

