from six import StringIO
import tensorflow as tf
import numpy as np
import pandas as pd
import sys
from sklearn import metrics
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier

np.random.seed(123)
tf.random.set_seed(123)

data_train = pd.read_csv("data/train.csv") 
data_test = pd.read_csv("data/test.csv")

cols = list(data_train.columns)

data_train = data_train.to_numpy() 
data_test = data_test.to_numpy() 

print(data_train.shape)
print(data_test.shape)

x_train = data_train[:,:-1]
y_train = data_train[:, -1]

x_test = data_test[:,:-1]
y_test = data_test[:, -1]

model_nn = tf.keras.models.Sequential([
  tf.keras.layers.Dense(6, activation=tf.nn.relu),
  tf.keras.layers.Dense(3, activation=tf.nn.relu),
  tf.keras.layers.Dense(1, activation=tf.nn.sigmoid) #, kernel_regularizer=tf.keras.regularizers.L1(0.01))
])
model_nn.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

print("Train:")
model_nn.fit(x_train, y_train, epochs=50)


print("Eval:")
model_nn.evaluate(x_test, y_test)

y_pred = model_nn.predict(x_train) 
np.savetxt('data/train_pred.csv', y_pred, fmt='%.3f')

y_pred = model_nn.predict(x_test) 
np.savetxt('data/test_pred.csv', y_pred, fmt='%.3f')

model_nn.save('trained_model')

def predict(x):
  y = model_nn.predict(x)
  y[y < 0.5] = 0.0
  y[y >= 0.5] = 1.0 
  y = y.reshape(-1)
  return y #np.zeros_like(y)


def check_performance(y_pred, y_test):
  tp = 0.001 
  fp = 0.001
  tn = 0.001
  fn = 0.001
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

  fpr, tpr, thresholds = metrics.roc_curve(y_test, y_pred, pos_label=1)
  auc = metrics.auc(fpr, tpr)

  print("Accuracy")
  print(acc)
  print("AUC")
  print(auc)
  print()
  return acc, precision, recall, f1, auc

def check_fidelity(y_pred_surrogate, y_pred_model):
  fidelity = 0.001
  for i in range(len(y_pred_surrogate)):
    if(y_pred_surrogate[i] == y_pred_model[i]):
      fidelity = fidelity + 1
  fidelity = fidelity / len(y_pred_model)
  print("Fidelity")
  print(fidelity)
  print()
  return fidelity


print("----------------")
print("Neural Network")
print("Training")
y_pred = predict(x_train) 
acc, precision, recall, f1, auc = check_performance(y_pred, y_train)
print("Testing")
y_pred = predict(x_test) 
acc, precision, recall, f1, auc = check_performance(y_pred, y_test)

error_mask = np.array(y_pred - y_test)
error_mask = np.square(error_mask)
"""
print("----------------")

from skater.core.explanations import Interpretation
from skater.model import InMemoryModel

model_inst = InMemoryModel(predict, examples=x_train, model_type='classifier', unique_values=[0, 1],
                       feature_names=cols[:-1], target_names=['0','1'])
interpreter = Interpretation(x_train, feature_names=cols[:-1], model_inst=model_inst, max_depth=6)
# Using the interpreter instance invoke call to the TreeSurrogate
surrogate = interpreter.get_surrogate()
surrogate.fit(x_train, y_train, use_oracle=True, prune='post', scorer_type='default')
surrogate.plot_global_decisions(colors=['coral', 'lightsteelblue','darkkhaki'],
                                          file_name='simple_tree_pre.png')
model_tree_surrogate = surrogate.estimator_


print("----------------")
print("Tree Surrogate")
#print(tree.export_text(model_tree_surrogate, feature_names=cols[:-1]))
print("Training")
y_pred = model_tree_surrogate.predict(x_train) 
acc, precision, recall, f1, auc = check_performance(y_pred, y_train)
y_pred_nn = predict(x_train) 
y_pred_tree_surrogate = model_tree_surrogate.predict(x_train) 
fidelity = check_fidelity(y_pred_tree_surrogate, y_pred_nn)
print("Testing")
y_pred = model_tree_surrogate.predict(x_test) 
acc, precision, recall, f1, auc = check_performance(y_pred, y_test)
y_pred_nn = predict(x_test) 
y_pred_tree_surrogate = model_tree_surrogate.predict(x_test) 
fidelity = check_fidelity(y_pred_tree_surrogate, y_pred_nn)
print("----------------")

"""

print("----------------")
print("Decision Tree")
model_tree = DecisionTreeClassifier(max_depth=6) 
model_tree.fit(x_train,y_train)
#print(tree.export_text(model_tree, feature_names=cols[:-1]))
print("Training")
y_pred = model_tree.predict(x_train) 
acc, precision, recall, f1, auc = check_performance(y_pred, y_train)
y_pred_nn = predict(x_train) 
y_pred_tree = model_tree.predict(x_train) 
fidelity = check_fidelity(y_pred_tree, y_pred_nn)
print("Testing")
y_pred = model_tree.predict(x_test) 
acc, precision, recall, f1, auc = check_performance(y_pred, y_test)
y_pred_nn = predict(x_test) 
y_pred_tree = model_tree.predict(x_test) 
fidelity = check_fidelity(y_pred_tree, y_pred_nn)
print("On correct samples")
fidelity = check_fidelity(y_pred_tree[error_mask < 0.5], y_pred_nn[error_mask < 0.5])
print("On error samples")
fidelity = check_fidelity(y_pred_tree[error_mask > 0.5], y_pred_nn[error_mask > 0.5])
print("----------------")

model_tree_surrogate = DecisionTreeClassifier(max_depth=6) 
model_tree_surrogate.fit(x_train, predict(x_train))

print("Tree Surrogate (Strawman)")
print("Training")
y_pred = model_tree_surrogate.predict(x_train) 
acc, precision, recall, f1, auc = check_performance(y_pred, y_train)
y_pred_nn = predict(x_train) 
y_pred_tree_surrogate = model_tree_surrogate.predict(x_train) 
fidelity = check_fidelity(y_pred_tree_surrogate, y_pred_nn)
print("Testing")
y_pred = model_tree_surrogate.predict(x_test) 
acc, precision, recall, f1, auc = check_performance(y_pred, y_test)
y_pred_nn = predict(x_test) 
y_pred_tree_surrogate = model_tree_surrogate.predict(x_test) 
fidelity = check_fidelity(y_pred_tree_surrogate, y_pred_nn)
print("On correct samples")
fidelity = check_fidelity(y_pred_tree_surrogate[error_mask < 0.5], y_pred_nn[error_mask < 0.5])
print("On error samples")
fidelity = check_fidelity(y_pred_tree_surrogate[error_mask > 0.5], y_pred_nn[error_mask > 0.5])
print("----------------")

