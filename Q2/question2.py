import numpy as np
import pandas as pd
from sklearn.datasets import load_iris

iris_dataset = load_iris()

X = iris_dataset['data']
y = iris_dataset['target']

#(2.1)
array1d = np.ones(y.shape[0],int)

#1次元から２次元に変換
array2d = np.expand_dims(array1d, axis=1)
array2d_X_new = np.hstack((array2d,X))

#(3.2)
def f(x,b):
  return 1/(1 + np.exp(-np.dot(x,b)))

np.random.seed(14)
array1d_random = np.array(np.random.randn(5))
y_pred = f(array2d_X_new,array1d_random)

#(3.3)
setosa_avg = np.full(50,np.average(y_pred[:50])) 
versicolor_avg = np.full(50, np.average(y_pred[50:100]))
virginica_avg=np.full(50, np.average(y_pred[100:]))
y_label = np.hstack((setosa_avg,versicolor_avg,virginica_avg))
y_result = y_pred >= y_label
correct_count = np.sum(y_result)
accuracy = correct_count / y.shape[0]

