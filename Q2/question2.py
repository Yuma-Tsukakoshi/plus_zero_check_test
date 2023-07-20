import numpy as np
import pandas as pd
from sklearn.datasets import load_iris

iris_dataset = load_iris()

X = iris_dataset['data']
y = iris_dataset['target']

#(2.1) 1次元から２次元に変換
array1d = np.ones(y.shape[0],int)

array2d = np.expand_dims(array1d, axis=1)
array2d_X_new = np.hstack((array2d,X))


#(3.2)
def f(x,b):
  return 1/(1 + np.exp(-np.dot(x,b)))

array1d_random = np.array(np.random.rand(5))

