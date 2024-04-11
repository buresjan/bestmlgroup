# exercise 1.5.1
import importlib_resources
import numpy as np
import pandas as pd
import importlib_resources
import matplotlib.pyplot as plt
import numpy as np
from scipy.io import loadmat
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

from dtuimldmtools import confmatplot, rocplot

# Load the dataset file using pandas
df = pd.read_csv("HeartDisease.csv")

raw_data = df.values

cols = range(0, 9)
X = raw_data[:, cols]

attributeNames = np.asarray(df.columns[cols])
classLabels = raw_data[:, -1]  # -1 takes the last column

classNames = np.unique(classLabels)

classDict = dict(zip(classNames, range(len(classNames))))

y = np.array([classDict[cl] for cl in classLabels])

N, M = X.shape

C = len(classNames)

K = 20

# Standardize the training and set set based on training set mean and std
mu = np.mean(X, 0)
sigma = np.std(X, 0)

X_train = (X - mu) / sigma
X_test = (X - mu) / sigma

best_lambda = 0.0000000000000016

mdl = LogisticRegression(penalty="l2", C=1 / best_lambda)

mdl.fit(X, y)
mdl.n_iter_ = 1000000

w_est = mdl.coef_
coefficient_norm = np.sqrt(np.sum(w_est**2))

print(w_est)
print(coefficient_norm)
print(attributeNames)

