import scipy.stats as st
import sklearn.linear_model
import sklearn.tree

# requires data from exercise 1.5.1

from sklearn import model_selection

from dtuimldmtools import *
from dtuimldmtools.statistics.statistics import correlated_ttest
# exercise 1.5.1
import importlib_resources
import numpy as np
import pandas as pd
import importlib_resources
from matplotlib.pyplot import (
    colorbar,
    figure,
    imshow,
    plot,
    show,
    title,
    xlabel,
    xticks,
    ylabel,
    yticks,
)
from scipy.io import loadmat
from sklearn.metrics import confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
import importlib_resources
import matplotlib.pyplot as plt
import numpy as np
from scipy.io import loadmat
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from dtuimldmtools import confmatplot, rocplot

######### DATA GATHERING
df = pd.read_csv("HeartDisease.csv")

raw_data = df.values

cols = range(0, 9)
X = raw_data[:, cols]

attributeNames = np.asarray(df.columns[cols])
classLabels = raw_data[:, -1]  
classNames = np.unique(classLabels)
classDict = dict(zip(classNames, range(len(classNames))))

y = np.array([classDict[cl] for cl in classLabels])


N, M = X.shape
C = len(classNames)


############ STATISTICAL EVALUATION

loss = 2

K = 10 # We presently set J=K
m = 1
r = []
kf = model_selection.KFold(n_splits=K)


dist = 2
metric = "minkowski"
metric_params = {}  # no parameters needed for minkowski


y_true = []
yhat = []

for dm in range(m):


    for train_index, test_index in kf.split(X):
        X_train, y_train = X[train_index,:], y[train_index]
        X_test, y_test = X[test_index, :], y[test_index]

        mA = sklearn.linear_model.LinearRegression().fit(X_train, y_train)
        mB = sklearn.tree.DecisionTreeRegressor().fit(X_train, y_train)

        yhatA = mA.predict(X_test)
        yhatB = mB.predict(X_test)[:, np.newaxis]  # justsklearnthings
        y_true.append(y_test)
        #yhat.append(np.concatenate([yhatA, yhatB], axis=1))

        #r.append( np.mean( np.abs( yhatA-y_test ) ** loss - np.abs( yhatB-y_test) ** loss ) )

# Initialize parameters and run test appropriate for setup II
alpha = 0.05
rho = 1/K
#p_setupII, CI_setupII = correlated_ttest(r, rho, alpha=alpha)
thetahat, CI_setupII, p_setupII = mcnemar(y_true, yhatA, yhatB, alpha)

print( p_setupII )
print(CI_setupII)