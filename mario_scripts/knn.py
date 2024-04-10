# exercise 1.5.1
import importlib_resources
import numpy as np
import pandas as pd

# Load the dataset file using pandas
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



K_1 = 10
K_2 = 10


for k_outer in range(0,K_1): #goes from 0 to 9

    X_par, X_test, y_par, y_test = train_test_split(X, y, test_size=0.2, stratify=y)

    # Standardize the training and set set based on training set mean and std
    mu = np.mean(X_par, 0)
    sigma = np.std(X_par, 0)

    X_par = (X_par - mu) / sigma
    X_test = (X_test - mu) / sigma   #why am I normalizing it with parmeters from different part of the dataset?

    knn_errors = np.zeros(K_1)
    knn_best_k = np.zeros(K_1)


    for k_inner in range(0, K_2):



        X_train, X_validation, y_train, y_validation = train_test_split(X_par, y_par, test_size=0.2, stratify=y_par)
    
        
        # Fit regularized logistic regression model to training data to predict
        # the type of wine
        k_interval = np.arange(1, 50, 1)
        train_error_rate = np.zeros(len(k_interval))
        test_error_rate = np.zeros(len(k_interval))

        for k in range(0, len(k_interval)):


            dist = 2
            metric = "minkowski"
            metric_params = {}  # no parameters needed for minkowski

            knclassifier = KNeighborsClassifier(n_neighbors=k_interval[k], p=dist, metric=metric, metric_params=metric_params)
            knclassifier.fit(X_train, y_train)
            y_validation_est = knclassifier.predict(X_validation)


            test_error_rate[k] = np.sum(y_validation_est != y_validation) / len(y_validation)

        min_error = np.min(test_error_rate)
        opt_k_idx = np.argmin(test_error_rate)
        opt_k = k_interval[opt_k_idx]

        knn_errors[k_inner] = min_error
        knn_best_k [k_inner] = opt_k


        # string = "For inner loop number "+ str(k_inner) + " and ouSter loop number " + str(k_outer) + " the best error rate was " + str(min_error) + ", found for lamba " + str(opt_lambda)
        # print(string)

    min_error = np.min(knn_errors)
    opt_k_idx = np.argmin(knn_errors)
    opt_k = k_interval[opt_k_idx]

    string = "For outer loop number " + str(k_outer) + " the best error rate was " + str(min_error) + ", found for k-parameter " + str(opt_k)
    print(string)    
    



