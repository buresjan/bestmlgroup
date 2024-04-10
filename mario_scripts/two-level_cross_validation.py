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
import matplotlib.pyplot as plt
import numpy as np
from scipy.io import loadmat
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

from dtuimldmtools import confmatplot, rocplot
# Create crossvalidation partition for evaluation
# using stratification and 95 pct. split between training and test
K_1 = 10
K_2 = 10


for k_outer in range(0,K_1): #goes from 0 to 9

    X_par, X_test, y_par, y_test = train_test_split(X, y, test_size=(1-1/K_1), stratify=y)

    # Standardize the training and set set based on training set mean and std
    mu = np.mean(X_par, 0)
    sigma = np.std(X_par, 0)

    X_par = (X_par - mu) / sigma
    X_test = (X_test - mu) / sigma   #why am I normalizing it with parmeters from different part of the dataset?

    log_regress_errors = np.zeros(K_1)
    log_regress_best_lambda = np.zeros(K_1)


    for k_inner in range(0, K_2):



        X_train, X_validation, y_train, y_validation = train_test_split(X_par, y_par, test_size=(1-1/K_2), stratify=y_par)
    
        
        # Fit regularized logistic regression model to training data to predict
        # the type of wine
        lambda_interval = np.logspace(-8, 2, 50)
        train_error_rate = np.zeros(len(lambda_interval))
        test_error_rate = np.zeros(len(lambda_interval))
        coefficient_norm = np.zeros(len(lambda_interval))
        for k in range(0, len(lambda_interval)):
            mdl = LogisticRegression(penalty="l2", C=1 / lambda_interval[k])

            mdl.fit(X_train, y_train)

            y_train_est = mdl.predict(X_train).T
            y_validation_est = mdl.predict(X_validation).T

            train_error_rate[k] = np.sum(y_train_est != y_train) / len(y_train)
            test_error_rate[k] = np.sum(y_validation_est != y_validation) / len(y_validation)

            w_est = mdl.coef_[0]
            coefficient_norm[k] = np.sqrt(np.sum(w_est**2))

        min_error = np.min(test_error_rate)
        opt_lambda_idx = np.argmin(test_error_rate)
        opt_lambda = lambda_interval[opt_lambda_idx]

        log_regress_errors[k_inner] = min_error
        log_regress_best_lambda [k_inner] = opt_lambda


        # string = "For inner loop number "+ str(k_inner) + " and outer loop number " + str(k_outer) + " the best error rate was " + str(min_error) + ", found for lamba " + str(opt_lambda)
        # print(string)

    min_error = np.min(log_regress_errors)
    opt_lambda_idx = np.argmin(log_regress_errors)
    opt_lambda = lambda_interval[opt_lambda_idx]

    string = "For outer loop number " + str(k_outer) + " the best error rate was " + str(min_error) + ", found for lamba " + str(opt_lambda)
    print(string)    
    