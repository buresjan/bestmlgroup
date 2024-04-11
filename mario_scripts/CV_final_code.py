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
from dtuimldmtools import confmatplot, rocplot, statistics
from dtuimldmtools import *

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

######################################

K_1 = 10
K_2 = 10

dataframe_baseline = np.ones(10)
dataframe_log_error = np.ones(10)
dataframe_log_lambda = np.ones(10)
dataframe_knn_error = np.ones(10)
dataframe_knn_par = np.ones(10)



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

        #### KNN

        k_interval = np.arange(1, 20, 1)
        knn_test_error_rate = np.zeros(len(k_interval))

        for k in range(0, len(k_interval)):


            dist = 2
            metric = "minkowski"
            metric_params = {}  # no parameters needed for minkowski

            knclassifier = KNeighborsClassifier(n_neighbors=k_interval[k], p=dist, metric=metric, metric_params=metric_params)
            knclassifier.fit(X_train, y_train)
            y_validation_est = knclassifier.predict(X_validation)


            knn_test_error_rate[k] = np.sum(y_validation_est != y_validation) / len(y_validation)

        knn_min_error = np.min(knn_test_error_rate)
        opt_k_idx = np.argmin(knn_test_error_rate)
        opt_k = k_interval[opt_k_idx]

        knn_errors[k_inner] = knn_min_error
        knn_best_k [k_inner] = opt_k




        ##### LOGISTIC REGRESSION

        log_regress_errors = np.zeros(K_1)
        log_regress_best_lambda = np.zeros(K_1)

         # Fit regularized logistic regression model to training data to predict
        # the type of wine
        lambda_interval = np.logspace(-16, 1, 50)

        log_regress_test_error_rate = np.zeros(len(lambda_interval))
        for k in range(0, len(lambda_interval)):
            mdl = LogisticRegression(penalty="l2", C=1 / lambda_interval[k])

            mdl.fit(X_train, y_train)

            y_validation_est = mdl.predict(X_validation).T

            log_regress_test_error_rate[k] = np.sum(y_validation_est != y_validation) / len(y_validation)

        log_regress_min_error = np.min(log_regress_test_error_rate)
        opt_lambda_idx = np.argmin(log_regress_test_error_rate)
        opt_lambda = lambda_interval[opt_lambda_idx]

        log_regress_errors[k_inner] = log_regress_min_error
        log_regress_best_lambda [k_inner] = opt_lambda





        # string = "For inner loop number "+ str(k_inner) + " and ouSter loop number " + str(k_outer) + " the best error rate was " + str(knn_min_error) + ", found for lamba " + str(opt_lambda)
        # print(string)


    #### Second level caracterization

    #knn

    knn_min_error = np.min(knn_errors)
    opt_k_idx = np.argmin(knn_errors)
    opt_k = k_interval[opt_k_idx]

    dist = 2
    metric = "minkowski"
    metric_params = {}  # no parameters needed for minkowski

    knclassifier = KNeighborsClassifier(n_neighbors=opt_k, p=dist, metric=metric, metric_params=metric_params)
    knclassifier.fit(X_par, y_par)
    y_test_est = knclassifier.predict(X_test).T
    knn_final_yest = y_test_est
    knn_final_error = np.sum(y_test_est != y_test) / len(y_test)
    knn_final_k = opt_k

    dataframe_knn_error [k_outer] = knn_final_error
    dataframe_knn_par [k_outer] = knn_final_k


    #log_regre

    log_regress_min_error = np.min(log_regress_errors)
    opt_lambda_idx = np.argmin(log_regress_errors)
    opt_lambda = lambda_interval[opt_lambda_idx]

    mdl = LogisticRegression(penalty="l2", C=1 / lambda_interval[k])
    mdl.fit(X_par, y_par)

    y_test_est = mdl.predict(X_test).T
    log_regress_final_yest = y_test_est

    log_regress_final_error = np.sum(y_test_est != y_test) / len(y_test)
    log_regress_final_lambda = opt_lambda

    dataframe_log_error [k_outer] = log_regress_final_error
    dataframe_log_lambda [k_outer] = log_regress_final_lambda


    # baseline
    y_baseline_est = np.full((y_test.size), np.bincount(y).argmax())
    misclass_rate_baseline = np.sum(y_baseline_est != y_test) / float(len(y_baseline_est))
    #print(np.mean(y_test == 1))
    dataframe_baseline [k_outer] = misclass_rate_baseline



    string1 = "(knn) For outer loop number " + str(k_outer) + " the best error rate was " + str(knn_final_error) + ", found for k-parameter " + str(knn_final_k)
    string2 = "(log) For outer loop number " + str(k_outer) + " the best error rate was " + str(log_regress_final_error) + ", found for lamba " + str(log_regress_final_lambda)
    string3 = "(bas) For outer loop number " + str(k_outer) + " the error rate was " + str(misclass_rate_baseline)
    print(string1)
    print(string2)    
    print(string3)


    print("################################### STATISTICAL EVALUATION")
    print("Log regress vs KNN")
    thetahat, CI_setupII, p_setupII = mcnemar(y_true = y_test, yhatA= log_regress_final_yest , yhatB = knn_final_yest, alpha = 0.05)

    # print(thetahat)
    # print( p_setupII )
    # print(CI_setupII)

    print("\n")

    print("Log regress vs Baseline")
    thetahat, CI_setupII, p_setupII = mcnemar(y_true = y_test, yhatA= log_regress_final_yest , yhatB = y_baseline_est, alpha = 0.05)

    # print(thetahat)
    # print( p_setupII )
    # print(CI_setupII)

    print("\n")
    
    print("baseline vs KNN")
    thetahat, CI_setupII, p_setupII = mcnemar(y_true = y_test, yhatA= y_baseline_est , yhatB = knn_final_yest, alpha = 0.05)

    # print(thetahat)
    # print( p_setupII )
    # print(CI_setupII)

    print("\n")
n_of_cols = 5
n_of_index = 10
df_output_table = pd.DataFrame(np.ones((n_of_index, n_of_cols)), index=range(1, n_of_index + 1))
df_output_table.index.name = "Outer fold"

df_output_table.columns = ['baseline_test_error', 'logistic_test_error', 'lambda', 'KNN_test_error', 'k-parameter']


## Add values to columns in output table
df_output_table['baseline_test_error'] = dataframe_baseline
df_output_table['logistic_test_error'] = dataframe_log_error
df_output_table['lambda'] = dataframe_log_lambda
df_output_table['KNN_test_error'] = dataframe_knn_error
df_output_table['k-parameter'] = dataframe_knn_par

df_output_table.to_csv('table2.csv')

#np.savetxt("table2.txt", df_output_table)



