import numpy as np
import pandas as pd
import sklearn
import torch
from sklearn import model_selection
from dtuimldmtools import train_neural_net


def standardize_data(df, skip_feature=None):
    data = data_to_numerical(df, skip_feature=skip_feature)
    n, m = np.shape(data)

    data_standardized = data - np.ones((n, 1)) * data.mean(axis=0)
    data_standardized /= np.std(data_standardized, axis=0, ddof=1)  # is ddof supposed to be 1??

    return data_standardized


def data_to_numerical(df, skip_feature=None):
    df = df.drop(columns=skip_feature, axis=1)
    data = df.to_numpy(dtype=np.float32)

    return data


def mean_squared_error(y_true, y_pred):
    """
    Calculate the Mean Squared Error between true and predicted values.

    Parameters:
    y_true (array): True values of y.
    y_pred (array): Predicted values of y.

    Returns:
    float: The mean squared error.
    """
    # Calculate the differences between true and predicted values
    error = y_true - y_pred
    # Square the differences
    squared_error = error ** 2
    # Calculate the average of the squared differences
    mse = squared_error.mean()
    return mse


def baseline_model_predict(y_train, X_test):
    """Predict using the mean of y_train.

    Parameters:
    y_train (array): The training target values
    X_test (array): The test data features (unused for predictions)

    Returns:
    array: Predicted values, all set to the mean of y_train
    """
    mean_y_train = np.mean(y_train)
    return np.full(X_test.shape[0], mean_y_train)


def generate_folds(k_1=10, k_2=10, random_seed=37):
    cv_1 = sklearn.model_selection.KFold(n_splits=k_1, shuffle=True, random_state=random_seed)
    cv_2 = sklearn.model_selection.KFold(n_splits=k_2, shuffle=True, random_state=random_seed)

    return cv_1, cv_2


def generate_d_test_sizes(X, y, outer_cv):
    d_test_sizes = []
    for train_index, test_index in outer_cv.split(X, y):
        y_train_outer, y_test_outer = y[train_index], y[test_index]
        d_test_sizes.append(float(len(y_test_outer)))

    return d_test_sizes


def lin_reg_validate(X, y, outer_cv, inner_cv, lambdas, k_2):
    errors_rlr = []
    optimal_lambdas = []

    for train_index, test_index in outer_cv.split(X, y):
        # Splitting the data for the outer loop
        X_train_outer, X_test_outer = X[train_index], X[test_index]
        y_train_outer, y_test_outer = y[train_index], y[test_index]
        d_par_size = (float(len(y_train_outer)))

        f = 0
        rlr_test_error = np.empty((k_2, len(lambdas)))
        d_val_sizes = []

        M = X.shape[1]
        w = np.empty((M, k_2, len(lambdas)))

        # Inner loop:
        for inner_train_index, inner_test_index in inner_cv.split(X_train_outer, y_train_outer):
            print('Computing CV inner fold: {0}/{1}..'.format(f + 1, k_2))
            X_train_inner = X_train_outer[inner_train_index]  # D train
            y_train_inner = y_train_outer[inner_train_index]
            X_test_inner = X_train_outer[inner_test_index]  # D val
            y_test_inner = y_train_outer[inner_test_index]

            # Standardize
            mu = np.mean(X_train_inner[:, 1:], 0)
            sigma = np.std(X_train_inner[:, 1:], 0)

            X_train_inner[:, 1:] = (X_train_inner[:, 1:] - mu) / sigma
            X_test_inner[:, 1:] = (X_test_inner[:, 1:] - mu) / sigma

            # precompute terms
            Xty = X_train_inner.T @ y_train_inner
            XtX = X_train_inner.T @ X_train_inner

            d_val_size = y_test_inner.shape[0]
            d_val_sizes.append(d_val_size)

            for l in range(0, len(lambdas)):
                # Standardizing is performed inside .Ridge
                lambdaI = lambdas[l] * np.eye(M)
                lambdaI[0, 0] = 0  # remove bias regularization
                w[:, f, l] = np.linalg.solve(XtX + lambdaI, Xty).squeeze()
                # Evaluate training and test performance
                rlr_test_error[f, l] = mean_squared_error(y_test_inner, X_test_inner @ w[:, f, l].T)

            f = f + 1

        d_val_sizes = np.array(d_val_sizes)
        outer_scale_factor_ary = d_val_sizes / d_par_size
        outer_scale_factor_ary = outer_scale_factor_ary[:, np.newaxis]
        summed_scaled_errors = np.sum(rlr_test_error * outer_scale_factor_ary, axis=0)
        min_idx = np.argmin(summed_scaled_errors)
        opt_lambda = lambdas[min_idx]
        optimal_lambdas.append(opt_lambda)

        mu_outer = np.mean(X_train_outer[:, 1:], 0)
        sigma_outer = np.std(X_train_outer[:, 1:], 0)

        X_train_outer[:, 1:] = (X_train_outer[:, 1:] - mu_outer) / sigma_outer
        X_test_outer[:, 1:] = (X_test_outer[:, 1:] - mu_outer) / sigma_outer

        Xty_outer = X_train_outer.T @ y_train_outer
        XtX_outer = X_train_outer.T @ X_train_outer

        lambdaI = opt_lambda * np.eye(M)
        lambdaI[0, 0] = 0
        w_rlr = np.linalg.solve(XtX_outer + lambdaI, Xty_outer).squeeze()

        error_rlr = mean_squared_error(y_test_outer, X_test_outer @ w_rlr.T)
        errors_rlr.append(error_rlr)

    return errors_rlr, optimal_lambdas

