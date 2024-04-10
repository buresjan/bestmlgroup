import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn
import torch
from sklearn import model_selection
from scipy import stats
from dtuimldmtools import train_neural_net, correlated_ttest
from part2utils import data_to_numerical, mean_squared_error, baseline_model_predict, generate_folds, generate_d_test_sizes, lin_reg_validate


def baseline_validate(X, y, cv):
    errors_baseline = []

    for train_index, test_index in cv.split(X, y):
        # Splitting the data for the outer loop
        X_train_outer, X_test_outer = X[train_index], X[test_index]
        y_train_outer, y_test_outer = y[train_index], y[test_index]

        # Evaluate the baseline model on the outer test fold
        y_pred_baseline = baseline_model_predict(y_train_outer, X_test_outer)
        error_baseline = mean_squared_error(y_test_outer, y_pred_baseline)
        errors_baseline.append(error_baseline)

    return errors_baseline


def nn_validate(X, y, outer_cv, inner_cv, hidden_units, k_1, k_2):
    errors_ann = []
    optimal_h = []

    y = y.reshape(-1, 1)

    # Parameters for neural network
    n_replicates = 1  # number of networks trained in each k-fold
    max_iter = 10000
    loss_fn = torch.nn.MSELoss()  # notice how this is now a mean-squared-error loss
    M = X.shape[1]

    k_outer = 0
    for train_index, test_index in outer_cv.split(X, y):
        print('Computing CV outer fold: {0}/{1}..'.format(k_outer + 1, k_1))
        k_outer += 1
        # Splitting the data for the outer loop
        X_train_outer, X_test_outer = X[train_index, :], X[test_index, :]
        y_train_outer, y_test_outer = y[train_index], y[test_index]
        d_par_size = (float(len(y_train_outer)))

        f = 0
        ann_test_error = np.empty((k_2, len(hidden_units)))
        d_val_sizes = []

        for inner_train_index, inner_test_index in inner_cv.split(X_train_outer, y_train_outer):
            print('Computing CV inner fold: {0}/{1}..'.format(f + 1, k_2))
            X_train_inner = X_train_outer[inner_train_index, :]  # D train
            y_train_inner = y_train_outer[inner_train_index]
            X_test_inner = X_train_outer[inner_test_index, :]  # D val
            y_test_inner = y_train_outer[inner_test_index]
            # Standardize
            mu = np.mean(X_train_inner[:, 1:], 0)
            sigma = np.std(X_train_inner[:, 1:], 0)

            X_train_inner[:, 1:] = (X_train_inner[:, 1:] - mu) / sigma
            X_test_inner[:, 1:] = (X_test_inner[:, 1:] - mu) / sigma

            d_val_size = y_test_inner.shape[0]
            d_val_sizes.append(d_val_size)

            # Extract training and test set for current CV fold, convert to tensors
            X_train_inner_tensor = torch.Tensor(X_train_inner)
            y_train_inner_tensor = torch.Tensor(y_train_inner)
            X_test_inner_tensor = torch.Tensor(X_test_inner)
            y_test_inner_tensor = torch.Tensor(y_test_inner)

            for h in range(0, len(hidden_units)):
                # Compute parameters for current value of hidden units and current CV fold
                n_hidden_units = hidden_units[h]

                # Define the model
                model = lambda: torch.nn.Sequential(
                    torch.nn.Linear(M, n_hidden_units),  # M features to n_hidden_units
                    torch.nn.Tanh(),  # 1st transfer function,
                    torch.nn.Linear(n_hidden_units, 1),  # n_hidden_units to 1 output neuron
                    # no final transfer function, i.e. "linear output"
                )


                # Train the net on training data
                net, final_loss, learning_curve = train_neural_net(
                    model,
                    loss_fn,
                    X=X_train_inner_tensor,
                    y=y_train_inner_tensor,
                    n_replicates=n_replicates,
                    max_iter=max_iter,
                )

                # Determine estimated predictions for train and test set
                y_test_est = net(X_test_inner_tensor)

                # Evaluate training and test performance
                ann_test_error[f, h] = mean_squared_error(y_test_inner_tensor.detach().numpy(), y_test_est.detach().numpy())

            f = f + 1

        d_val_sizes = np.array(d_val_sizes)
        outer_scale_factor_ary = d_val_sizes / d_par_size
        outer_scale_factor_ary = outer_scale_factor_ary[:, np.newaxis]
        summed_scaled_errors = np.sum(ann_test_error * outer_scale_factor_ary, axis=0)
        min_idx = np.argmin(summed_scaled_errors)
        opt_h = hidden_units[min_idx]
        optimal_h.append(opt_h)

        mu_outer = np.mean(X_train_outer[:, 1:], 0)
        sigma_outer = np.std(X_train_outer[:, 1:], 0)

        X_train_outer[:, 1:] = (X_train_outer[:, 1:] - mu_outer) / sigma_outer
        X_test_outer[:, 1:] = (X_test_outer[:, 1:] - mu_outer) / sigma_outer

        X_train_outer_tensor = torch.Tensor(X_train_outer)
        y_train_outer_tensor = torch.Tensor(y_train_outer)
        X_test_outer_tensor = torch.Tensor(X_test_outer)
        y_test_outer_tensor = torch.Tensor(y_test_outer)

        # Define the model
        model_outer = lambda: torch.nn.Sequential(
            torch.nn.Linear(M, opt_h),  # M features to n_hidden_units
            torch.nn.Tanh(),  # 1st transfer function,
            torch.nn.Linear(opt_h, 1),  # n_hidden_units to 1 output neuron
            # no final transfer function, i.e. "linear output"
        )

        # Train the net on training data
        net, final_loss, learning_curve = train_neural_net(
            model_outer,
            loss_fn,
            X=X_train_outer_tensor,
            y=y_train_outer_tensor,
            n_replicates=n_replicates,
            max_iter=max_iter,
        )

        # Determine estimated predictions for train and test set
        y_test_outer_est = net(X_test_outer_tensor)

        error_ann = mean_squared_error(y_test_outer_tensor.detach().numpy(), y_test_outer_est.detach().numpy())
        errors_ann.append(error_ann)

    return errors_ann, optimal_h


def two_level_cross_validation(X, y, lambdas, hidden_units, k_1=10, k_2=10, random_seed=37):
    outer_cv, inner_cv = generate_folds(k_1, k_2, random_seed)
    d_test_sizes = generate_d_test_sizes(X, y, outer_cv)
    print(d_test_sizes)
    N = np.shape(X)[0]

    errors_baseline = baseline_validate(X, y, outer_cv)
    print("Finished baseline validation.")
    errors_rlr, optimal_lambdas = lin_reg_validate(X, y, outer_cv, inner_cv, lambdas, k_2)
    print("Finished regression validation.")
    errors_ann, optimal_layers = nn_validate(X, y, outer_cv, inner_cv, hidden_units, k_1, k_2)
    print("Finished ANN validation.")

    print("Errors for baseline are:", errors_baseline)
    print("Errors for regression are:", errors_rlr)
    print("Errors for ANN are:", errors_ann)
    print()
    return {
        'data_size': d_test_sizes,
        'errors_baseline': errors_baseline,
        'errors_rlr': errors_rlr,
        'errors_ANN': errors_ann,
        'baseline_mean_error': np.sum(np.multiply(errors_baseline, d_test_sizes)) / N,
        'regularized_lr_mean_error': np.sum(np.multiply(errors_rlr, d_test_sizes)) / N,
        'optimal_lambdas': optimal_lambdas,
        'ann_error': np.sum(np.multiply(errors_ann, d_test_sizes)) / N,
        'optimal_h': optimal_layers,
    }


if __name__ == "__main__":
    # Load your dataset
    data = pd.read_csv("HeartDisease.csv")
    X = data_to_numerical(data, skip_feature=['tobacco'])
    y = data['tobacco'].values.ravel()

    n_of_cols = 6
    n_of_index = 10
    df_output_table = pd.DataFrame(np.ones((n_of_index, n_of_cols)), index=range(1, n_of_index + 1))
    df_output_table.index.name = "Outer fold"

    df_output_table.columns = ['test_data_size', 'baseline_test_error', 'linear_test_error', 'lambda', 'ANN_test_error',
                               'n_hidden_units']


    lambdas = np.array(range(10, 310, 10))
    # lambdas = np.array(range(0, 1000, 20))
    # hidden_units = np.array([1, 2, 3, 4])
    # hidden_units = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15])
    hidden_units = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    results = two_level_cross_validation(X, y, lambdas, hidden_units, 10, 10, 37)


    ## Add values to columns in output table
    df_output_table['test_data_size'] = results['data_size']
    df_output_table['n_hidden_units'] = results['optimal_h']
    df_output_table['ANN_test_error'] = results['errors_ANN']
    df_output_table['lambda'] = results['optimal_lambdas']
    df_output_table['linear_test_error'] = results['errors_rlr']
    df_output_table['baseline_test_error'] = results['errors_baseline']

    df_output_table.to_csv('model_comparison.csv')

    res_ary = np.array([results['baseline_mean_error'], results['regularized_lr_mean_error'], results['ann_error']])

    np.savetxt("generalization_errors.txt", res_ary)