import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn
import torch
from sklearn import model_selection
from scipy import stats
from dtuimldmtools import train_neural_net, correlated_ttest
from part2utils import data_to_numerical, mean_squared_error, baseline_model_predict, generate_folds, generate_d_test_sizes, lin_reg_validate

data = pd.read_csv("HeartDisease.csv")
X = data_to_numerical(data, skip_feature=['tobacco'])
y = data['tobacco'].values.ravel()

results = pd.read_csv("model_comparison_part_b.csv")

## Statistical test settings
loss_in_r_function = 2  ## This implies the loss is squared in the r_j formula of box 11.4.1
r_baseline_vs_linear = []  ## The list to keep the r test size
r_baseline_vs_ANN = []  ## The list to keep the r test size
r_ANN_vs_linear = []  ## The list to keep the r test size
alpha_t_test = 0.05
k = 10
rho_t_test = 1 / k
n_replicates = 1
max_iter = 10000
loss_fn = torch.nn.MSELoss()  # notice how this is now a mean-squared-error loss
M = X.shape[1]

most_common_lambda = stats.mode(np.array(results['lambda']))[0].astype('float64')
# most_common_h = stats.mode(np.array(results['n_hidden_units']))[0].astype('int32')
y_true = []
yhat = []

cv_setup_ii = sklearn.model_selection.KFold(n_splits=k, shuffle=True, random_state=38)  ## Ensures that the CV for setup ii test is never the same randomization as for the estimation CVs
k = 0
for train_index, test_index in cv_setup_ii.split(X, y):
    print('Computing setup II CV K-fold: {0}/{1}..'.format(k + 1, 10))
    X_train, y_train = X[train_index, :], y[train_index]
    X_test, y_test = X[test_index, :], y[test_index]

    model_baseline = np.mean(y_train)
    yhat_baseline = np.ones((y_test.shape[0], 1)) * model_baseline.squeeze()

    # Standardize
    mu_outer = np.mean(X_train[:, 1:], 0)
    sigma_outer = np.std(X_train[:, 1:], 0)

    X_train[:, 1:] = (X_train[:, 1:] - mu_outer) / sigma_outer
    X_test[:, 1:] = (X_test[:, 1:] - mu_outer) / sigma_outer

    Xty_outer = X_train.T @ y_train
    XtX_outer = X_train.T @ X_train

    lambdaI = most_common_lambda * np.eye(M)
    lambdaI[0, 0] = 0
    w_rlr = np.linalg.solve(XtX_outer + lambdaI, Xty_outer).squeeze()

    yhat_linear = X_test @ w_rlr.T  ## use reshape to ensure it is a nested array
    yhat_linear = yhat_linear.reshape(-1, 1)

    X_train_tensor = torch.tensor(X_train)
    y_train_tensor = torch.tensor(y_train.reshape(-1, 1))
    X_test_tensor = torch.tensor(X_test)
    y_test_tensor = torch.tensor(y_test.reshape(-1, 1))

    model_second = lambda: torch.nn.Sequential(
        torch.nn.Linear(M, 9),  # M features to H hidden units
        # 1st transfer function, either Tanh or ReLU:
        # torch.nn.ReLU(),
        torch.nn.Tanh(),
        torch.nn.Linear(9, 1),  # H hidden units to 1 output neuron
        # no final tranfer function, i.e. "linear output"
    )

    ## Run optimization
    net, final_loss, learning_curve = train_neural_net(model_second,
                                                       loss_fn,
                                                       X=X_train_tensor.float(),
                                                       y=y_train_tensor.float(),
                                                       n_replicates=n_replicates,
                                                       max_iter=max_iter)

    # Determine estimated regression value for test set
    yhat_ann = net(X_test_tensor)
    yhat_ann = yhat_ann.detach().numpy()

    ## Add true classes and store estimated classes
    y_true.append(y_test)
    yhat.append(np.concatenate([yhat_baseline, yhat_linear, yhat_ann], axis=1))

    ## Compute the r test size and store it
    r_baseline_vs_linear.append(np.mean(np.abs(yhat_baseline - y_test) ** loss_in_r_function - np.abs(
        yhat_linear - y_test) ** loss_in_r_function))
    r_baseline_vs_ANN.append(np.mean(
        np.abs(yhat_baseline - y_test) ** loss_in_r_function - np.abs(yhat_ann - y_test) ** loss_in_r_function))
    r_ANN_vs_linear.append(np.mean(
        np.abs(yhat_ann - y_test) ** loss_in_r_function - np.abs(yhat_linear - y_test) ** loss_in_r_function))

    ## add to counter
    k += 1


## Baseline vs logistic regression
p_setupII_base_vs_linear, CI_setupII_base_vs_linear = correlated_ttest(r_baseline_vs_linear, rho_t_test,
                                                                       alpha=alpha_t_test)

## Baseline vs 2nd model
p_setupII_base_vs_ANN, CI_setupII_base_vs_ANN = correlated_ttest(r_baseline_vs_ANN, rho_t_test,
                                                                 alpha=alpha_t_test)

## Logistic regression vs 2nd model
p_setupII_ANN_vs_linear, CI_setupII_ANN_vs_linear = correlated_ttest(r_ANN_vs_linear, rho_t_test,
                                                                     alpha=alpha_t_test)

## Create output table for statistic tests
df_output_table_statistics = pd.DataFrame(np.ones((3, 5)),
                                          columns=['H_0', 'p_value', 'CI_lower', 'CI_upper', 'conclusion'])
df_output_table_statistics['H_0'] = ['err_baseline-err_linear=0', 'err_ANN-err_linear=0',
                                       'err_baseline-err_ANN=0']
df_output_table_statistics['p_value'] = [p_setupII_base_vs_linear, p_setupII_ANN_vs_linear,
                                          p_setupII_base_vs_ANN]
df_output_table_statistics['CI_lower'] = [CI_setupII_base_vs_linear[0], CI_setupII_ANN_vs_linear[0],
                                           CI_setupII_base_vs_ANN[0]]
df_output_table_statistics['CI_upper'] = [CI_setupII_base_vs_linear[1], CI_setupII_ANN_vs_linear[1],
                                            CI_setupII_base_vs_ANN[1]]
rejected_null = (df_output_table_statistics.loc[:, 'p_value'] < alpha_t_test)
df_output_table_statistics['conclusion'] = df_output_table_statistics['conclusion'].astype('str')
df_output_table_statistics.loc[rejected_null, 'conclusion'] = 'H_0 rejected'
df_output_table_statistics.loc[~rejected_null, 'conclusion'] = 'H_0 not rejected'
df_output_table_statistics = df_output_table_statistics.set_index('H_0')

## Export df as csv
df_output_table_statistics.to_csv('statistic_test_part_b.csv', encoding='UTF-8')