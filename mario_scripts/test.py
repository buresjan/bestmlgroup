# exercise 1.5.1
import importlib_resources
import numpy as np
import pandas as pd

# Load the dataset file using pandas
df = pd.read_csv("HeartDisease.csv")

raw_data = df.values

#print(df)

# We start by making the data matrix X by indexing into data.
# We know that the attributes are stored in the columns from inspecting
# the file.
cols = range(0, 9)
X = raw_data[:, cols]

#print(X)

# We can extract the attribute names that came from the header of the csv
attributeNames = np.asarray(df.columns[cols])

#print(attributeNames)

# Before we can store the class index, we need to convert the strings that
# specify the class of a given object to a numerical value. We start by
# extracting the strings for each sample from the raw data loaded from the csv:
classLabels = raw_data[:, -1]  # -1 takes the last column

#print (classLabels)
# Then determine which classes are in the data by finding the set of
# unique class labels
classNames = np.unique(classLabels)


# We can assign each type of Iris class with a number by making a
# Python dictionary as so:

classDict = dict(zip(classNames, range(len(classNames))))
#print(classDict)

# The function zip simply "zips" togetter the classNames with an integer,
# like a zipper on a jacket.
# For instance, you could zip a list ['A', 'B', 'C'] with ['D', 'E', 'F'] to
# get the pairs ('A','D'), ('B', 'E'), and ('C', 'F').
# A Python dictionary is a data object that stores pairs of a key with a value.
# This means that when you call a dictionary with a given key, you
# get the stored corresponding value. Try highlighting classDict and press F9.
# You'll see that the first (key, value)-pair is ('Iris-setosa', 0).
# If you look up in the dictionary classDict with the value 'Iris-setosa',
# you will get the value 0. Try it with classDict['Iris-setosa']

# With the dictionary, we can look up each data objects class label (the string)
# in the dictionary, and determine which numerical value that object is
# assigned. This is the class index vector y:

y = np.array([classDict[cl] for cl in classLabels])

# In the above, we have used the concept of "list comprehension", which
# is a compact way of performing some operations on a list or array.
# You could read the line  "For each class label (cl) in the array of
# class labels (classLabels), use the class label (cl) as the key and look up
# in the class dictionary (classDict). Store the result for each class label
# as an element in a list (because of the brackets []). Finally, convert the
# list to a numpy array".
# Try running this to get a feel for the operation:
# list = [0,1,2]
# new_list = [element+10 for element in list]

# We can determine the number of data objects and number of attributes using
# the shape of X
N, M = X.shape

# Finally, the last variable that we need to have the dataset in the
# "standard representation" for the course, is the number of classes, C:
C = len(classNames)



# exercise 8.1.2

import importlib_resources
import matplotlib.pyplot as plt
import numpy as np
from scipy.io import loadmat
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

from dtuimldmtools import confmatplot, rocplot
# Create crossvalidation partition for evaluation
# using stratification and 95 pct. split between training and test
K = 20
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.95, stratify=y)
# Try to change the test_size to e.g. 50 % and 99 % - how does that change the
# effect of regularization? How does differetn runs of  test_size=.99 compare
# to eachother?

# Standardize the training and set set based on training set mean and std
mu = np.mean(X_train, 0)
sigma = np.std(X_train, 0)

X_train = (X_train - mu) / sigma
X_test = (X_test - mu) / sigma

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
    y_test_est = mdl.predict(X_test).T

    train_error_rate[k] = np.sum(y_train_est != y_train) / len(y_train)
    test_error_rate[k] = np.sum(y_test_est != y_test) / len(y_test)

    w_est = mdl.coef_[0]
    coefficient_norm[k] = np.sqrt(np.sum(w_est**2))

min_error = np.min(test_error_rate)
opt_lambda_idx = np.argmin(test_error_rate)
opt_lambda = lambda_interval[opt_lambda_idx]

plt.figure(figsize=(8, 8))
# plt.plot(np.log10(lambda_interval), train_error_rate*100)
# plt.plot(np.log10(lambda_interval), test_error_rate*100)
# plt.plot(np.log10(opt_lambda), min_error*100, 'o')
plt.semilogx(lambda_interval, train_error_rate * 100)
plt.semilogx(lambda_interval, test_error_rate * 100)
plt.semilogx(opt_lambda, min_error * 100, "o")
plt.text(
    1e-8,
    3,
    "Minimum test error: "
    + str(np.round(min_error * 100, 2))
    + " % at 1e"
    + str(np.round(np.log10(opt_lambda), 2)),
)
# plt.xlabel("Regularization strength, $\log_{10}(\lambda)$")
# plt.ylabel("Error rate (%)")
# plt.title("Classification error")
# plt.legend(["Training error", "Test error", "Test minimum"], loc="upper right")
# plt.ylim([0, 4])
# plt.grid()
# plt.show()



print(lambda_interval)
print(test_error_rate)
