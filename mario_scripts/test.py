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



# exercise 6.3.1

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
from sklearn.model_selection import train_test_split


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.95, stratify=y)

# Plot the training data points (color-coded) and test data points.
figure(1)
styles = [".b", ".r", ".g", ".y"]
for c in range(C):
    class_mask = y_train == c
    plot(X_train[class_mask, 0], X_train[class_mask, 1], styles[c])


# K-nearest neighbors
K = 1

# Distance metric (corresponds to 2nd norm, euclidean distance).
# You can set dist=1 to obtain manhattan distance (cityblock distance).
dist = 2
metric = "minkowski"
metric_params = {}  # no parameters needed for minkowski

# You can set the metric argument to 'cosine' to determine the cosine distance
# metric = 'cosine'
# metric_params = {} # no parameters needed for cosine

# To use a mahalonobis distance, we need to input the covariance matrix, too:
# metric='mahalanobis'
# metric_params={'V': cov(X_train, rowvar=False)}

# Fit classifier and classify the test points
knclassifier = KNeighborsClassifier(
    n_neighbors=K, p=dist, metric=metric, metric_params=metric_params
)
knclassifier.fit(X_train, y_train)
y_est = knclassifier.predict(X_test)


# Plot the classfication results
styles = ["ob", "or", "og", "oy"]
for c in range(C):
    class_mask = y_est == c
    plot(X_test[class_mask, 0], X_test[class_mask, 1], styles[c], markersize=10)
    plot(X_test[class_mask, 0], X_test[class_mask, 1], "kx", markersize=8)
title("Synthetic data classification - KNN")

# Compute and plot confusion matrix
cm = confusion_matrix(y_test, y_est)
accuracy = 100 * cm.diagonal().sum() / cm.sum()
error_rate = 100 - accuracy
figure(2)
imshow(cm, cmap="binary", interpolation="None")
colorbar()
xticks(range(C))
yticks(range(C))
xlabel("Predicted class")
ylabel("Actual class")
title(
    "Confusion matrix (Accuracy: {0}%, Error Rate: {1}%)".format(accuracy, error_rate)
)

show()

print("Ran Exercise 6.3.1")


