import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import datasets

from Perceptron import Perceptron


def mean_squared_error(y_actual, y_pred):
    for i in range(len(y_pred)):
        sum_of_diff_squared = (y_actual[i] - y_pred[i])**2
    mse = 1/len(y_pred) * sum_of_diff_squared
    return 100 - mse*100


# Generate the dataset
X, y = datasets.make_blobs(n_samples=150, n_features=2,
                           centers=2, cluster_std=1.05, random_state=2)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=123)

list_of_activation_funcs = ["tanh", "relu", "binary step", "sigmoid"]

# Loops through each of the activation functions listed
# Returns mse accuracy of using each
for activation_func in list_of_activation_funcs:
    # Has a default learning_rate and n_iters
    model = Perceptron(activation_function=activation_func)
    model.fit(X_train, y_train)
    prediction = model.predict(X_test)

    print(
        f"Perceptron classification accuracy, {activation_func}: {mean_squared_error(y_test, prediction)}%")
