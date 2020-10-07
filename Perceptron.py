import numpy as np
import math


class Perceptron:
    weights = []

    def __init__(self, learning_rate=0.01, n_iters=1000, activation_function="tanh"):
        self.lr = learning_rate
        self.n_iters = n_iters
        if activation_function.lower() == "tanh":
            self.activation_func = self.tanh
        elif activation_function.lower() == "relu":
            self.activation_func = self.ReLu
        elif activation_function.lower() == "binary step":
            self.activation_func = self.binary_step
        elif activation_function.lower() == "sigmoid":
            self.activation_func = self.sigmoid

        self.weights = None
        self.bias = None

    def fit(self, X, y):
        # X vector is a matrix, where rows are samples and columns are features
        n_samples, n_features = X.shape

        # init weights
        self.weights = np.zeros(n_features)
        self.bias = 0

        # Ensures y is only 0 or 1
        # Uses list comprehension
        y_ = [1 if i > 0 else 0 for i in y]

        # Loops through for number of iterations
        # Updates weights and bias for each iteration
        for _ in range(self.n_iters):
            for idx, x_i in enumerate(X):
                # Remember, dot product returns a scalar!
                linear_output = np.dot(x_i, self.weights) + self.bias
                y_predicted = self.activation_func(linear_output)

                update = self.lr * (y_[idx] - y_predicted)
                self.weights += update * x_i
                self.bias += update

    def predict(self, X):
        linear_output = np.dot(X, self.weights) + self.bias
        y_predicted = self.activation_func(linear_output)
        return y_predicted

    def ReLu(self, x):
        return np.maximum(x, 0)

    def tanh(self, x):
        return np.tanh(x)

    def binary_step(self, x):
        # Returns 1 if x>=0, otherwise 0
        return np.where(x >= 0, 1, 0)

    def sigmoid(self, x):
        return 1/(1 + math.e**(-x))
