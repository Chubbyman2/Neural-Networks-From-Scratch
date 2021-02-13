import numpy as np
import matplotlib.pyplot as plt
import random


# Activation Functions
# For each of the activation functions, you're gonna have to implement the derivatives manually
def tanh(x):
    return np.tanh(x)


def d_tanh(x):
    return 1 - np.square(np.tanh(x))


def sigmoid(x):
    return 1/(1 + np.exp(-x))


def d_sigmoid(x):
    return (1 - sigmoid(x)) * sigmoid(x)


# Loss Functions - Logloss (a.k.a. cross-entropy) is used here
# Implement the derivative manually here, as well
def logloss(y, a):
    return -(y * np.log(a) + (1-y) * np.log(1-a))


def d_logloss(y, a):
    return (a-y)/(a * (1-a))


class Layer:
    # Dictionary of activation functions and their derivatives
    activation_fns = {
        "tanh": (tanh, d_tanh),
        "sigmoid": (sigmoid, d_sigmoid)
    }

    def __init__(self, inputs, neurons, activation, learning_rate=1):
        # Creates a random matrix with shape (# neurons, # inputs)
        self.w = np.random.randn(neurons, inputs)

        # Creates a zero matrix with shape (# neurons, 1)
        self.b = np.zeros((neurons, 1))

        # Retrieves the specified activation function from the dict
        self.activation, self.d_activation = self.activation_fns.get(
            activation)

        self.lr = learning_rate

    # a = activation_fn(z)
    # z = w * a_prev + b
    # * is dot product
    def feedforward(self, a_prev):
        self.a_prev = a_prev
        self.z = np.dot(self.w, self.a_prev) + self.b
        self.a = self.activation(self.z)
        return self.a

    def backpropagate(self, da):
        # All derivative values here are w/ respect to cost
        # Refer to the five derivatives involved in backpropagation
        dz = np.multiply(self.d_activation(self.z), da)
        dw = 1/dz.shape[1] * np.dot(dz, self.a_prev.T)  # m = 1/dz.shape[1]

        # Adds column vectors of dz to reduce it to shape (neurons, 1)
        db = 1/dz.shape[1] * np.sum(dz, axis=1, keepdims=True)

        da_prev = np.dot(self.w.T, dz)

        # Refer to the two update equations
        self.w -= self.lr * dw
        self.b -= self.lr * db

        return da_prev


if __name__ == "__main__":

    # Demo
    plt.style.use("seaborn")
    width = 500
    height = 500
    x_train = [[], []]
    y_train = [[]]
    colour_labels = []
    m = 100  # Num samples
    epochs = 10000

    # Generate 100 random data points with labels
    # Output of feedforward will be 0 or 1 (cuz output is sigmoid)
    for i in range(100):
        x1 = random.randint(0, width)
        x2 = random.randint(0, height)
        x_train[0].append(x1)
        x_train[1].append(x2)
        if x1 >= x2:
            y_train[0].append(1)
            colour_labels.append("r")  # This will correspond to 1
        else:
            y_train[0].append(0)
            colour_labels.append("b")  # This will correspond to 0

    # Layer(inputs, neurons, activation_fn)
    network = [Layer(2, 32, "tanh"),  # 2 arrays of 100 elements each
               Layer(32, 1, "sigmoid")]
    costs = []
    x_train = np.asarray(x_train)
    y_train = np.asarray(y_train)

    for epoch in range(epochs):
        a = x_train
        for layer in network:
            # Stores output of feedforward layer as 'a' for next layer
            a = layer.feedforward(a)

        cost = 1/m * np.sum(logloss(y_train, a))
        costs.append(cost)  # This is for loss visualization later

        da = d_logloss(y_train, a)

        # Backpropagation
        for layer in reversed(network):
            # Stores output of backpropagation as 'da' for prev layer
            da = layer.backpropagate(da)

    # Make prediction
    a = x_train
    for layer in network:
        a = layer.feedforward(a)  # a is now a matrix with the predicted values
    # print(y_train)
    # print(a)

    # Process prediction into correct labels
    predicted_labels = []
    for value in a[0]:
        if round(value) == 0:
            predicted_labels.append("b")
        elif round(value) == 1:
            predicted_labels.append("r")

    # Display final loss value
    print(f"Final Loss: {costs[-1]}")

    num_correct = 0
    for i in range(len(colour_labels)):
        if colour_labels[i] == predicted_labels[i]:
            num_correct += 1
    print(f"Number of Correct Predictions: {num_correct}")

    # Visualize
    plot1 = plt.figure(1)
    plt.scatter(x_train[0], x_train[1], c=colour_labels)
    plt.plot([0, width], [0, height], 'k-', color='k')
    plt.title("Actual Values")

    plot2 = plt.figure(2)
    plt.scatter(x_train[0], x_train[1], c=predicted_labels)
    plt.plot([0, width], [0, height], 'k-', color='k')
    plt.title("Predicted Values")

    # Plot loss
    plot3 = plt.figure(3)
    plt.plot(range(epochs), costs)
    plt.ylim(ymin=0)
    plt.show()
