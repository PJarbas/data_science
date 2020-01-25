import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

np.random.seed(42)


def plot_points(X, y):
    admitted = X[np.argwhere(y == 1)]
    rejected = X[np.argwhere(y == 0)]
    plt.scatter([s[0][0] for s in rejected],
                [s[0][1] for s in rejected],
                s = 25, color = 'blue',
                edgecolor='k')

    plt.scatter([s[0][0] for s in admitted],
                [s[0][1] for s in admitted],
                s=25, color='red',
                edgecolor='k')


def display(m, b, color='g--'):
    plt.xlim(-0.05, 1.05)
    plt.ylim(-0.05, 1.05)
    x = np.arange(-10, 10, 0.1)
    plt.plot(x, m*x+b, color)


def step_function(t):
    if t >= 0:
        return 1
    return 0


def prediction(X, W, b):
    return step_function((np.matmul(X, W)+b)[0])


# TODO: Fill in the code below to implement the perceptron trick.
# The function should receive as inputs the data X, the labels y,
# the weights W (as an array), and the bias b,
# update the weights and bias W, b, according to the perceptron algorithm,
# and return W and b.
def perceptron_step(X, y, W, b, learn_rate=0.1):

    for i in range(len(X)):
        y_hat = prediction(X[i], W, b)
        if y[i]-y_hat == 1:
            W[0] += X[i][0]*learn_rate
            W[1] += X[i][1]*learn_rate
            b += learn_rate
        elif y[i]-y_hat == -1:
            W[0] -= X[i][0]*learn_rate
            W[1] -= X[i][1]*learn_rate
            b -= learn_rate

    return W, b


# This function runs the perceptron algorithm repeatedly on the dataset,
# and returns a few of the boundary lines obtained in the iterations,
# for plotting purposes.
def train_perceptron_algorithm(X, y, learn_rate=0.01, num_epochs=25):
    x_min, x_max = min(X.T[0]), max(X.T[0])
    y_min, y_max = min(X.T[1]), max(X.T[1])
    W = np.array(np.random.rand(2, 1))
    b = np.random.rand(1)[0] + x_max
    # These are the solution lines that get plotted below.
    boundary_lines = []
    for i in range(num_epochs):
        # In each epoch, we apply the perceptron step.
        W, b = perceptron_step(X, y, W, b, learn_rate)
        w = W.squeeze()
        boundary_lines.append([-w[0]/w[1], -b/w[1]])
    return boundary_lines


if __name__ == "__main__":

    data = pd.read_csv('data/data.csv', header=None)
    X = np.array(data[[0, 1]])
    y = np.array(data[2])

    boundary_lines = train_perceptron_algorithm(X, y, learn_rate=0.01, num_epochs=200)

    # Plotting the solution boundary
    plt.title("Solution boundary")

    boundary_lines = np.array(boundary_lines)

    display(boundary_lines[:, 0], boundary_lines[:, 1], 'black')

    # Plotting the data
    plot_points(X, y)
    plt.show()
