from helpers import *
import numpy as np


''' MSE loss'''
def compute_loss(y, tx, w):
    e = y - tx.dot(w)
    return 0.5 * np.mean(e ** 2)

''' Gradient of MSE'''
def compute_gradient(y, tx, w):
    e = y - tx.dot(w)
    grad = -tx.T.dot(e) / len(e)
    return grad



''' MSE Gradient descent
    Args:
        y: numpy array of shape (N,), N is the number of samples.
        tx: numpy array of shape (N,D), D is the number of features
        initial_w: initial weight vector
        max_iter: number of iterations
        gamma: step size of the gradient descent
    Returns:
        w : weights
        loss: last loss value from the last iteration
'''
def mean_squared_error_gd(y, tx, initial_w, max_iters, gamma):

    w = initial_w
    for _ in range(max_iters):
        grad = compute_gradient(y, tx, w)
        w = w - gamma * grad

    loss = compute_loss(y, tx, w)
    return w, loss


''' Linear regression using stochastic gradient descent'''


def mean_squared_error_sgd(y, tx, initial_w, max_iters, gamma):
    w = initial_w
    for _ in range(max_iters):
        for y_batch, tx_batch in batch_iter(y, tx, batch_size=1, num_batches=1):
            grad = compute_gradient(y_batch, tx_batch, w)
            w = w - gamma * grad

    loss = compute_loss(y, tx, w)
    return w, loss



''' Least squares using normal equations
    Args:
        y: numpy array of shape (N,), N is the number of samples.
        tx: numpy array of shape (N,D), D is the number of features
    Returns:
        w : weights
        loss: loss value'''

def least_squares(y, tx):
    w = np.linalg.solve(tx.T.dot(tx), tx.T.dot(y))
    loss = compute_loss(y, tx, w)
    return w, loss


''' Ridge regression using normal equations.
    Args:
        y: numpy array of shape (N,), N is the number of samples.
        tx: numpy array of shape (N,D), D is the number of features
    Returns:
        w : weights
        loss: last loss value from the last iteration
        '''


def ridge_regression(y, tx, lambda_):
    lambda_prime = lambda_ * 2 * tx.shape[0]
    a = tx.T.dot(tx) + lambda_prime * np.identity(tx.shape[1])
    b = tx.T.dot(y)
    w = np.linalg.solve(a, b)
    loss = compute_loss(y, tx, w)
    return w, loss



def sigmoid(t):
    return 1 / (1 + np.exp(-t))


''' Loss for the logistic function '''
def logistic_loss(y, y_hat):
    loss = (-1 / len(y)) * (y.T.dot(np.log(y_hat)) + (1 - y).T.dot(np.log(1 - y_hat)))
    return np.squeeze(loss)  # Remove axes of length 1


''' Logistic regression using gradient descent or SGD'''
def logistic_regression(y, tx, initial_w, max_iters, gamma):
    w = initial_w
    for _ in range(max_iters):
        gradient = 1 / len(y) * tx.T.dot(sigmoid(tx.dot(w)) - y)
        w = w - gamma * gradient
        print("w is", w)


    y_hat = sigmoid(tx.dot(w))
    loss = logistic_loss(y, y_hat)
    return w, loss


''' Regularized logistic regression using gradient descent or SGD'''
def reg_logistic_regression(y, tx, lambda_, initial_w, max_iters, gamma):
    w = initial_w
    for _ in range(max_iters):
        y_hat = sigmoid(tx.dot(w))
        gradient = 1 / len(y) * tx.T.dot(y_hat - y) + 2 * lambda_ * w
        w = w - gamma * gradient

    y_hat = sigmoid(tx.dot(w))
    loss = logistic_loss(y, y_hat)

    return w, loss
