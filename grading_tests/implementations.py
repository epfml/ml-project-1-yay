from helpers import *
import numpy as np



def compute_MSE_loss(y, tx, w):
    """

       Parameters:
        y:  shape=(N, 1)
        tx: shape=(N, D)
        w:  shape=(D, 1)

       Returns:
       float: The Mean Squared Error (MSE) loss between true labels and predictions.
       """
    e = y - tx.dot(w)
    return 0.5 * np.mean(e ** 2)


def compute_gradient(y, tx, w):
    """
        Parameters:
         y: shape = (N, 1)
         tx: shape = (N, D)
         w: shape = (D, 1)

        Returns:
        The gradient of the MSE loss
    """
    e = y - tx.dot(w)
    grad = -tx.T.dot(e) / len(e)
    return grad




def mean_squared_error_gd(y, tx, initial_w, max_iters, gamma):
    """ MSE Gradient descent
        Args:
            y: numpy array of shape (N,), N is the number of samples.
            tx: numpy array of shape (N,D), D is the number of features
            initial_w: initial weight vector
            max_iter: number of iterations
            gamma: step size of the gradient descent
        Returns:
            w : weights
            loss: last loss value from the last iteration
    """
    w = initial_w
    for _ in range(max_iters):
        grad = compute_gradient(y, tx, w)
        w = w - gamma * grad

    loss = compute_MSE_loss(y, tx, w)
    return w, loss




def mean_squared_error_sgd(y, tx, initial_w, max_iters, gamma):
    """
        Perform Stochastic Gradient Descent (SGD) to optimize the loss between true labels and predictions.

        Parameters:
        y: numpy array of shape (N,), N is the number of samples.
        tx: numpy array of shape (N,D), D is the number of features
        initial_w (D, 1): Initial model parameters (weights), a 1D array or list.
        max_iters (int): The maximum number of iterations for updating model parameters.
        gamma (float): The learning rate, controlling the step size during parameter updates.

        Returns:
        tuple: A tuple containing the final model parameters (weights) and the MSE loss.
          - w (array-like): The learned model parameters after SGD optimization.
          - loss (float): The Mean Squared Error (MSE) loss between true labels and predictions.
        """
    w = initial_w
    for _ in range(max_iters):
        for y_batch, tx_batch in batch_iter(y, tx, batch_size=1, num_batches=1):
            grad = compute_gradient(y_batch, tx_batch, w)
            w = w - gamma * grad

    loss = compute_MSE_loss(y, tx, w)
    return w, loss



def least_squares(y, tx):
    """
        Least squares using normal equations
        Args:
            y: numpy array of shape (N,), N is the number of samples.
            tx: numpy array of shape (N,D), D is the number of features
        Returns:
            w:  numpy array of shape (D,1),  weights
            loss: loss value
    """
    w = np.linalg.solve(tx.T.dot(tx), tx.T.dot(y))
    loss = compute_MSE_loss(y, tx, w)
    return w, loss





def ridge_regression(y, tx, lambda_):
    """
        Perform Ridge regression using normal equations.
        Parameters:
            y: numpy array of shape (N,), N is the number of samples.
            tx: numpy array of shape (N,D), D is the number of featuresy_hat = sigmoid(tx.dot(w))
            lambda_: float, a regularization parameter

        Returns:
            w : weights
            loss: last loss value from the last iteration

    """
    lambda_prime = lambda_ * 2 * tx.shape[0]
    a = tx.T.dot(tx) + lambda_prime * np.identity(tx.shape[1])
    b = tx.T.dot(y)
    w = np.linalg.solve(a, b)
    loss = compute_MSE_loss(y, tx, w)
    return w, loss


def sigmoid(t):
    """
      Calculate the sigmoid function for a given input.
      Parameters:
      t (float): The input value

      Returns:
      float: The result of the sigmoid function
      """
    # return np.where(t < 0, np.exp(t)/(1.0 +np.exp(t)) , 1.0 / (1.0 + np.exp(-t)))  ##
    return 1.0 / (1 + np.exp(-t))




def logistic_loss(y, tx, w):
    """
        Calculate the logistic loss between true labels and predicted probabilities.
        Parameters:
        y: numpy array containing N (-1 or 1) values for true labels
        tx: numpy array of shape (N,D), D is the number of features
        w : numpy array of shape weights
        Returns:
        float: The logistic loss between true labels and predicted probabilities.
        """
    # epsilon = 0.000000001
    # y_hat = sigmoid(tx.dot(w))
    # y_hat = np.clip(y_hat, epsilon, 1-epsilon)

    # loss = - np.average(y*np.log(y_hat) + (1-y)*np.log(1-y_hat))
    # loss = (-1 / len(y)) * (y.T.dot(np.log(y_hat)) + (1 - y).T.dot(np.log(1 - y_hat)))
    ## return np.squeeze(loss)  # Remove axes of length 1


    pred = sigmoid(tx.dot(w))
    loss = y.T.dot(np.log(pred)) + (1 - y).T.dot(np.log(1 - pred))
    # loss = y.T.dot(np)
    return np.squeeze(-loss).item() * (1 / y.shape[0])

    return loss

def logistic_regression(y, tx, initial_w, max_iters, gamma, verbose=False):
    """
      Perform logistic regression using gradient descent or SGD

      Parameters: d
      y (array-like): True labels
      tx : Feature matrix
      initial_w : Initial weights
      max_iters (int): The maximum number of iterations for updating model parameters.
      gamma (float): The learning rate

      Returns:
      tuple: A tuple containing the final model parameters (weights) and the logistic loss.
        - w : The learned model parameters after logistic regression.
        - loss : The logistic loss (cross-entropy) between true labels and predicted probabilities.
      """
    w = initial_w
    for i in range(max_iters):
        gradient =  tx.T.dot(sigmoid(tx.dot(w)) - y)/ len(y)
        w = w - gamma * gradient
        if verbose and i % (max_iters // 20) == 0:
            print(f"loss at step #{i}: {logistic_loss(y, tx, w)}")

    loss = logistic_loss(y, tx, w)
    return w, loss


def reg_logistic_regression(y, tx, lambda_, initial_w, max_iters, gamma):
    """
        Perform regularized logistic regression using gradient descent or SGD

        Parameters:
        y (N, 1): True labels
        tx (N, D): Feature matrix
        lambda_ (float): Regularization parameter
        initial_w : Initial model weights
        max_iters (int): The maximum number of iterations for updating model parameters.
        gamma (float): The learning rate

        Returns:
          - w : The learned model parameters after regularized logistic regression.
          - loss (float): The logistic loss between true labels and predicted probabilities, including the regularization term.
        """

    w = initial_w
    for _ in range(max_iters):
        y_hat = sigmoid(tx.dot(w))
        gradient = 1 / len(y) * tx.T.dot(y_hat - y) + 2 * lambda_ * w
        w = w - gamma * gradient

    loss = logistic_loss(y, tx, w)
    return w, loss
