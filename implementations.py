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
    return 0.5 * np.mean(e**2)


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
    """MSE Gradient descent
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
    pred = sigmoid(tx.dot(w))
    loss = y.T.dot(np.log(pred)) + (1 - y).T.dot(np.log(1 - pred))
    -loss * (1 / y.shape[0])


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
        gradient = tx.T.dot(sigmoid(tx.dot(w)) - y) / len(y)
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


def logistic_gradient(y, tx, w):
    return tx.T.dot(sigmoid(tx.dot(w)) - y) / len(y)


def compute_gradient_logistic_loss_regularized(y, tx, w, lambda_):
    """
    Compute the gradient of regularized logistic loss.

    Args:
        y (numpy.ndarray): Data labels.
        tx (numpy.ndarray): Data features.
        w (numpy.ndarray): Weights.
        lambda_ (float): Regularization parameter.

    Returns:
        numpy.ndarray: Gradient of the regularized logistic loss.
    """
    grad = logistic_gradient(y, tx, w) + lambda_ * w
    return grad


def regularized_log_reg_sgd(y, tx, initial_w, max_iters, gamma, lambda_):
    """
    Regularized logistic regression using stochastic gradient descent.

    Args:
        y (numpy.ndarray): Data labels.
        tx (numpy.ndarray): Data features.
        ... (Other parameters as in the original function)

    Returns:
        tuple: Final weights, logistic loss, list of losses for each iteration, and total iterations run.
    """
    w = initial_w
    prev_loss = float("inf")
    losses = []
    t = 0
    for n_iter in range(max_iters):
        # Each iteration corresponds to one epoch (num_batches=len(y)) and each batch has size 1
        for batch_y, batch_x in batch_iter(y, tx, 1, num_batches=300):
            # Computing the gradient of the logistic loss with respect to w
            gradient = compute_gradient_logistic_loss_regularized(
                batch_y, batch_x, w, lambda_
            )
            # Updating w
            w -= gamma * gradient

        loss = logistic_loss(y, tx, w) + (lambda_ / 2) * np.squeeze(w.T @ w)
        if prev_loss <= loss:
            gamma *= 0.1  # adapt step size
        prev_loss = loss
        t += 1
        losses.append(loss)
    return w, loss, losses, t


def regularized_log_AdaGrad(y, tx, initial_w, max_iters, gamma, lambda_, epsilon=1e-5):
    """
    Regularized logistic regression using AdaGrad optimization.

    Args:
        y (numpy.ndarray): Data labels.
        tx (numpy.ndarray): Data features.
        ... (Other parameters as in the original function)

    Returns:
        tuple: Final weights, logistic loss, list of losses for each iteration, and total iterations run.
    """

    # Initialize weights and squared gradient accumulator
    w = initial_w
    squared_grad_accumulator = np.zeros(w.shape)
    losses = []
    # Initialize previous loss for convergence check
    prev_loss = float("inf")
    # t keeps count of the actual iterations run
    t = 0
    for n_iter in range(max_iters):
        for batch_y, batch_tx in batch_iter(y, tx, batch_size=1, num_batches=328):
            # Compute the gradient of the logistic loss with regularization
            gradient = compute_gradient_logistic_loss_regularized(
                batch_y, batch_tx, w, lambda_
            )

            # Accumulate squared gradient
            squared_grad_accumulator += gradient**2

            # Update weights using Adagrad update rule
            w -= gradient * (1.0 / (np.sqrt(squared_grad_accumulator) + epsilon))
            t += 1
        # Calculate the logistic loss with regularization
        loss = logistic_loss(y, tx, w) + (lambda_ / 2) * np.squeeze(w.T @ w)

        # STOPPING CRITERIA
        relative_epsilon = 1e-4  # Define a threshold for relative change
        if abs(prev_loss - loss) / prev_loss < relative_epsilon:
            break
        prev_loss = loss

        losses.append(prev_loss)

    return w, loss, losses, t


def build_k_indices(y, k_fold, seed):
    """build k indices for k-fold.

    Args:
        y:      shape=(N,)
        k_fold: K in K-fold, i.e. the fold num
        seed:   the random seed

    Returns:
        A 2D array of shape=(k_fold, N/k_fold) that indicates the data indices for each fold
    """
    num_row = y.shape[0]
    interval = int(num_row / k_fold)

    np.random.seed(seed)
    indices = np.random.permutation(num_row)
    k_indices = [indices[k * interval : (k + 1) * interval] for k in range(k_fold)]
    return np.array(k_indices)


def cross_validation(y, x, max_iters, k_fold, gamma, lambda_, seed):
    """
    Perform k-fold cross-validation on the dataset.

    Args:
        y (numpy.ndarray): Data labels.
        x (numpy.ndarray): Data features.
        ... (Other parameters as in the original function)

    Returns:
        tuple: Average weight, average training loss, average test loss, and average f1_score.
    """

    k_indices = build_k_indices(y, k_fold, seed)
    train_losses = []
    test_losses = []
    weights = []
    f_scores = []
    for k in range(k_fold):
        test_indices = k_indices[k]
        train_indices = np.delete(k_indices, k, axis=0).flatten()
        kx_train = x[train_indices]
        ky_train = y[train_indices]
        kx_test = x[test_indices]
        ky_test = y[test_indices]

        w, loss = regularized_log_reg_sgd(
            ky_train,
            kx_train,
            initial_w=np.zeros(kx_train.shape[1]),
            max_iters=max_iters,
            gamma=gamma,
            lambda_=lambda_,
        )
        train_losses.append(loss)
        test_loss = logistic_loss(ky_test, kx_test, w) + (lambda_ / 2) * np.squeeze(
            w.T @ w
        )
        test_losses.append(test_loss)
        weights.append(w)

        ## newly added, getting the fscores predicted here
        y_pred = prediction_labels(w, kx_test)
        f = f1_score(y_pred, ky_test)
        f_scores.append(f)

    return (
        np.mean(weights, axis=0),
        np.mean(train_losses),
        np.mean(test_losses),
        np.mean(f_scores),
    )


def prediction_labels(weights, data):
    """Generates class predictions given weights, and a test data matrix."""
    y_pred = sigmoid(np.dot(data, weights))
    y_pred[np.where(y_pred >= 0.5)] = 1
    y_pred[np.where(y_pred < 0.5)] = 0
    return y_pred


def accuracy(y_pred, y_train):
    """Calculate accuracy of predictions."""
    return (y_pred == y_train).sum() / len(y_train)


def precision(y_pred, y_train):
    """Calculate accuracy of predictions."""
    TP = np.sum((y_train == 1) & (y_pred == 1))
    FP = np.sum((y_train == 0) & (y_pred == 1))
    return TP / (TP + FP)


def recall(y_pred, y_train):
    """Calculate recall of predictions."""
    recall = np.sum((y_train == 1) & (y_pred == 1)) / np.sum(y_train == 1)
    return recall


def f1_score(y_pred, y_train):
    """Calculate F1 score of predictions."""
    return (
        2
        * precision(y_pred, y_train)
        * recall(y_pred, y_train)
        / (precision(y_pred, y_train) + recall(y_pred, y_train))
    )


def get_best_parameters(y, tx, intitial_w, max_iters, k_fold, gamma, lambdas):
    """
    Determine the best parameters using cross-validation.

    Args:
        y (numpy.ndarray): Data labels.
        tx (numpy.ndarray): Data features.
        ... (Other parameters as in the original function)

    Returns:
        tuple: Training loss, test loss, F1 scores, weights, and predicted labels for the best parameters.
    """
    seed = 55

    weights = []
    loss_tr = []
    loss_tt = []
    f1_scores = []
    y_preds = []

    for lambda_ in lambdas:
        # make cross validation return weight
        avg_w, avg_train_loss, avg_test_loss, avg_f_score = cross_validation(
            y, tx, max_iters, k_fold, gamma, lambda_, seed
        )
        loss_tr.append(avg_train_loss)
        loss_tt.append(avg_test_loss)
        weights.append(avg_w)
        f1_scores.append(avg_f_score)
        y_pred = prediction_labels(avg_w, tx)
        y_preds.append(y_pred)

    return loss_tr, loss_tt, f1_scores, weights, y_preds
