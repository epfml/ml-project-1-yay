from implementations import *
from helpers import *

import pickle

'''
with open('training_x.pickle', 'rb') as file:
   tx = pickle.load(file)

with open('training_y.pickle', 'rb') as file:
   y = pickle.load(file)
'''
# loading the data
data_path = '../data/dataset_to_release'
xtrain, xtest, ytrain, train_ids, test_ids = load_csv_data(data_path)
print("done loading data")
print('x_train shape before splitting: ', xtrain.shape)
print('y_train shape before splitting: ', ytrain.shape)


## splitting training data into train and test sets
data = xtrain.copy()
x_train, x_test, y_train, y_test = split_data(data, ytrain, 0.8)
print('x_train shape after split: ', x_train.shape)
print('y_train shape after split: ', y_train.shape)
print('x_test shape after split: ', x_test.shape)
print('y_test shape after split: ', y_test.shape)


## initial processing of the data like replace Nan values, standarize and add column of 1
def process_data(x):
   col_means = np.nanmean(x, axis=0)
   inds = np.where(np.isnan(x))
   x[inds] = np.take(col_means, inds[1])  # replace columns with values NaN with the mean of that column
   x = (x - np.mean(x)) / np.std(x)  # standarize the data
   x = np.c_[np.ones(len(x)), x]  # add the column of ones
   return x

print('the data fucking sucks ')
print('I agree ')
x_train = x_train[:,:-6] # drop the last 6 columns ?? why? 

x_train = process_data(x_train)
x_test = process_data(x_test)
print('x_train after processing ',x_train.shape)
print('x_test after processing ', x_test.shape)

def prediction_labels(weights, data): ## isn't this for linear regression only ? Don't we need the sigmoid?
   """Generates class predictions given weights, and a test data matrix."""
   y_pred = np.dot(data, weights)
   y_pred[np.where(y_pred <= 0)] = -1
   y_pred[np.where(y_pred > 0)] = 1
   return y_pred


### logistic regression

# our y_train has 1, -1 values
y_train01 = y_train.copy()
y_train01[y_train == -1] = 0
print(y_train01)

# our y_train has 1, -1 values
y_test01 = y_test.copy()
y_test01[y_test == -1] = 0
#print(y_test01)
#print(np.sum(y_test01))
def compute_gradient_logistic_loss(y, tx, w):
   """Compute the gradient of the logistic regression."""
   return tx.T.dot((1 / (1 + np.exp(- (tx.dot(w)))) - y))

losses_train = []
losses_test = []
accuracies_train = []
accuracies_test = []

def logistic_regression(y, tx, initial_w, max_iters, gamma):
   """Logistic regression using SGD and computing the accuracy of training and testing."""

   w = initial_w
   prev_loss = float('inf')

   for _ in range(max_iters):
      for batch_y, batch_x in batch_iter(y, tx, 1, num_batches=len(tx)):
         gradient = compute_gradient_logistic_loss(batch_y, batch_x, w)
         w = w - gamma * gradient

      loss = logistic_loss(y, tx, w)
      if prev_loss <= loss:
         gamma *= 0.1        # control of the step size
      prev_loss = loss
      '''
      y_pred_train = prediction_labels(w, tx) # generate prediction matrix
      y_pred_train[np.where(y_pred_train <= 0.5)] = -1   # make the prediction
      y_pred_train[np.where(y_pred_train > 0.5)] = 1

      accuracy_train = compute_accuracy(y_train, y_pred_train)
      losses_train.append(loss)
      accuracies_train.append(accuracy_train)

      test_loss = logistic_loss(y_test01, x_test, w)
      y_pred_test = prediction_labels(w, x_test)
      y_pred_test[np.where(y_pred_test <= 0.5)] = -1
      y_pred_test[np.where(y_pred_test > 0.5)] = 1

      accuracy_test = compute_accuracy(y_test, y_pred_test)
      losses_test.append(test_loss)
      accuracies_test.append(accuracy_test)

   print('losses_train ', losses_train)
   print('losses_test ', losses_test)
   '''
   return w, loss




def compute_accuracy(y, y_pred):
   """Computer the accuracy of """
   # Build an array of 0 (if y == y_pred) and 2 (if y != y_pred).
   results = y - y_pred
   N = len(y)
   # Count the number of time y==y_pred divided by the total number of examples
   accuracy = (N - np.count_nonzero(results)) / float(N)
   return accuracy


##  PARAMATERS
initial_w = np.zeros(x_train.shape[1])
max_iters = 100
gamma = 0.001


w, train_loss = logistic_regression(y_train01, x_train, initial_w, max_iters, gamma)
print('weights are : ', w)
'''
y_pred_train = prediction_labels(w, x_train)  # we
y_pred_train[np.where(y_pred_train <= 0.5)] = -1
y_pred_train[np.where(y_pred_train > 0.5)] = 1
accuracy_train = compute_accuracy(y_train, y_pred_train)

y_pred_test = prediction_labels(w, x_test)
y_pred_test[np.where(y_pred_test <= 0.5)] = -1
y_pred_test[np.where(y_pred_test > 0.5)] = 1
accuracy_test = compute_accuracy(y_test, y_pred_test)

test_loss = logistic_loss(y_test01, x_test, w)  #

print("train loss = ",train_loss)
print("train accuracy = ",accuracy_train)

print("test loss = ", test_loss)
print("test accuracy = ", accuracy_test)
'''
# /////////////////////////////////////// try submitting ///////////////////////////////////////////////////////////
# submission
# trying on the full set of data
xtest = process_data(xtest)
xtest = xtest[:,:-6]  # drop the last 6 columns
print('xtest shape ', xtest.shape)
y_pred = prediction_labels(w, xtest)
print('prediction output ',y_pred)


import pandas as pd
csv_file_path = '../data/predictions.csv'
predictions_df = pd.DataFrame({'Prediction': y_pred})
# Save the predictions to a CSV file
predictions_df.to_csv(csv_file_path, index=False)



# //////////////////////////////////////// Cross Validation ////////////////////////////////////////////////////////
# TODO hypertune





