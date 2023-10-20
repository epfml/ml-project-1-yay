from helpers import *
import numpy as np

data_path = '../data/dataset/dataset_to_release'
x_train, x_test, y_train, train_ids, test_ids = load_csv_data(data_path)

print(len(x_train))



def predict_labels(weights, data):
    """
      Generates class predictions given weights, and a test data matrix.
      Parameters:

    """
    y_pred = np.dot(data, weights)
    y_pred[np.where(y_pred <= 0)] = -1
    y_pred[np.where(y_pred > 0)] = 1

    return y_pred