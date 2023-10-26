import numpy as np
from helpers import *
from implementations import *


data_path_debug = "../data/dataset/debug"
x_train_preclean, x_test_preclean, y_train, train_ids, test_ids = load_csv_data(data_path_debug)
x_train_preclean


def percentageFilled(data):
    return 1 - np.isnan(data).sum() / len(data)

def threshold_col_filter(data, threshold):
    percentage_filled = np.apply_along_axis(percentageFilled, 0, data)
    # keep_indicies = np.argwhere(percentage_filled > threshold).flatten()
    return percentage_filled > threshold

def non_constant_filter(data):
    return np.logical_not(np.logical_or(np.isnan(np.nanstd(data, 0)), np.nanstd(data, 0) == 0))


keep_indicies = np.argwhere(np.logical_and(
    threshold_col_filter(x_train_preclean, 0.2),
    non_constant_filter(x_train_preclean)
    )
).flatten()
print("keep_indicies", keep_indicies)

def filter_columns_by_indicies(data, keep_indicies):
    """
    used to process test data
    only keep the columns that are in the indicies
    """
    return data[:, keep_indicies]


x_train = filter_columns_by_indicies(x_train_preclean, keep_indicies)
print("X train", x_train.shape)

x_test = filter_columns_by_indicies(x_test_preclean, keep_indicies)
print("X test", x_test.shape)


def standardize(x):
    """Standardize the original data set."""
    return np.nan_to_num((x - np.nanmean(x, axis=0)) / np.nanstd(x, axis=0))


def process_data(x):
    # col_means = np.nanmean(x, axis=0)
    # inds = np.where(np.isnan(x))
    # x[inds] = np.take(col_means, inds[1])  # replace columns with values NaN with the mean of that column
    # x = (x - np.mean(x)) / np.std(x)  # standarize the data
    ## ?? I feel like standardizing by column shouldn't be done like above
    x = standardize(x)
    x = np.c_[np.ones(len(x)), x]  # add the column of ones
    return x


x_train_std = process_data(x_train)


initial_w = np.zeros(x_train_std.shape[1], dtype=np.float128)
max_iters = 100
gamma = 0.5

w, loss = logistic_regression(y_train, x_train_std, initial_w, max_iters, gamma)

print("loss is ", loss)
print("w is ", w)