from helpers import *
import numpy as np
from implementations import *

import pickle # remove later

data_path = '../data/dataset_to_release'
print("inside run.py")
x_train, x_test, y_train, train_ids, test_ids = load_csv_data(data_path)
print("done loading data")


def preprocess(x):
    x = np.c_[np.ones((x.shape[0], 1)), x]  # add the column of ones
    col_means = np.nanmean(x, axis=0)
    inds = np.where(np.isnan(x))
    x[inds] = np.take(col_means, inds[1])  # replace columns with values NaN with the mean of that column
    x = (x-np.mean(x))/np.std(x)  # standarize the data
    return x

print('x_train shape', x_train.shape)
#tx = preprocess(x_train)
#print('tx shape', tx.shape)  done later after the split

#print(tx)
# are the values repeating?
unique_values = np.unique(tx)
print('how many unique values ',len(unique_values))

'''
# to pickle the tx training matrix
with open('training_x.pickle', 'wb') as file:
    pickle.dump(x_tru, file)
# to pickle y_train
with open('training_y.pickle', 'wb') as file:
    pickle.dump(y_train, file)

'''