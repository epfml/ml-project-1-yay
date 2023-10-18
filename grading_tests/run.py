from helpers import *
import numpy as np

data_path = '../data/dataset/dataset_to_release'
x_train, x_test, y_train, train_ids, test_ids = load_csv_data(data_path)

print(len(x_train))