import numpy as np
import matplotlib.pyplot as plt
import sys

from helpers import *
from implementations import *
from encoding import *

data_path = 'data'
x_train_preclean, x_test_preclean, y_train, train_ids, test_ids = load_csv_data(data_path)

print("data is loaded")
"""# Data Cleaning"""

x_train, filter, categorical_filter, corr_filter, num_transform, cat_transform = process_train(x_train_preclean)
x_test = process_test(x_test_preclean, filter, categorical_filter, corr_filter, num_transform, cat_transform)

"""# Logistic regression *WITH* regularization"""

initial_w = np.zeros(x_train.shape[1], dtype=np.float64)
max_iters = 100
gamma = 0.01
lambda_ = 0.001
w_ada, loss_AdaGrad, losses, t = regularized_log_AdaGrad(y_train, x_train, initial_w, max_iters, gamma, lambda_)

y_pred_test_ada = prediction_labels(w_ada, x_test)
y_pred_test_ada[y_pred_test_ada == 0] = -1
create_csv_submission(test_ids, y_pred_test, "submission_for_repo.csv")