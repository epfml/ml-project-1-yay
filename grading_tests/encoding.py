import numpy as np
from helpers import *


def percentageFilled(data):
    return 1 - np.isnan(data).sum() / len(data)


def threshold_col_filter(data, threshold):
    percentage_filled = np.apply_along_axis(percentageFilled, 0, data)
    return percentage_filled > threshold


def non_constant_filter(data):
    return np.logical_not(
        np.logical_or(np.isnan(np.nanstd(data, 0)), np.nanstd(data, 0) == 0)
    )


def standardize(x):
    """Standardize the original data set."""
    std = np.nanstd(x, axis=0)
    mean = np.nanmean(x, axis=0)
    return np.nan_to_num((x - np.nanmean(x, axis=0)) / np.nanstd(x, axis=0)), mean, std


def transform_train(feature):
    m = dict()
    for x in feature:
        if x not in m and not np.isnan(x):
            m[x] = len(m)
    f = np.vstack((np.eye(len(m)), np.zeros(len(m))))
    u = f[np.vectorize(lambda key: m.get(key, len(m)))(feature)]
    return u, m


def transform_test(feature, m):
    n_uniq = len(m)
    f = np.vstack((np.eye(n_uniq), np.zeros(n_uniq)))
    ind = np.array([m[k] if k in m else n_uniq for k in feature])
    return f[ind]


def process_train(data, corr_threshold=0.5, cat_threshold=10):
    n, m = data.shape
    # define feature selection filters
    filter = np.logical_and(threshold_col_filter(data, 0.2), non_constant_filter(data))
    # heuristic-based feature partition into categorical and numerical
    categorical_filter = np.apply_along_axis(
        lambda x: len(set(x)) < cat_threshold, 0, data
    )
    # storing feature transformation for consistent application on test set
    cat_transform = dict()
    num_transform = dict()
    features = np.empty((n, 0))
    for i in range(m):
        # apply feature selection filter
        if not filter[i]:
            continue
        # if feature is categorical, one-hot encode it and store value mapping to indices
        if categorical_filter[i]:
            encoded, mp = transform_train(data[:, i])
            cat_transform[i] = mp
            features = np.append(features, encoded, axis=1)
        # if feature is numerical, standartize it and store the transformation for consistent application on test set
        else:
            x_num_std, mean, std = standardize(data[:, i])
            x_num_std[abs(x_num_std) > 3] = 0
            num_transform[i] = (mean, std)
            features = np.append(features, x_num_std.reshape((n, 1)), axis=1)
    # build correlation matrix
    corr_filter = np.full((features.shape[1],), True, dtype=bool)
    cm = np.corrcoef(features, rowvar=False)
    # filter out features with the correlation to others above threshold
    for i in range(len(cm)):
        if corr_filter[i]:
            for j in range(i + 1, len(cm)):
                if corr_filter[j] and abs(cm[i][j]) >= corr_threshold:
                    corr_filter[j] = False

    return (
        np.c_[np.ones(n), features[:, corr_filter]],
        filter,
        categorical_filter,
        corr_filter,
        num_transform,
        cat_transform,
    )


def process_test(
    data, filter, categorical_filter, corr_filter, num_transform, cat_transform
):
    n, m = data.shape
    res = np.empty((n, 0))
    for i in range(m):
        # filter out features according to stored selection filter
        if not filter[i]:
            continue
        # distinguish between categorical and numerical features consistently using
        # the same filter for the train and test sets
        if categorical_filter[i]:
            res = np.append(res, transform_test(data[:, i], cat_transform[i]), axis=1)
        else:
            # transform numerical features consistently with the train set
            mean, std = num_transform[i]  # std shouldn't be 0
            res = np.append(
                res, np.nan_to_num((data[:, i] - mean) / std).reshape((n, 1)), axis=1
            )
    return np.c_[np.ones(n), res[:, corr_filter]]
