import numpy as np
from helpers import *


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


def percentageFilled(data):
    return 1 - np.isnan(data).sum() / len(data)


def threshold_col_filter(data, threshold):
    percentage_filled = np.apply_along_axis(percentageFilled, 0, data)
    return percentage_filled > threshold


def non_constant_filter(data):
    return np.logical_not(np.logical_or(np.isnan(np.nanstd(data, 0)), np.nanstd(data, 0) == 0))

def standardize(x):
    """Standardize the original data set."""
    std = np.nanstd(x, axis=0)
    mean = np.nanmean(x, axis=0)
    return np.nan_to_num((x - np.nanmean(x, axis=0)) / np.nanstd(x, axis=0)), mean, std


def process_train(data, corr_threshold = 0.5, cat_threshold = 10, verbose = False):
    n, m = data.shape
    filter = np.logical_and(threshold_col_filter(data, 0.2), non_constant_filter(data))
    categorical_filter = np.apply_along_axis(lambda x: len(set(x)) < cat_threshold, 0, data)
    cat_transform = dict()
    num_transform = dict()
    features = np.empty((n, 0))
    transform_log = list()
    cnt = 0
    for i in range(m):
        if not filter[i]:
            transform_log.append(f"{i} deleted")
            cnt += 1
            continue
        if categorical_filter[i]:
            encoded, mp = transform_train(data[:, i])
            l = encoded.shape[1]
            transform_log.append(f"{i} cat exp: {cnt}-{cnt + l - 1}")
            cnt += l
            cat_transform[i] = mp
            features = np.append(features, encoded, axis=1)
        else:
            transform_log.append(f"{i} num: {cnt}")
            cnt += 1
            x_num_std, mean, std = standardize(data[:, i])
            x_num_std[abs(x_num_std) > 3] = 0
            num_transform[i] = (mean, std)
            features = np.append(features, x_num_std.reshape((n,1)), axis=1)
    corr_filter = np.full((features.shape[1],), True, dtype=bool)
    cm = np.corrcoef(features, rowvar=False)
    for i in range(len(cm)):
        if corr_filter[i]:
            for j in range(i + 1, len(cm)):
                if corr_filter[j] and abs(cm[i][j]) >= corr_threshold:
                    corr_filter[j] = False
    if verbose:
        print(f"number of features with pairwise corr <= {corr_threshold}: {sum(corr_filter)}")

    if verbose:
        print("\n".join(transform_log))
    return features[:, corr_filter], filter, categorical_filter, corr_filter, num_transform, cat_transform


def process_test(data, filter, categorical_filter, corr_filter, num_transform, cat_transform, verbose=False):
    n, m = data.shape
    res = np.empty((n, 0))
    transform_log = list()
    cnt = 0
    for i in range(m):
        if not filter[i]:
            transform_log.append(f"{i} deleted")
            cnt += 1
            continue
        if categorical_filter[i]:
            l = len(cat_transform[i])
            transform_log.append(f"{i} cat exp: {cnt}-{cnt + l - 1}")
            cnt += l
            res = np.append(res, transform_test(data[:, i], cat_transform[i]), axis=1)
        else:
            transform_log.append(f"{i} num: {cnt}")
            cnt += 1
            mean, std = num_transform[i] # std shouldn't be 0
            res = np.append(res, np.nan_to_num((data[:, i] - mean) / std).reshape((n,1)), axis=1)
    if verbose:
        print("\n".join(transform_log))
    return res[:, corr_filter]


if __name__ == '__main__':
    data_path = '../data/dataset/debug'
    x_train_preclean, x_test_preclean, y_train, train_ids, test_ids = load_csv_data(data_path)

    x_train, filter, categorical_filter, corr_filter, num_transform, cat_transform = process_train(x_train_preclean, verbose=True)
    x_test = process_test(x_test_preclean, filter, categorical_filter, corr_filter, num_transform, cat_transform, verbose=True)

    np.set_printoptions(precision=2)

    print(x_train.shape)
    print(x_train)

    print("-------------")
    print("-------------")
    print("-------------")
    print("-------------")
    print("-------------")

    print(x_test.shape)
    print(x_test)
