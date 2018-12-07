import numpy as np
from scipy.spatial.distance import correlation
from numba import jit


def distance_metrics(method):
    method_dict = {
        'Euclidean': euclidean,
        'Manhattan': manhattan,
        'Chessboard': chessboard,
        'Cosine': cosine,
        'Correlation': cross_correlation,
        'Intersection': intersection,
        'KL_Divergence': kullback_leibler_divergence,
        'JS_Divergence': jensen_shannon_divergence,
        'Quadratic_form_distance': quadratic_form_histogram_dist,
        'Mahalanobis': mahalanobis_dist
    }

    method_func = method_dict.get(method, lambda: "Distance matrix not found")
    return method_func


@jit
def euclidean(train_features, test_feature):
    return np.linalg.norm(train_features - test_feature, axis=1)


@jit
def manhattan(train_features, test_feature):
    return np.sum(abs(train_features - test_feature), axis=1)


@jit
def chessboard(train_features, test_feature):
    return np.max(abs(train_features - test_feature), axis=1)


@jit
def cosine(train_features, test_feature):
    return train_features @ test_feature / (np.linalg.norm(train_features, axis=1) * np.linalg.norm(test_feature))


@jit
def cross_correlation(train_features, test_feature):
    feature_correlations = []
    for train_feature in train_features:
        feature_correlations.append(correlation(train_feature, test_feature))
    return np.asarray(feature_correlations)


@jit
def intersection(train_features, test_feature):
    min_sum = 0
    for i in range(len(train_features)):
        min_sum += min(train_features[i], test_feature)
    # intersection = ((min_sum / sum(train_features)) + (min_sum / sum(test_feature))) / 2
    return min_sum


@jit
def kullback_leibler_divergence(train_features, test_feature):
    return np.sum(train_features*np.log(train_features / test_feature), axis=1)


@jit
def jensen_shannon_divergence(train_features, test_feature):
    half_KL_P = (1/2) * np.sum(train_features*np.log((2 * train_features) / (train_features + test_feature)), axis=1)
    half_KL_Q = (1/2) * np.sum(np.log((2 * test_feature) / (train_features + test_feature)) * train_features, axis=1)
    return half_KL_P + half_KL_Q


@jit
def quadratic_form_histogram_dist(train_features, test_feature):
    QF_arr = []
    for train_feature in train_features:
        feature_diff = train_feature - test_feature
        QF_arr.append(np.sqrt(feature_diff @ np.cov(feature_diff) @ feature_diff))
    return np.asarray(QF_arr)


@jit
def mahalanobis_dist(train_features, test_feature):
    QF_arr = []
    for train_feature in train_features:
        feature_diff = train_feature - test_feature
        QF_arr.append(np.sqrt(feature_diff @ np.linalg.inv(np.cov(feature_diff)) @ feature_diff))
    return np.asarray(QF_arr)


@jit
def chi_square(train_features, test_feature):
    chi_sq_sum = np.sum((train_features - test_feature) ** 2 / (train_features + test_feature), axis=1) / 2
    return chi_sq_sum