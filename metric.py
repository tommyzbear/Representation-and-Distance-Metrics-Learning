import numpy as np
from scipy.spatial.distance import correlation as correlation


def euclidean(x1, x2):
    sum = 0
    for i in range(len(x1)):
        sum += (x1[i] - x2[i])**2
    return sum


def cosine(x1, x2):
    cos_similarity = x1.T * x2 / (np.linalg.norm(x1, 2) * np.linalg.norm(x2, 2))
    return cos_similarity


def cross_correlation(x1, x2):
    cross_corr = correlation(x1, x2)
    return cross_corr


def intersection(x1, x2):
    min_sum = 0
    for i in range(len(x1)):
        min_sum += min(x1[i], x2[i])
    unnormalized_intersection = ((min_sum/sum(x1))+(min_sum/sum(x2)))/2
    return unnormalized_intersection


def chi_square(x1, x2):
    chi_sq_sum = (np.sum((x1 - x2) ** 2 / (x1 + x2)))/2
    return chi_sq_sum