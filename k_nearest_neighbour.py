import numpy as np
from numba import jit
from metric import distance_metrics


class KNN:
    def __init__(self, test_feature, train_features, train_labels, dist_matrix='Euclidean', k=1):
        self.test_feature = test_feature
        self.train_features = train_features
        self.train_labels = train_labels
        self.dist_matrix = dist_matrix
        self.k = k

    def predict(self):
        euclidean_dist = np.linalg.norm(self.train_features - self.test_feature.T, axis=1)
        nearest_k_indices = np.argsort(euclidean_dist)[:self.k]
        nearest_k_labels = self.train_labels[nearest_k_indices]
        nearest_labels, labels_count = np.unique(nearest_k_labels, return_counts=True)
        most_common_labels = nearest_labels[np.where(labels_count == labels_count.max())]

        if len(most_common_labels) == 1:
            return most_common_labels
        else:
            dist = np.zeros(len(most_common_labels))
            i = 0
            for label in most_common_labels:
                dist = min(euclidean_dist[nearest_k_indices[np.where(nearest_k_labels == label)]])
                i += 1
            return most_common_labels[np.argmin(dist)]

    @jit
    def nearest_neighbours(self, n_nearest_neighbours=1, covariance=None):
        method_func = distance_metrics(self.dist_matrix)
        if covariance is None and self.dist_matrix is 'Mahalanobis':
            raise Exception("Covariance matrix not provided")
        elif covariance is None:
            dist_matrix = method_func(self.train_features, self.test_feature)
        else:
            dist_matrix = method_func(self.train_features, self.test_feature, covariance)
        nearest_n_indices = np.argsort(dist_matrix)[:n_nearest_neighbours]
        nearest_n_labels = self.train_labels[nearest_n_indices]
        return nearest_n_indices, nearest_n_labels
