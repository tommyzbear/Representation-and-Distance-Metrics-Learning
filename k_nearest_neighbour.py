import numpy as np


class KNN:
    def __init__(self, test_feature, train_features, train_labels, k=1):
        self.test_features = test_feature
        self.train_features = train_features
        self.train_labels = train_labels
        self.k = k

    def predict(self):
        euclidean_dist = np.linalg.norm(self.train_features.T - self.test_features.T, axis=1)
        nearest_k_indices = np.argsort(euclidean_dist)[:self.k]
        nearest_k_labels = self.train_labels[nearest_k_indices]
        nearest_labels, labels_count = np.unique(nearest_k_labels, return_counts=True)
        return nearest_labels[np.where(labels_count == labels_count.max())]

    def nearest_neighbours(self, n_nearest_neighbours=1):
        euclidean_dist = np.linalg.norm(self.train_features - self.test_features.T, axis=1)
        nearest_n_indices = np.argsort(euclidean_dist)[:n_nearest_neighbours]
        nearest_n_labels = self.train_labels[nearest_n_indices]
        return nearest_n_indices, nearest_n_labels
