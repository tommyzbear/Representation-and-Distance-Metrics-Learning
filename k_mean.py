import numpy as np
from k_nearest_neighbour import KNN


class KMean:
    def __init__(self, test_feature, train_features, train_labels, K):
        self.test_feature = test_feature
        self.train_features = train_features
        self.train_labels = train_labels
        self.K = K

    def classify(self, n_nearest_cluster=1):
        centroids = np.random.random((self.K, self.test_feature.shape[0]))
        centroid_labels = np.zeros(self.K)
        fixed = False
        num_of_train_features = self.train_features.shape[0]
        euclidean_dist_arr = np.zeros((num_of_train_features, self.K))
        while not fixed:
            cluster_dict = dict()
            previous_centroids = centroids.copy()
            for i in range(num_of_train_features):
                euclidean_dist_arr[i] = np.linalg.norm(centroids - self.train_features[i], axis=1)
            feature_idx = 0
            for dist in euclidean_dist_arr:
                centroid = np.argmin(dist)
                if centroid not in cluster_dict.keys():
                    cluster_dict[centroid] = [feature_idx]
                else:
                    cluster_dict[centroid].append(feature_idx)
                feature_idx += 1
            for key in cluster_dict:
                nodes = cluster_dict[key]
                mean_node = sum(self.train_features[nodes]) / len(nodes)
                centroids[key] = mean_node

            if np.array_equal(centroids, previous_centroids):
                fixed = True

        for key in cluster_dict:
            feature_labels = self.train_labels[cluster_dict[key]]
            labels, counts = np.unique(feature_labels, return_counts=True)
            most_occurence_label = labels[np.where(counts == counts.max())]

            if len(most_occurence_label) == 1:
                centroid_labels[key] = most_occurence_label[0]
            else:
                dist = np.zeros(len(most_occurence_label))
                i = 0
                for label in most_occurence_label:
                    dist = min(euclidean_dist_arr.T[key][np.where(self.train_labels == label)])
                    i += 1
                centroid_labels[key] = most_occurence_label[np.argmin(dist)]

        knn = KNN(self.test_feature, centroids, centroid_labels)
        nearest_cluster_indices, labels = knn.nearest_neighbours(n_nearest_neighbours=n_nearest_cluster)

        return labels
# test_feature_dist = np.linalg.norm(centroids - self.train_features, axis=1)
#  cluster_index = np.argmin(test_feature_dist)


