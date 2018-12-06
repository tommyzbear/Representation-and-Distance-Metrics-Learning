from k_mean import KMean
import numpy as np

test_feature = np.asarray([12, 12])
train_features = np.asarray([[0, 0], [0, 1], [1, 0], [1, 1], [10, 10], [10, 11], [11, 10], [11, 11], [13, 10], [13, 14], [14, 13], [14, 14]])
train_labels = np.asarray([1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3])
K = 3

kmean = KMean(test_feature, train_features, train_labels, K)
print(kmean.classify(n_nearest_cluster=2))
