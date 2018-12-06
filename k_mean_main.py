from scipy.io import loadmat
import json
from sklearn.cluster import KMeans
import numpy as np
import time
import os


# Sorting data for later use
dir = os.path.dirname(os.path.realpath(__file__)) + "\\PR_data\\"
cuhk03_data = loadmat(dir + 'cuhk03_new_protocol_config_labeled.mat')

query_idxs = cuhk03_data['query_idx'].flatten()
gallery_idxs = cuhk03_data['gallery_idx'].flatten()
labels = cuhk03_data['labels'].flatten()

with open(dir + 'feature_data.json', 'r') as f:
    features = json.load(f)

features = np.asarray(features)

# Get query features
query_features = []
for idx in query_idxs:
    query_features.append(features[idx - 1])

# Get correct labels for testing set
query_labels = []
for idx in query_idxs:
    query_labels.append(labels[idx - 1])

gallery_vectors = features[gallery_idxs - 1]
gallery_labels = labels[gallery_idxs - 1]

num_of_clusters_arr = [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]

n = 10

for num_of_clusters in num_of_clusters_arr:
    start_time = time.time()

    rank_one_score = 0
    rank_five_score = 0
    rank_ten_score = 0

    k_mean = KMeans(n_clusters=num_of_clusters).fit(gallery_vectors)
    feature_clusters = k_mean.labels_
    cluster_dict = dict()
    feature_idx = 0

    for cluster in feature_clusters:
        if cluster not in cluster_dict.keys():
            cluster_dict[cluster] = [gallery_labels[feature_idx]]
        else:
            cluster_dict[cluster].append(gallery_labels[feature_idx])
        feature_idx += 1

    cluster_labels = np.zeros(num_of_clusters)

    for j in range(num_of_clusters):
        label, counts = np.unique(cluster_dict[j], return_counts=True)
        cluster_labels[j] = label[0]

    centroids = k_mean.cluster_centers_

    for i in range(len(query_features)):
        dist_to_centroids = np.linalg.norm(centroids - query_features[i], axis=1)

        result_label = cluster_labels[np.argsort(dist_to_centroids)[:n]]

        if query_labels[i] in result_label:
            rank_ten_score += 1
            if query_labels[i] in result_label[:5]:
                rank_five_score += 1
                if query_labels[i] == result_label[0]:
                    rank_one_score += 1

    end_time = time.time()
    print("Accuracy for K-Mean @rank 1 : ", "{:.4%}".format(rank_one_score / len(query_idxs)), "K = %d" % num_of_clusters)
    print("Accuracy for K-Mean @rank 5 : ", "{:.4%}".format(rank_five_score / len(query_idxs)), "K = %d" % num_of_clusters)
    print("Accuracy for K-Mean @rank 10 : ", "{:.4%}".format(rank_ten_score / len(query_idxs)), "K = %d" % num_of_clusters)

    print("Computation Time: %s seconds" % (end_time - start_time))
