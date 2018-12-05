from scipy.io import loadmat
import json
from k_nearest_neighbour import KNN
from sklearn.cluster import KMeans
from k_mean import KMean
import numpy as np
import random
import time
import os


# Sorting data for later use
dir = os.path.dirname(os.path.realpath(__file__)) + "\\PR_data\\"
cuhk03_data = loadmat(dir + 'cuhk03_new_protocol_config_labeled.mat')

# index starts from 1
train_idxs = list(cuhk03_data['train_idx'].flatten())
query_idxs = cuhk03_data['query_idx'].flatten()
gallery_idxs = cuhk03_data['gallery_idx'].flatten()
labels = cuhk03_data['labels'].flatten()
cam_Id = cuhk03_data['camId'].flatten()

with open(dir + 'feature_data.json', 'r') as f:
    features = json.load(f)

features = np.asarray(features)

train_labels = []
for idx in train_idxs:
    train_labels.append(labels[idx - 1])

# Get all identities in training set
train_distinct_labels = set([])
for cluster in train_labels:
    train_distinct_labels.add(cluster)

# Randomly select 100 validation identities
validation_labels = random.sample(train_distinct_labels, 100)

print(validation_labels)

# Get indices of the validation identities
validation_idxs = []
for i in range(len(labels)):
    if labels[i] in validation_labels:
        validation_idxs.append(i + 1)

# Remove validation identities from training set
for idx in validation_idxs:
    if idx in train_idxs:
        train_idxs.remove(idx)

# Get training features
train_features = []
for idx in train_idxs:
    train_features.append(features[idx - 1])

# Get validation features
validation_features = []
for idx in validation_idxs:
    validation_features.append(features[idx - 1])

# Get query features
query_features = []
for idx in query_idxs:
    query_features.append(features[idx - 1])

# Get correct labels for training set
train_labels = []
for idx in train_idxs:
    train_labels.append(labels[idx - 1])

# Get correct labels for testing set
query_labels = []
for idx in query_idxs:
    query_labels.append(labels[idx - 1])

unsorted_rank_list = []
for idx in query_idxs:
    sample_rank_list = []
    temp_cam_id = cam_Id[idx - 1]
    temp_label = labels[idx - 1]
    for gallery_idx in gallery_idxs:
        if cam_Id[gallery_idx - 1] != temp_cam_id or labels[gallery_idx - 1] != temp_label:
            sample_rank_list.append(gallery_idx)
    unsorted_rank_list.append(np.asarray(sample_rank_list))

# Compute Simple Nearest Neighbour to get baseline measurements
n = 10
# start_time = time.time()
#
# rank_one_score = 0
# rank_five_score = 0
# rank_ten_score = 0
# for i in range(len(query_idxs)):
#     feature_vector = features[query_idxs[i] - 1]
#     gallery_vectors = features[unsorted_rank_list[i] - 1]
#     gallery_labels = labels[unsorted_rank_list[i] - 1]
#     knn = KNN(feature_vector, gallery_vectors, gallery_labels)
#
#     neighbours, cluster_labels = knn.nearest_neighbours(n_nearest_neighbours=n)
#     if labels[query_idxs[i] - 1] in cluster_labels:
#         rank_ten_score += 1
#         if labels[query_idxs[i] - 1] in cluster_labels[:5]:
#             rank_five_score += 1
#             if labels[query_idxs[i] - 1] == cluster_labels[0]:
#                 rank_one_score += 1
#
# end_time = time.time()
# print("Accuracy for Simple Nearest Neighbour @rank 1 : ", "{:.4%}".format(rank_one_score/len(query_idxs)))
# print("Accuracy for Simple Nearest Neighbour @rank 5 : ", "{:.4%}".format(rank_five_score/len(query_idxs)))
# print("Accuracy for Simple Nearest Neighbour @rank 10 : ", "{:.4%}".format(rank_ten_score/len(query_idxs)))
#
# print("Computation Time: %s seconds" % (end_time - start_time))
#
#----------K-Mean Baseline----------
rank_one_score = 0
rank_five_score = 0
rank_ten_score = 0

num_of_clusters = 100

start_time = time.time()

for i in range(len(query_idxs)):
    sub_start_time = time.time()
    feature_vector = features[query_idxs[i] - 1]
    gallery_vectors = features[unsorted_rank_list[i] - 1]
    gallery_labels = labels[unsorted_rank_list[i] - 1]
    k_mean = KMeans(n_clusters=num_of_clusters, max_iter=15).fit(gallery_vectors)
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
    dist_to_centroids = np.linalg.norm(centroids - feature_vector, axis=1)

    result_label = cluster_labels[np.argsort(dist_to_centroids)[:n]]

    if labels[query_idxs[i] - 1] in result_label:
        rank_ten_score += 1
        if labels[query_idxs[i] - 1] in result_label[:5]:
            rank_five_score += 1
            if labels[query_idxs[i] - 1] == result_label[0]:
                rank_one_score += 1

    sub_end_time = time.time()
    print("k-mean time: %s sec" % (sub_end_time - sub_start_time))

end_time = time.time()
print("Accuracy for K-Mean @rank 1 : ", "{:.4%}".format(rank_one_score/len(query_idxs)))
print("Accuracy for K-Mean @rank 5 : ", "{:.4%}".format(rank_five_score/len(query_idxs)))
print("Accuracy for K-Mean @rank 10 : ", "{:.4%}".format(rank_ten_score/len(query_idxs)))

print("Computation Time: %s seconds" % (end_time - start_time))
