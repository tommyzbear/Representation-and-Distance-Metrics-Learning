from scipy.io import loadmat
import json
from k_nearest_neighbour import KNN
import numpy as np
import random
import time
import os
from numba import jit
from sklearn import preprocessing


@jit
def compute_NN_result(data_features, query_indices, query_gallery, method_name):
    n = 10
    start_time = time.time()
    rank_one_score = 0
    rank_five_score = 0
    rank_ten_score = 0
    for k in range(len(query_indices)):
        feature_vector = data_features[query_indices[k] - 1]
        gallery_vectors = data_features[query_gallery[k] - 1]
        gallery_labels = labels[query_gallery[k] - 1]
        knn = KNN(feature_vector, gallery_vectors, gallery_labels, method_name)

        neighbours, cluster_labels = knn.nearest_neighbours(n_nearest_neighbours=n)
        if labels[query_indices[k] - 1] in cluster_labels:
            rank_ten_score += 1
            if labels[query_indices[k] - 1] in cluster_labels[:5]:
                rank_five_score += 1
                if labels[query_indices[k] - 1] == cluster_labels[0]:
                    rank_one_score += 1

    end_time = time.time()
    print("Accuracy for Simple Nearest Neighbour @rank 1 : ", "{:.4%}".format(rank_one_score / len(query_indices)))
    print("Accuracy for Simple Nearest Neighbour @rank 5 : ", "{:.4%}".format(rank_five_score / len(query_indices)))
    print("Accuracy for Simple Nearest Neighbour @rank 10 : ", "{:.4%}".format(rank_ten_score / len(query_indices)))

    print("Computation Time: %s seconds" % (end_time - start_time))


# Sorting data for later use
dir = os.path.dirname(os.path.realpath(__file__)) + "\\PR_data\\"
cuhk03_data = loadmat(dir + 'cuhk03_new_protocol_config_labeled.mat')

# index starts from 1
train_idxs = cuhk03_data['train_idx'].flatten()
query_idxs = cuhk03_data['query_idx'].flatten()
gallery_idxs = cuhk03_data['gallery_idx'].flatten()
labels = cuhk03_data['labels'].flatten()
cam_Id = cuhk03_data['camId'].flatten()

with open(dir + 'feature_data.json', 'r') as f:
    features = json.load(f)

features = np.asarray(features)

train_labels = labels[train_idxs - 1]

# Get all identities in training set
train_distinct_labels = set([])
for label in train_labels:
    train_distinct_labels.add(label)

# Randomly select 100 validation identities
validation_labels = random.sample(train_distinct_labels, 100)

print(validation_labels)

# Get indices of the validation identities
validation_idxs = []
for i in range(len(labels)):
    if labels[i] in validation_labels:
        validation_idxs.append(i + 1)
validation_idxs = np.asarray(validation_idxs)

# Remove validation identities from training set
train_idxs = list(train_idxs)
for idx in validation_idxs:
    if idx in train_idxs:
        train_idxs.remove(idx)
train_idxs = np.asarray(train_idxs)

# Get training features
train_features = features[train_idxs - 1]

# Get validation features
validation_features = features[validation_idxs - 1]

# Get query features
query_features = features[query_idxs - 1]

# Get correct labels for training set
train_labels = labels[train_idxs - 1]

# Get correct labels for testing set
query_labels = labels[query_idxs - 1]

gallery_data_idx = []
for idx in query_idxs:
    sample_gallery_list = []
    temp_cam_id = cam_Id[idx - 1]
    temp_label = labels[idx - 1]
    for gallery_idx in gallery_idxs:
        if cam_Id[gallery_idx - 1] != temp_cam_id or labels[gallery_idx - 1] != temp_label:
            sample_gallery_list.append(gallery_idx)
    gallery_data_idx.append(np.asarray(sample_gallery_list))

# Compute baseline Simple Nearest Neighbour
print("-----Baseline Simple NN------")
compute_NN_result(features, query_idxs, gallery_data_idx, method_name='Euclidean')

# Min Max Normalization on features
print("-----Simple NN with min-max normalization-----")
min_max_scaler = preprocessing.MinMaxScaler()
normalized_features = min_max_scaler.fit_transform(features)

compute_NN_result(normalized_features, query_idxs, gallery_data_idx, method_name='Euclidean')

# Standardization on features
print("-----Simple NN with Standardization-----")
normalized_features = preprocessing.scale(features)

compute_NN_result(normalized_features, query_idxs, gallery_data_idx, method_name='Euclidean')
# # Compute Simple Nearest Neighbour to get baseline measurements
# dist_metrics = ['Euclidean',
#                 'Manhattan',
#                 'Chessboard',
#                 'Cosine',
#                 'Correlation',
#                 'Intersection',
#                 'KL_Divergence',
#                 'JS_Divergence',
#                 'Quadratic_form_distance',
#                 'Mahalanobis']
#
# for method in dist_metrics:
#     compute_NN_result(features, query_idxs, gallery_data_idx, method)
