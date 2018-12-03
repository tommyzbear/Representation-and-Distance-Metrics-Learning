from scipy.io import loadmat
import json
from k_nearest_neighbour import KNN
from sklearn.neighbors import NearestNeighbors
import numpy as np
import random
import time

dir = "D:/EEE Year4/Representation-and-Distance-Metrics-Learning/PR_data/"
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

n = 10
start_time = time.time()

rank_one_score = 0
rank_five_score = 0
rank_ten_score = 0
for i in range(len(query_idxs)):
    feature_vector = features[query_idxs[i] - 1]
    gallery_vectors = features[unsorted_rank_list[i] - 1]
    gallery_labels = labels[unsorted_rank_list[i] - 1]
    knn = KNN(feature_vector, gallery_vectors, gallery_labels)

    neighbours, neighbours_labels = knn.nearest_neighbours(n_nearest_neighbours=n)
    if labels[query_idxs[i] - 1] in neighbours_labels:
        rank_ten_score += 1
        if labels[query_idxs[i] - 1] in neighbours_labels[:5]:
            rank_five_score += 1
            if labels[query_idxs[i] - 1] == neighbours_labels[0]:
                rank_one_score += 1
    # knn = NearestNeighbors(n_neighbors=1)
    # knn.fit(gallery_vectors)
    # rank_list = knn.kneighbors(np.asarray([feature_vector]), n_neighbors=5, return_distance=False)
    # rank_list_labels = []
    # for element in rank_list[0]:
    #     rank_list_labels.append(gallery_labels[element])
    # if labels[query_idxs[i] - 1] in rank_list_labels:
    #     score += 1
end_time = time.time()
print("Accuracy for Simple Nearest Neighbour @rank 1 : ", "{:.2%}".format(rank_one_score/len(query_idxs)))
print("Accuracy for Simple Nearest Neighbour @rank 5 : ", "{:.2%}".format(rank_five_score/len(query_idxs)))
print("Accuracy for Simple Nearest Neighbour @rank 10 : ", "{:.2%}".format(rank_ten_score/len(query_idxs)))

print("Computation Time: %s seconds" % (end_time - start_time))
