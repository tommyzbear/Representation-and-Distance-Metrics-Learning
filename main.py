from scipy.io import loadmat
import json
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import classification_report
import numpy as np
import random

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
        if cam_Id[gallery_idx - 1] != temp_cam_id and labels[gallery_idx - 1] != temp_label:
            sample_rank_list.append(gallery_idx)
    unsorted_rank_list.append(np.asarray(sample_rank_list))

score = 0
for i in range(len(query_idxs)):
    knn = NearestNeighbors(n_neighbors=1)
    feature_vector = features[query_idxs[i] - 1]
    gallery_vectors = features[unsorted_rank_list[i] - 1]
    knn.fit(gallery_vectors)
    neighbour = knn.kneighbors(np.asarray([feature_vector]), n_neighbors=5, return_distance=False)
    #if unsorted_rank_list[i][neighbour[0][0]] == query_idxs[i]:
    rank_list = []
    for element in neighbour:
        rank_list.append(element[0])
    if query_idxs[i] in np.asarray(unsorted_rank_list[i])[rank_list]:
        score += 1

print(score)

