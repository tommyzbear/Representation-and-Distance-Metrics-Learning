from scipy.io import loadmat
import json
from k_nearest_neighbour import KNN
import numpy as np
import random
import time
import os
from numba import jit
from sklearn import preprocessing
from data_normaliser import Normaliser
import seaborn as sns
import matplotlib.pyplot as plt
from metric_learn import LMNN
import progressbar


# Set up progress bar
bar = progressbar.ProgressBar(maxval=1400,
                              widgets=[progressbar.Bar('=', '[', ']'), ' ', progressbar.Percentage()])


@jit
def compute_NN_result(query_data,
                      gallery_data,
                      test_labels,
                      gallery_test_labels,
                      query_gallery_idxs,
                      method_name='Euclidean'):
    n = 10
    start_time = time.time()
    rank_one_score = 0
    rank_five_score = 0
    rank_ten_score = 0
    for i in range(len(test_labels)):
        feature_vector = query_data[i]
        gallery_vectors = gallery_data[query_gallery_idxs[i]]
        gallery_labels = gallery_test_labels[query_gallery_idxs[i]]
        knn = KNN(feature_vector, gallery_vectors, gallery_labels, method_name)

        neighbours, cluster_labels = knn.nearest_neighbours(n_nearest_neighbours=n)
        if test_labels[i] in cluster_labels:
            rank_ten_score += 1
            if test_labels[i] in cluster_labels[:5]:
                rank_five_score += 1
                if test_labels[i] == cluster_labels[0]:
                    rank_one_score += 1
        bar.update(i + 1)

    end_time = time.time()
    print("Accuracy for Simple Nearest Neighbour @rank 1 : ", "{:.4%}".format(rank_one_score / len(query_data)))
    print("Accuracy for Simple Nearest Neighbour @rank 5 : ", "{:.4%}".format(rank_five_score / len(query_data)))
    print("Accuracy for Simple Nearest Neighbour @rank 10 : ", "{:.4%}".format(rank_ten_score / len(query_data)))

    print("Computation Time: %s seconds" % (end_time - start_time))


# Sorting data for later use
dir = os.path.dirname(os.path.realpath(__file__)) + "\\PR_data\\"
cuhk03_data = loadmat(dir + 'cuhk03_new_protocol_config_labeled.mat')

# index starts from 1
original_train_idxs = cuhk03_data['train_idx'].flatten()
query_idxs = cuhk03_data['query_idx'].flatten()
gallery_idxs = cuhk03_data['gallery_idx'].flatten()
labels = cuhk03_data['labels'].flatten()
cam_Id = cuhk03_data['camId'].flatten()

with open(dir + 'feature_data.json', 'r') as f:
    features = json.load(f)

features = np.asarray(features)

original_train_labels = labels[original_train_idxs - 1]
original_train_features = features[original_train_labels - 1]

# Get all identities in training set
train_distinct_labels = set([])
for label in original_train_labels:
    train_distinct_labels.add(label)
num_of_distinct_train_labels = len(train_distinct_labels)

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
train_idxs = list(original_train_idxs)
for idx in validation_idxs:
    if idx in original_train_idxs:
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

# Get Gallery features
gallery_features = features[gallery_idxs - 1]

# Get Gallery labels
gallery_labels = labels[gallery_idxs - 1]

gallery_data_idx = []
for idx in query_idxs:
    sample_gallery_list = []
    temp_cam_id = cam_Id[idx - 1]
    temp_label = labels[idx - 1]
    for j in range(len(gallery_idxs)):
        if cam_Id[gallery_idxs[j] - 1] != temp_cam_id or labels[gallery_idxs[j] - 1] != temp_label:
            sample_gallery_list.append(j)
    gallery_data_idx.append(np.asarray(sample_gallery_list))

# Compute covariance of original training set
# training_covariance = np.cov(original_train_features.T)
# sns.heatmap(training_covariance, center=0, vmin=-1, vmax=1, cmap="YlGnBu")
# plt.show()
# Compute baseline Simple Nearest Neighbour
# print("-----Baseline Simple NN------")
# compute_NN_result(query_features, gallery_features, query_labels, gallery_labels, gallery_data_idx)
#
# # Compute NN result with normalized data
# normalization_methods = ['Std', 'l1', 'l2', 'max', 'MinMax', 'MaxAbs', 'Robust']
# for normalization_method in normalization_methods:
#     print("-----NN with normalised data using %s normalization method-----" % normalization_method)
#     normaliser = Normaliser()
#     normaliser.fit(original_train_features, method=normalization_method)
#     normalised_query_features = normaliser.transform(query_features)
#     normalised_gallery_features = normaliser.transform(gallery_features)
#     compute_NN_result(normalised_query_features,
#                       normalised_gallery_features,
#                       query_labels,
#                       gallery_labels,
#                       gallery_data_idx)

# Compute LMNN
lmnn = LMNN(k=5, use_pca=False, convergence_tol=1e-7, verbose=True)
lmnn.fit(original_train_features, original_train_labels)
transformed_query_features = lmnn.transform(query_features)
transformed_gallery_features = lmnn.transform(gallery_features)
bar.start()
compute_NN_result(transformed_query_features,
                  transformed_gallery_features,
                  query_labels,
                  gallery_labels,
                  gallery_data_idx)
bar.finish()
# # Min Max Normalization on features
# print("-----Simple NN with min-max normalization-----")
# min_max_scaler = preprocessing.MinMaxScaler()
# normalized_features = min_max_scaler.fit_transform(features)
#
# compute_NN_result(normalized_features, query_idxs, gallery_data_idx, method_name='Euclidean')
#
# # Standardization on features
# print("-----Simple NN with Standardization-----")
# normalized_features = preprocessing.scale(features)
#
# compute_NN_result(normalized_features, query_idxs, gallery_data_idx, method_name='Euclidean')
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
