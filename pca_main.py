from scipy.io import loadmat
import json
from k_nearest_neighbour import KNN
import numpy as np
import random
import time
import os
from numba import jit
from pca import PCA
from pca_lda import PCA_LDA
from metric_learn import MMC_Supervised
import progressbar

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

gallery_data_idx = []
for idx in query_idxs:
    sample_gallery_list = []
    temp_cam_id = cam_Id[idx - 1]
    temp_label = labels[idx - 1]
    for gallery_idx in gallery_idxs:
        if cam_Id[gallery_idx - 1] != temp_cam_id or labels[gallery_idx - 1] != temp_label:
            sample_gallery_list.append(gallery_idx)
    gallery_data_idx.append(np.asarray(sample_gallery_list))

# Set up progress bar
bar = progressbar.ProgressBar(maxval=len(query_idxs),
                              widgets=[progressbar.Bar('=', '[', ']'), ' ', progressbar.Percentage()])

# PCA learning
# print("-----PCA-----")
# pca = PCA(original_train_features, original_train_labels, M=500, low_dimension=False)
# pca.fit()
#
# projected_query_features = pca.project(query_features)
#
# n = 10
# start_time = time.time()
# rank_one_score = 0
# rank_five_score = 0
# rank_ten_score = 0
#
# bar.start()
#
# for k in range(len(projected_query_features)):
#     feature_vector = projected_query_features[k]
#     gallery_vectors = features[gallery_data_idx[k] - 1]
#     projected_gallery_vectors = pca.project(gallery_vectors)
#     gallery_labels = labels[gallery_data_idx[k] - 1]
#     knn = KNN(feature_vector, projected_gallery_vectors, gallery_labels)
#
#     neighbours, cluster_labels = knn.nearest_neighbours(n_nearest_neighbours=n)
#     if query_labels[k] in cluster_labels:
#         rank_ten_score += 1
#         if query_labels[k] in cluster_labels[:5]:
#             rank_five_score += 1
#             if query_labels[k] == cluster_labels[0]:
#                 rank_one_score += 1
#
#     bar.update(k + 1)
#
# bar.finish()
# end_time = time.time()
# print("Accuracy for Simple Nearest Neighbour @rank 1 : ", "{:.4%}".format(rank_one_score / len(query_labels)))
# print("Accuracy for Simple Nearest Neighbour @rank 5 : ", "{:.4%}".format(rank_five_score / len(query_labels)))
# print("Accuracy for Simple Nearest Neighbour @rank 10 : ", "{:.4%}".format(rank_ten_score / len(query_labels)))
#
# print("Computation Time: %s seconds" % (end_time - start_time))

# PCA-MMC
print("-----PCA_MMC-----")
pca = PCA(original_train_features, original_train_labels, M=500, low_dimension=False)
pca.fit()
mmc = MMC_Supervised(max_iter=20, convergence_threshold=1e-5)
mmc_metric = mmc.fit(pca.train_sample_projection, original_train_labels)
transformed_features = mmc_metric.transform(features)
transformed_query_features = transformed_features[query_idxs - 1]

n = 10
start_time = time.time()
rank_one_score = 0
rank_five_score = 0
rank_ten_score = 0
bar.start()
for k in range(len(query_features)):
    bar.update(k + 1)
    feature_vector = transformed_query_features[k]
    gallery_vectors = transformed_features[gallery_data_idx[k] - 1]
    gallery_labels = labels[gallery_data_idx[k] - 1]
    knn = KNN(feature_vector, gallery_vectors, gallery_labels)

    neighbours, cluster_labels = knn.nearest_neighbours(n_nearest_neighbours=n)
    if query_labels[k] in cluster_labels:
        rank_ten_score += 1
        if query_labels[k] in cluster_labels[:5]:
            rank_five_score += 1
            if query_labels[k] == cluster_labels[0]:
                rank_one_score += 1

bar.finish()
end_time = time.time()
print("Accuracy for Simple Nearest Neighbour @rank 1 : ", "{:.4%}".format(rank_one_score / len(query_labels)))
print("Accuracy for Simple Nearest Neighbour @rank 5 : ", "{:.4%}".format(rank_five_score / len(query_labels)))
print("Accuracy for Simple Nearest Neighbour @rank 10 : ", "{:.4%}".format(rank_ten_score / len(query_labels)))

print("Computation Time: %s seconds" % (end_time - start_time))


# PCA_LDA
# print("-----PCA_LDA-----")
# n = 10
# start_time = time.time()
# rank_one_score = 0
# rank_five_score = 0
# rank_ten_score = 0
#
# pca_lda = PCA_LDA(train_features, train_labels, M_pca=500, M_lda=num_of_distinct_train_labels - 1)
# pca_lda.fit()
#
# projected_query_features = pca_lda.project(query_features)
#
# bar.start()
#
# for k in range(len(projected_query_features)):
#     feature_vector = projected_query_features[k]
#     gallery_vectors = features[gallery_data_idx[k] - 1]
#     projected_gallery_vectors = pca_lda.project(gallery_vectors)
#     gallery_labels = labels[gallery_data_idx[k] - 1]
#     knn = KNN(feature_vector, projected_gallery_vectors, gallery_labels)
#
#     neighbours, cluster_labels = knn.nearest_neighbours(n_nearest_neighbours=n)
#     if query_labels[k] in cluster_labels:
#         rank_ten_score += 1
#         if query_labels[k] in cluster_labels[:5]:
#             rank_five_score += 1
#             if query_labels[k] == cluster_labels[0]:
#                 rank_one_score += 1
#
#     bar.update(k + 1)
#
# bar.finish()
# end_time = time.time()
# print("Accuracy for Simple Nearest Neighbour @rank 1 : ", "{:.4%}".format(rank_one_score / len(query_labels)))
# print("Accuracy for Simple Nearest Neighbour @rank 5 : ", "{:.4%}".format(rank_five_score / len(query_labels)))
# print("Accuracy for Simple Nearest Neighbour @rank 10 : ", "{:.4%}".format(rank_ten_score / len(query_labels)))
#
# print("Computation Time: %s seconds" % (end_time - start_time))

