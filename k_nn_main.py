from scipy.io import loadmat
import json
from k_nearest_neighbour import KNN
import numpy as np
import random
import time
import os
from numba import jit
from data_normaliser import Normaliser
from metric_learn import LMNN, MMC_Supervised, NCA, ITML_Supervised
from pca import PCA


@jit
def compute_NN_result(query_data,
                      gallery_data,
                      test_labels,
                      gallery_test_labels,
                      query_gallery_idxs,
                      method_name='Euclidean'):
    print("Computing KNN ranklist...")
    n = 10
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

    print("\nAccuracy for Simple Nearest Neighbour @rank 1 : ", "{:.4%}".format(rank_one_score / len(query_data)))
    print("Accuracy for Simple Nearest Neighbour @rank 5 : ", "{:.4%}".format(rank_five_score / len(query_data)))
    print("Accuracy for Simple Nearest Neighbour @rank 10 : ", "{:.4%}".format(rank_ten_score / len(query_data)))

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
original_train_features = features[original_train_idxs - 1]

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


# Compute baseline Simple Nearest Neighbour
print("-----Baseline Simple NN------")
compute_NN_result(query_features, gallery_features, query_labels, gallery_labels, gallery_data_idx)

# Validation query and gallery
validation_query_labels = random.sample(validation_labels, 30)
validation_query_idxs = []
for validation_query_label in validation_query_labels:
    for validation_idx in validation_idxs:
        if cam_Id[validation_idx - 1] == 1 and labels[validation_idx - 1] == validation_query_label:
            validation_query_idxs.append(validation_idx)
            break
validation_query_idxs = np.asarray(validation_query_idxs)

validation_gallery_idxs = validation_idxs.copy()
for validation_query_idx in validation_query_idxs:
    if validation_query_idx in validation_gallery_idxs:
        validation_gallery_idxs.remove(validation_query_idx)
validation_gallery_idxs = np.asarray(validation_gallery_idxs)

validation_query_features = features[validation_query_idxs - 1]
validation_gallery_features = features[validation_gallery_idxs - 1]

validation_query_labels = labels[validation_query_idxs - 1]
validation_gallery_labels = labels[validation_gallery_idxs - 1]

validation_gallery_data_idx = []
for validation_query_idx in validation_query_idxs:
    sample_gallery_list = []
    temp_cam_id = cam_Id[validation_query_idx - 1]
    temp_label = labels[validation_query_idx - 1]
    for i in range(len(validation_gallery_idxs)):
        if cam_Id[validation_gallery_idxs[i] - 1] != temp_cam_id or labels[validation_gallery_idxs[i] - 1] != temp_label:
            sample_gallery_list.append(i)
    validation_gallery_data_idx.append(np.asarray(sample_gallery_list))

# Compute validation baseline KNN
print("-----Validation for Simple NN-----")
compute_NN_result(validation_query_features,
                  validation_gallery_features,
                  validation_query_labels,
                  validation_gallery_labels,
                  validation_gallery_data_idx)

# Compute PCA result
print("\n-----PCA------")
pca = PCA(original_train_features, M=500)
pca.fit()
pca_query_features = pca.project(query_features)
pca_gallery_features = pca.project(gallery_features)
compute_NN_result(pca_query_features, pca_gallery_features, query_labels, gallery_labels, gallery_data_idx)

# Compute LMNN (Large Margin Nearest Neighbour) Learning
print("\n-----LMNN------")
lmnn = LMNN(k=5, max_iter=20, use_pca=False, convergence_tol=1e-6, learn_rate=1e-6, verbose=True)
lmnn.fit(original_train_features, original_train_labels)
transformed_query_features = lmnn.transform(query_features)
transformed_gallery_features = lmnn.transform(gallery_features)
compute_NN_result(transformed_query_features, transformed_gallery_features, query_labels, gallery_labels, gallery_data_idx)

# Compute PCA_LMNN Learning
print("\n-----PCA_LMNN-----")
lmnn = LMNN(k=5, max_iter=20, use_pca=False, convergence_tol=1e-6, learn_rate=1e-6, verbose=True)
start_time = time.time()
lmnn.fit(pca.train_sample_projection, original_train_labels)
end_time = time.time()
print("Learning time: %s" % (end_time - start_time))
transformed_query_features = lmnn.transform(pca_query_features)
transformed_gallery_features = lmnn.transform(pca_gallery_features)
compute_NN_result(transformed_query_features, transformed_gallery_features, query_labels, gallery_labels, gallery_data_idx)

# Compute NCA (Neighbourhood Components Analysis) Learning
print("\n-----NCA-----")
nca = NCA(max_iter=20, verbose=True)
nca.fit(original_train_features, original_train_labels)
transformed_query_features = nca.transform(query_features)
transformed_gallery_features = nca.transform(gallery_features)
compute_NN_result(transformed_query_features, transformed_gallery_features, query_labels, gallery_labels, gallery_data_idx)

# Compute PCA_NCA Learning
print("\n-----PCA_NCA-----")
nca = NCA(max_iter=20, verbose=True)
start_time = time.time()
nca.fit(pca.train_sample_projection, original_train_labels)
end_time = time.time()
print("Learning time: %s" % (end_time - start_time))
transformed_query_features = nca.transform(pca_query_features)
transformed_gallery_features = nca.transform(pca_gallery_features)
compute_NN_result(transformed_query_features, transformed_gallery_features, query_labels, gallery_labels, gallery_data_idx)

# Compute ITML (Information Theoretic Metric Learning)
print("\n-----ITML-----")
itml = ITML_Supervised(max_iter=20, convergence_threshold=1e-5, num_constraints=500, verbose=True)
itml.fit(original_train_features, original_train_labels)
transformed_query_features = itml.transform(query_features)
transformed_gallery_features = itml.transform(gallery_features)
compute_NN_result(transformed_query_features, transformed_gallery_features, query_labels, gallery_labels, gallery_data_idx)

# Compute PCA_ITML
print("\n-----PCA_ITML-----")
itml = ITML_Supervised(max_iter=20, convergence_threshold=1e-5, num_constraints=500, verbose=True)
start_time = time.time()
itml.fit(pca.train_sample_projection, original_train_labels)
end_time = time.time()
print("Learning time: %s" % (end_time - start_time))
transformed_query_features = itml.transform(pca_query_features)
transformed_gallery_features = itml.transform(pca_gallery_features)
compute_NN_result(transformed_query_features, transformed_gallery_features, query_labels, gallery_labels, gallery_data_idx)

# Compute MMC
print("\n-----MMC-----")
mmc = MMC_Supervised(max_iter=20, convergence_threshold=1e-5, num_constraints=500, verbose=True)
mmc.fit(original_train_features, original_train_labels)
transformed_query_features = mmc.transform(query_features)
transformed_gallery_features = mmc.transform(gallery_features)
compute_NN_result(transformed_query_features, transformed_gallery_features, query_labels, gallery_labels, gallery_data_idx)

# Compute PCA_MMC (Mahalanobis Metric Learning for Clustering)
print("\n-----PCA MMC-----")
mmc = MMC_Supervised(max_iter=20, convergence_threshold=1e-5, num_constraints=500, verbose=True)
start_time = time.time()
mmc.fit(pca.train_sample_projection, original_train_labels)
end_time = time.time()
print("Learning time: %s" % (start_time - end_time))
transformed_query_features = mmc.transform(pca_query_features)
transformed_gallery_features = mmc.transform(pca_gallery_features)
compute_NN_result(transformed_query_features, transformed_gallery_features, query_labels, gallery_labels, gallery_data_idx)

print("\n-----MMC diagonal-----")
mmc = MMC_Supervised(max_iter=20, convergence_threshold=1e-5, num_constraints=500, diagonal=True, verbose=True)
mmc.fit(original_train_features, original_train_labels)
transformed_query_features = mmc.transform(query_features)
transformed_gallery_features = mmc.transform(gallery_features)
compute_NN_result(transformed_query_features, transformed_gallery_features, query_labels, gallery_labels, gallery_data_idx)

print("\n-----PCA_MMC diagonal-----")
mmc = MMC_Supervised(max_iter=20, convergence_threshold=1e-5, num_constraints=500, diagonal=True, verbose=True)
start_time = time.time()
mmc.fit(pca.train_sample_projection, original_train_labels)
end_time = time.time()
print("Learning time: %s" % (start_time - end_time))
transformed_query_features = mmc.transform(pca_query_features)
transformed_gallery_features = mmc.transform(pca_gallery_features)
compute_NN_result(transformed_query_features, transformed_gallery_features, query_labels, gallery_labels, gallery_data_idx)

# Compute NN result with normalized data
normalization_methods = ['Std', 'l1', 'l2', 'max', 'MinMax', 'MaxAbs', 'Robust']
for normalization_method in normalization_methods:
    print("-----NN with normalised data using %s normalization method-----" % normalization_method)
    normaliser = Normaliser()
    normaliser.fit(original_train_features, method=normalization_method)
    normalised_query_features = normaliser.transform(query_features)
    normalised_gallery_features = normaliser.transform(gallery_features)
    compute_NN_result(normalised_query_features,
                      normalised_gallery_features,
                      query_labels,
                      gallery_labels,
                      gallery_data_idx)
