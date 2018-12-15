from scipy.io import loadmat
import json
from sklearn.cluster import KMeans
import numpy as np
from pca import PCA
import os
import linear_assignment
from statistics import mode
from metric_learn import LMNN, NCA, MMC_Supervised, ITML_Supervised
import time


def compute_k_mean(n_clusters, query_data, gallery_data, gallery_results):
    rank_one_score = 0
    rank_five_score = 0
    rank_ten_score = 0
    ap = 0
    print("Processing K-means clustering...")
    k_mean = KMeans(n_clusters=n_clusters).fit(gallery_data)
    feature_clusters = k_mean.labels_
    linear_assigned_cluster_labels = linear_assignment.best_map(gallery_results, feature_clusters)
    cluster_labels = np.zeros(n_clusters)
    for i in range(n_clusters):
        cluster_labels[i] = mode(linear_assigned_cluster_labels[np.where(feature_clusters == i)])
    centroids = k_mean.cluster_centers_
    for i in range(len(query_data)):
        dist_to_centroids = np.linalg.norm(centroids - query_data[i], axis=1)

        result_label = cluster_labels[np.argsort(dist_to_centroids)[:n]]

        if query_labels[i] in result_label:
            rank_ten_score += 1
            if query_labels[i] in result_label[:5]:
                rank_five_score += 1
                if query_labels[i] == result_label[0]:
                    rank_one_score += 1

    print("\nAccuracy for K-Mean @rank 1 : ", "{:.4%}".format(rank_one_score / len(query_data)),
          "K = %d" % n_clusters)
    print("Accuracy for K-Mean @rank 5 : ", "{:.4%}".format(rank_five_score / len(query_data)),
          "K = %d" % n_clusters)
    print("Accuracy for K-Mean @rank 10 : ", "{:.4%}".format(rank_ten_score / len(query_data)),
          "K = %d" % n_clusters)


# Sorting data for later use
dir = os.path.dirname(os.path.realpath(__file__)) + "\\PR_data\\"
cuhk03_data = loadmat(dir + 'cuhk03_new_protocol_config_labeled.mat')

# index starts from 1
original_train_idxs = cuhk03_data['train_idx'].flatten()
query_idxs = cuhk03_data['query_idx'].flatten()
gallery_idxs = cuhk03_data['gallery_idx'].flatten()
labels = cuhk03_data['labels'].flatten()

with open(dir + 'feature_data.json', 'r') as f:
    features = json.load(f)

features = np.asarray(features)

# Get original train features and labels
original_train_features = features[original_train_idxs - 1]
original_train_labels = labels[original_train_idxs - 1]

# Get query features and labels
query_features = features[query_idxs - 1]
query_labels = labels[query_idxs - 1]

# Get gallery features and labels
gallery_features = features[gallery_idxs - 1]
gallery_labels = labels[gallery_idxs - 1]

# num_of_clusters_arr = [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]
num_of_clusters = 700
n = 10

# Compute K-Means baseline solution
print("-----Baseline K-Means-----")
compute_k_mean(num_of_clusters, query_features, gallery_features, gallery_labels)

# Compute PCA result
print("\n-----PCA------")
pca = PCA(original_train_features, M=500)
pca.fit()
pca_query_features = pca.project(query_features)
pca_gallery_features = pca.project(gallery_features)
compute_k_mean(num_of_clusters, pca_query_features, pca_gallery_features, gallery_labels)

# Compute LMNN (Large Margin Nearest Neighbour) Learning
print("\n-----LMNN------")
lmnn = LMNN(k=5, max_iter=20, use_pca=False, convergence_tol=1e-6, learn_rate=1e-6, verbose=True)
lmnn.fit(original_train_features, original_train_labels)
transformed_query_features = lmnn.transform(query_features)
transformed_gallery_features = lmnn.transform(gallery_features)
compute_k_mean(num_of_clusters, transformed_query_features, transformed_gallery_features, gallery_labels)

# Compute PCA_LMNN Learning
print("\n-----PCA_LMNN-----")
lmnn = LMNN(k=5, max_iter=20, use_pca=False, convergence_tol=1e-6, learn_rate=1e-6, verbose=True)
start_time = time.time()
lmnn.fit(pca.train_sample_projection, original_train_labels)
end_time = time.time()
print("Learning time: %s" % (end_time - start_time))
transformed_query_features = lmnn.transform(pca_query_features)
transformed_gallery_features = lmnn.transform(pca_gallery_features)
compute_k_mean(num_of_clusters, transformed_query_features, transformed_gallery_features, gallery_labels)

# Compute NCA (Neighbourhood Components Analysis) Learning
# print("\n-----NCA-----")
# nca = NCA(max_iter=20, verbose=True)
# nca.fit(original_train_features, original_train_labels)
# transformed_query_features = nca.transform(query_features)
# transformed_gallery_features = nca.transform(gallery_features)
# compute_k_mean(num_of_clusters, transformed_query_features, transformed_gallery_features, gallery_labels)

# Compute PCA_NCA Learning
print("\n-----PCA_NCA-----")
nca = NCA(max_iter=20, verbose=True)
start_time = time.time()
nca.fit(pca.train_sample_projection, original_train_labels)
end_time = time.time()
print("Learning time: %s" % (end_time - start_time))
transformed_query_features = nca.transform(pca_query_features)
transformed_gallery_features = nca.transform(pca_gallery_features)
compute_k_mean(num_of_clusters, transformed_query_features, transformed_gallery_features, gallery_labels)
#
# # Compute ITML (Information Theoretic Metric Learning)
# print("\n-----ITML-----")
# itml = ITML_Supervised(max_iter=20, convergence_threshold=1e-5, num_constraints=500, verbose=True)
# itml.fit(original_train_features, original_train_labels)
# transformed_query_features = itml.transform(query_features)
# transformed_gallery_features = itml.transform(gallery_features)
# compute_k_mean(num_of_clusters, transformed_query_features, transformed_gallery_features, gallery_labels)

# Compute PCA_ITML
print("\n-----PCA_ITML-----")
itml = ITML_Supervised(max_iter=20, convergence_threshold=1e-5, num_constraints=500, verbose=True)
start_time = time.time()
itml.fit(pca.train_sample_projection, original_train_labels)
end_time = time.time()
print("Learning time: %s" % (end_time - start_time))
transformed_query_features = itml.transform(pca_query_features)
transformed_gallery_features = itml.transform(pca_gallery_features)
compute_k_mean(num_of_clusters, transformed_query_features, transformed_gallery_features, gallery_labels)

# Compute PCA_MMC (Mahalanobis Metric Learning for Clustering)
print("\n-----PCA_MMC-----")
mmc = MMC_Supervised(max_iter=20, convergence_threshold=1e-5, num_constraints=500, verbose=True)
start_time = time.time()
mmc.fit(pca.train_sample_projection, original_train_labels)
end_time = time.time()
print("Learning time: %s" % (end_time - start_time))
transformed_query_features = mmc.transform(pca_query_features)
transformed_gallery_features = mmc.transform(pca_gallery_features)
compute_k_mean(num_of_clusters, transformed_query_features, transformed_gallery_features, gallery_labels)

print("\n-----MMC diagonal-----")
mmc = MMC_Supervised(max_iter=20, convergence_threshold=1e-5, num_constraints=500, diagonal=True, verbose=True)
mmc.fit(original_train_features, original_train_labels)
transformed_query_features = mmc.transform(query_features)
transformed_gallery_features = mmc.transform(gallery_features)
compute_k_mean(num_of_clusters, transformed_query_features, transformed_gallery_features, gallery_labels)

print("\n-----PCA_MMC diagonal-----")
mmc = MMC_Supervised(max_iter=20, convergence_threshold=1e-5, num_constraints=500, diagonal=True, verbose=True)
start_time = time.time()
mmc.fit(pca.train_sample_projection, original_train_labels)
end_time = time.time()
print("Learning time: %s" % (end_time - start_time))
transformed_query_features = mmc.transform(pca_query_features)
transformed_gallery_features = mmc.transform(pca_gallery_features)
compute_k_mean(num_of_clusters, transformed_query_features, transformed_gallery_features, gallery_labels)
