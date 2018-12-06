import numpy as np
from sklearn.cluster import KMeans
import os
from scipy.io import loadmat
import json
import time
import matplotlib.pyplot as plt

# Sorting data for later use
dir = os.path.dirname(os.path.realpath(__file__)) + "\\PR_data\\"
cuhk03_data = loadmat(dir + 'cuhk03_new_protocol_config_labeled.mat')

gallery_idxs = cuhk03_data['gallery_idx'].flatten()

with open(dir + 'feature_data.json', 'r') as f:
    features = json.load(f)

features = np.asarray(features)

gallery_vectors = features[gallery_idxs - 1]

num_of_clusters_arr = range(50, 1401, 50)

sum_of_squared_err = np.zeros(len(num_of_clusters_arr))

for i in range(len(num_of_clusters_arr)):
    start_time = time.time()

    k_mean = KMeans(n_clusters=num_of_clusters_arr[i]).fit(gallery_vectors)

    feature_cluster_labels = k_mean.labels_

    centroids = k_mean.cluster_centers_

    sum_of_squared_err[i] = k_mean.inertia_

    end_time = time.time()

    print("Computation Time: %s seconds" % (end_time - start_time), "for K = %d" % num_of_clusters_arr[i])

plt.figure()
plt.title('Elbow Method \n SSE against different number of clusters')
plt.ylabel('Sum of Within Clusters Squared Errors')
plt.xlabel('Number of Clusters')
plt.plot(num_of_clusters_arr, sum_of_squared_err, '.r-')
plt.show()
