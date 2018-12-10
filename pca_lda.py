from pca import *
import numpy as np
from numba import jit


class PCA_LDA:
    '''Fisherface'''
    def __init__(self,
                 train_samples,
                 train_results,
                 M_pca,
                 M_lda):
        self.train_samples = train_samples
        self.train_results = train_results
        self.num_of_train_samples = train_samples.shape[0]
        self.distinct_labels = distinct_labels(train_results)
        self.num_of_distinct_samples = count_distinct_labels(train_results)
        self.resolution = train_samples.shape[1]
        self.M_pca = M_pca
        self.M_lda = M_lda
        self.train_avg_vector = np.mean(train_samples, axis=0)
        self.opt_eig_vec = None
        self.train_sample_projection = None
        self.class_mean = None
        self.projected = False

    @jit
    def fit(self):
        self.class_mean = []
        for label in self.distinct_labels:
            self.class_mean.append(np.mean(self.train_samples[np.where(self.train_results == label)], axis=0))
        self.class_mean = np.asarray(self.class_mean)
        class_normalized_mean = self.class_mean - self.train_avg_vector

        # Between-class scatter matrix
        S_B = class_normalized_mean.T @ class_normalized_mean

        # Compute x - mi
        discriminant_train_samples = np.zeros((self.train_samples.shape[0], self.train_samples.shape[1]))
        index = 0
        for label in self.distinct_labels:
            idxs = np.where(self.train_results == label)
            discriminant_train_samples[idxs] = self.train_samples[idxs] - class_normalized_mean[index]
            index += 1

        # Within-class scatter matrix
        S_W = discriminant_train_samples.T @ discriminant_train_samples

        # Get low-dimension PCA training projections
        pca = PCA(self.train_samples,
                  self.train_results,
                  self.resolution,
                  self.M_pca)

        pca.fit()
        pca_best_eig_vec = pca.dimensioned_eig_vectors

        # Compute generalized eigenvectors and eigenvalues
        temp = pca_best_eig_vec.T @ S_B @ pca_best_eig_vec
        temp_inv = pca_best_eig_vec.T @ S_W @ pca_best_eig_vec

        lda_eig_val, lda_eig_vec = np.linalg.eig(np.linalg.inv(temp_inv) @ temp)

        # Retrieve largest M eigenvalue indices in the array
        largest_eig_value_indices = np.argsort(lda_eig_val)[-self.M_lda:]

        # Initialize best eigenvectors
        best_lda_eig_vec = np.zeros((lda_eig_vec.shape[0], self.M_lda), dtype=np.complex)

        # Retrieve corresponding eigenvectors mapping to top M eigenvalues
        for i in range(0, self.M_lda):
            best_lda_eig_vec[:, i] = lda_eig_vec[:, largest_eig_value_indices[i]]

        self.opt_eig_vec = best_lda_eig_vec.T @ pca_best_eig_vec.T

        # normalize training samples
        normalized_train_samples = self.train_samples - self.train_avg_vector

        self.train_sample_projection = normalized_train_samples @ self.opt_eig_vec.T

        self.projected = True

    @jit
    def project(self, samples):
        if self.projected is False:
            raise Exception("Need to train PCA_LDA first by using .fit().")
        else:
            normalized_samples = (samples - self.train_avg_vector)
            return np.matmul(normalized_samples, self.opt_eig_vec.T)


class LDA:
    '''LDA'''
    def __init__(self,
                 train_samples,
                 train_results,
                 M_lda,
                 pca):
        self.train_samples = train_samples
        self.train_results = train_results
        self.num_of_train_samples = train_samples.shape[0]
        self.distinct_labels = distinct_labels(train_results)
        self.num_of_distinct_samples = count_distinct_labels(train_results)
        self.resolution = train_samples.shape[1]
        self.M_lda = M_lda
        self.pca = pca
        self.train_avg_vector = np.mean(train_samples, axis=0)
        self.opt_eig_vec = None
        self.train_sample_projection = None
        self.class_mean = None

    def fit(self):
        self.class_mean = []
        for label in self.distinct_labels:
            self.class_mean.append(np.mean(self.train_samples[np.where(self.train_results == label)], axis=0))
        self.class_mean = np.asarray(self.class_mean)
        class_normalized_mean = self.class_mean - self.train_avg_vector

        # Between-class scatter matrix
        S_B = class_normalized_mean.T @ class_normalized_mean

        # Compute x - mi
        discriminant_train_samples = np.zeros((self.train_samples.shape[0], self.train_samples.shape[1]))
        index = 0
        for label in self.distinct_labels:
            idxs = np.where(self.train_results == label)
            discriminant_train_samples[idxs] = self.train_samples[idxs] - class_normalized_mean[index]
            index += 1

        # Within-class scatter matrix
        S_W = discriminant_train_samples.T @ discriminant_train_samples

        pca_best_eig_vec = self.pca.dimensioned_eig_vectors

        # Compute generalized eigenvectors and eigenvalues
        temp = pca_best_eig_vec.T @ S_B @ pca_best_eig_vec
        temp_inv = pca_best_eig_vec.T @ S_W @ pca_best_eig_vec

        lda_eig_val, lda_eig_vec = np.linalg.eig(np.linalg.inv(temp_inv) @ temp)

        # Retrieve largest M eigenvalue indices in the array
        largest_eig_value_indices = np.argsort(lda_eig_val)[-self.M_lda:]

        # Initialize best eigenvectors
        best_lda_eig_vec = np.zeros((lda_eig_vec.shape[0], self.M_lda), dtype=np.complex)

        # Retrieve corresponding eigenvectors mapping to top M eigenvalues
        for i in range(0, self.M_lda):
            best_lda_eig_vec[:, i] = lda_eig_vec[:, largest_eig_value_indices[i]]

        self.opt_eig_vec = best_lda_eig_vec.T @ pca_best_eig_vec.T

        # normalize training samples
        normalized_train_samples = self.train_samples - self.train_avg_vector

        self.train_sample_projection = normalized_train_samples @ self.opt_eig_vec.T


def count_distinct_labels(distinct_labels):
    return len(distinct_labels)


@jit
def distinct_labels(labels):
    distinct_identities = set({})
    for label in labels:
        distinct_identities.add(label)
    return distinct_identities
