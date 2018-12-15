import numpy as np
import image_data_processor as idp
#from numba import jit


class PCA:
    """Principle Component Analysis"""
    def __init__(self,
                 train_samples,
                 M,
                 low_dimension=False):
        self.train_samples = train_samples.T
        self.num_of_train_samples = train_samples.shape[0]
        self.resolution = train_samples.shape[1]
        self.M = M
        self.low_dimension = low_dimension
        self.train_avg_vector = np.mean(train_samples, axis=0)
        self.best_eig_vectors = False
        self.projected = False
        self.train_sample_projection = None
        # Only used for low dimension case
        self.dimensioned_eig_vectors = None
        self.covariance = None

#    @jit
    def fit(self):
        # normalize training samples
        normalized_train_samples = (self.train_samples.T - self.train_avg_vector).T

        # Calculate covariance matrix
        if self.low_dimension is False:
            self.covariance = (1 / self.num_of_train_samples) * np.matmul(normalized_train_samples,
                                                                          normalized_train_samples.T)
        else:
            self.covariance = (1 / self.num_of_train_samples) * np.matmul(normalized_train_samples.T,
                                                                          normalized_train_samples)

        # Compute eigen values and eigen vectors of the covariance matrix
        eig_values, eig_vectors = np.linalg.eig(self.covariance)

        # Plot Eigenvalues
        # idp.plot_eig_values(eig_values.real)

        # Retrieve largest M eigen value indices in the array
        largest_eig_value_indices = np.argsort(eig_values)[-self.M:]

        # Initialize best eigen vectors
        self.best_eig_vectors = np.zeros((len(self.covariance), self.M))

        # Retrieve corresponding eigen vectors mapping to top M eigen values
        for i in range(0, self.M):
            self.best_eig_vectors[:, i] = eig_vectors[:, largest_eig_value_indices[i]].real

        # Compute projections of training samples onto eigen space
        if self.low_dimension is False:
            self.train_sample_projection = np.matmul(normalized_train_samples.T, self.best_eig_vectors)
        else:
            # Compute eigen vector that matches the dimension using relationship u = Av,
            # where u is eigen vector of size D, v is eigen vector of size N<<D, A is normalized training faces
            self.dimensioned_eig_vectors = np.matmul(normalized_train_samples, self.best_eig_vectors).T
            for v in self.dimensioned_eig_vectors:
                idp.normalization(v)
            self.dimensioned_eig_vectors = self.dimensioned_eig_vectors.T
            self.train_sample_projection = np.matmul(normalized_train_samples.T, self.dimensioned_eig_vectors)

        self.projected = True

#    @jit
    def project(self, samples):
        if self.projected is False:
            raise Exception("Need to train PCA first by using pca.fit().")
        else:
            normalized_samples = (samples - self.train_avg_vector).T
            if self.low_dimension is False:
                return np.matmul(normalized_samples.T, self.best_eig_vectors)
            else:
                return np.matmul(normalized_samples.T, self.dimensioned_eig_vectors)
