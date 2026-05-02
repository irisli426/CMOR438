import numpy as np

class PCA:
    """
    Principal Component Analysis (PCA) implementation to reduce dimensionality 
    while preserving maximum variance.
    """
    def __init__(self, n_components=None):
        self.k = n_components
        self.eigenvectors = None
        self.variance_values = None
        self.variance_ratio = None
        self.center_point = None

    def fit(self, data):
        # 1. Store the mean and center the data
        self.center_point = np.mean(data, axis=0)
        centered_data = data - self.center_point

        # 2. Calculate Covariance Matrix
        # rowvar=False means columns are features
        cov_mat = np.cov(centered_data, rowvar=False)

        # 3. Eigen Decomposition
        values, vectors = np.linalg.eigh(cov_mat)

        # 4. Sort in descending order
        indices = np.argsort(values)[::-1]
        values = values[indices]
        vectors = vectors[:, indices]

        # 5. Subset to k components
        if self.k is not None:
            values = values[:self.k]
            vectors = vectors[:, :self.k]

        self.eigenvectors = vectors.T
        self.variance_values = values
        self.variance_ratio = values / np.sum(np.linalg.eigvalsh(cov_mat))

        return self

    def transform(self, data):
        if self.eigenvectors is None:
            raise Exception("Model must be fitted before transforming.")
        
        # Center and project onto principal axes
        centered_data = data - self.center_point
        return np.dot(centered_data, self.eigenvectors.T)

    def fit_transform(self, data):
        self.fit(data)
        return self.transform(data)