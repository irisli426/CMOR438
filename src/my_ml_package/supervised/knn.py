import numpy as np

class KNearestNeighbors:
    def __init__(self, k_neighbors=3):
        """
        Custom KNN Classifier.
        :param k_neighbors: Number of neighbors to participate in voting.
        """
        self.k = k_neighbors
        self.points = None
        self.labels = None

    def fit(self, X, y):
        """
        Stores the training data.
        """
        self.points = np.array(X)
        self.labels = np.array(y)

    def _compute_distance(self, p1, p2):
        """Calculates Euclidean distance between two points."""
        return np.sqrt(np.sum((p1 - p2) ** 2))

    def predict(self, X_new):
        """
        Predicts labels for a set of new data points.
        """
        X_new = np.array(X_new)
        predictions = [self._single_prediction(row) for row in X_new]
        return np.array(predictions)

    def _single_prediction(self, sample):
        # 1. Get distances to all training points
        distances = [self._compute_distance(sample, tr_point) for tr_point in self.points]
        
        # 2. Find indices of the k-nearest neighbors
        k_indices = np.argsort(distances)[:self.k]
        
        # 3. Get labels of those neighbors
        k_neighbor_labels = [self.labels[i] for i in k_indices]
        
        # 4. Return the most common label (Majority Vote)
        counts = {}
        for label in k_neighbor_labels:
            counts[label] = counts.get(label, 0) + 1
        
        return max(counts, key=counts.get)