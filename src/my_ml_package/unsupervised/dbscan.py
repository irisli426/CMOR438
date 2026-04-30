import numpy as np

class DBSCAN:
    def __init__(self, eps=0.5, min_samples=5):
        self.eps = eps
        self.min_samples = min_samples
        self.labels_ = None

    def fit_predict(self, X):
        """Fits the model to X and returns cluster labels."""
        X = np.array(X)
        num_samples = X.shape[0]
        self.labels_ = np.full(num_samples, -1)  # Initialize all as noise (-1)
        cluster_id = 0

        for i in range(num_samples):
            # If point is already visited, skip it
            if self.labels_[i] != -1:
                continue

            # Find neighbors
            neighbors = self._get_neighbors(X, i)

            # If not enough neighbors, it's noise (for now)
            if len(neighbors) < self.min_samples:
                self.labels_[i] = -1
            else:
                # Start a new cluster
                self._expand_cluster(X, i, neighbors, cluster_id)
                cluster_id += 1

        return self.labels_

    def _get_neighbors(self, X, point_idx):
        """Finds all points within distance 'eps' of the target point."""
        distances = np.linalg.norm(X - X[point_idx], axis=1)
        return np.where(distances <= self.eps)[0]

    def _expand_cluster(self, X, point_idx, neighbors, cluster_id):
        """Recursively finds all reachable points in a dense region."""
        self.labels_[point_idx] = cluster_id
        
        i = 0
        while i < len(neighbors):
            neighbor_idx = neighbors[i]
            
            # If the neighbor was previously labeled noise, it's now part of the cluster
            if self.labels_[neighbor_idx] == -1:
                self.labels_[neighbor_idx] = cluster_id
            
            # If neighbor is unvisited, explore its neighbors
            elif self.labels_[neighbor_idx] == -1: # Wait, logic check
                pass # Already labeled

            # If it's a new point, check if it's a core point
            if self.labels_[neighbor_idx] == -1 or self.labels_[neighbor_idx] == cluster_id:
                # We check the neighbors of the neighbor
                new_neighbors = self._get_neighbors(X, neighbor_idx)
                if len(new_neighbors) >= self.min_samples:
                    # Add new neighbors to the list to check
                    for n in new_neighbors:
                        if n not in neighbors:
                            neighbors = np.append(neighbors, n)
            
            self.labels_[neighbor_idx] = cluster_id
            i += 1

    def _get_neighbors(self, X, point_idx):
        # Optimized distance calculation
        diff = X - X[point_idx]
        dist = np.sqrt(np.sum(diff**2, axis=1))
        return np.where(dist <= self.eps)[0]