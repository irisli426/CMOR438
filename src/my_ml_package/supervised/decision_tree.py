import numpy as np

class DecisionTree:
    def __init__(self, max_depth=5):
        self.max_depth = max_depth
        self.tree = None

    def fit(self, X, y):
        """Standard API for training the model."""
        # Ensure data is numpy format to avoid Pandas errors
        X = np.array(X)
        y = np.array(y)
        self.tree = self._build_tree(X, y)

    def _build_tree(self, X, y, depth=0):
        num_samples = len(y)
        
        # 1. Safety check: If no data reached this branch, return a default
        if num_samples == 0:
            return 0
            
        # 2. Stop conditions: Max depth reached or all data has the same label
        if depth >= self.max_depth or len(np.unique(y)) == 1:
            return np.mean(y)

        # 3. Find the best split point
        best_feature, best_threshold = self._get_best_split(X, y)
        
        # 4. If no valid split was found, return the average
        if best_feature is None:
            return np.mean(y)

        # 5. Split the data into left and right branches
        left_idx = X[:, best_feature] <= best_threshold
        right_idx = X[:, best_feature] > best_threshold
        
        left_branch = self._build_tree(X[left_idx], y[left_idx], depth + 1)
        right_branch = self._build_tree(X[right_idx], y[right_idx], depth + 1)
        
        return (best_feature, best_threshold, left_branch, right_branch)

    def _get_best_split(self, X, y):
        """Finds the feature and value that best separates the data."""
        best_gain = -1
        split_idx, split_threshold = None, None
        
        # Loop through every feature (GDP, Population, etc.)
        for feature_idx in range(X.shape[1]):
            thresholds = np.unique(X[:, feature_idx])
            
            # Test different split points for this feature
            for threshold in thresholds:
                left_y = y[X[:, feature_idx] <= threshold]
                right_y = y[X[:, feature_idx] > threshold]
                
                if len(left_y) > 0 and len(right_y) > 0:
                    # Calculate Variance Reduction (Information Gain)
                    gain = np.var(y) - (len(left_y)/len(y)*np.var(left_y) + 
                                       len(right_y)/len(y)*np.var(right_y))
                    
                    if gain > best_gain:
                        best_gain = gain
                        split_idx = feature_idx
                        split_threshold = threshold
                        
        return split_idx, split_threshold

    def predict(self, X):
        """Standard API for making predictions."""
        X = np.array(X)
        return np.array([self._traverse_tree(x, self.tree) for x in X])

    def _traverse_tree(self, x, node):
        """Walks down the tree for a single data point."""
        # If we reached a leaf (a single number), return it
        if not isinstance(node, tuple):
            return node
        
        # Otherwise, ask the 'Question' at this node
        feature, threshold, left, right = node
        if x[feature] <= threshold:
            return self._traverse_tree(x, left)
        return self._traverse_tree(x, right)