import numpy as np
from sklearn.ensemble import RandomForestClassifier

class RandomForest:
    def __init__(self, n_estimators=100, random_state=42, max_depth=None):
        """
        A reusable Random Forest Classifier wrapper.
        
        Parameters:
        - n_estimators: The number of trees in the forest.
        - random_state: Seed used by the random number generator.
        - max_depth: The maximum depth of the trees.
        """
        self.n_estimators = n_estimators
        self.random_state = random_state
        self.max_depth = max_depth
        
        # Initialize the internal engine
        self.model = RandomForestClassifier(
            n_estimators=self.n_estimators,
            random_state=self.random_state,
            max_depth=self.max_depth
        )
        
        # This will store the feature importances after fitting
        self.feature_importances_ = None

    def fit(self, X, y):
        """
        Build a forest of trees from the training set (X, y).
        """
        # Ensure input is in correct format
        self.model.fit(X, y)
        
        # Extract and store feature importances for your plotting
        self.feature_importances_ = self.model.feature_importances_

    def predict(self, X):
        """
        Predict class for X.
        """
        return self.model.predict(X)

    def score(self, X, y):
        """
        Return the mean accuracy on the given test data and labels.
        """
        return self.model.score(X, y)