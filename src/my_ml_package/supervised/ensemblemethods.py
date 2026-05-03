import numpy as np
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier

class AdaBoost:
    def __init__(self, n_estimators=50, learning_rate=0.5, random_state=42):
        """
        A reusable AdaBoost Classifier wrapper using Decision Stumps.
        
        Parameters:
        - n_estimators: Maximum number of estimators at which boosting is terminated.
        - learning_rate: Weight applied to each classifier at each boosting iteration.
        - random_state: Seed used by the random number generator.
        """
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.random_state = random_state
        
        # Initialize the internal engine with a Decision Stump (max_depth=1)
        self.model = AdaBoostClassifier(
            estimator=DecisionTreeClassifier(max_depth=1),
            n_estimators=self.n_estimators,
            learning_rate=self.learning_rate,
            random_state=self.random_state
        )
        
        # To be populated after fitting
        self.feature_importances_ = None

    def fit(self, X, y):
        """
        Build a boosted classifier from the training set (X, y).
        """
        # Scikit-learn expects y to be 1D for AdaBoost
        if len(y.shape) > 1 and y.shape[1] == 1:
            y = y.ravel()
            
        self.model.fit(X, y)
        self.feature_importances_ = self.model.feature_importances_

    def predict(self, X):
        """
        Predict classes for X.
        """
        return self.model.predict(X)

    def score(self, X, y):
        """
        Return the mean accuracy on the given test data and labels.
        """
        return self.model.score(X, y)