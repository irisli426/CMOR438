import numpy as np
import pytest
from my_ml_package.supervised.logistic_regression import LogisticRegression

def test_logistic_regression_fit():
    # Simple linearly separable data
    X = np.array([[1, 2], [2, 3], [3, 4], [10, 11], [11, 12], [12, 13]])
    y = np.array([0, 0, 0, 1, 1, 1])
    
    model = LogisticRegression(learning_rate=0.1, epochs=1000)
    model.fit(X, y)
    predictions = model.predict(X)
    
    # Accuracy should be 100% for this simple set
    assert np.array_equal(predictions, y)

def test_sigmoid_range():
    model = LogisticRegression()
    # Test extreme values to ensure sigmoid is stable
    assert model._sigmoid(0) == 0.5
    assert 0 <= model._sigmoid(100) <= 1
    assert 0 <= model._sigmoid(-100) <= 1