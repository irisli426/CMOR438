import pytest
import numpy as np
from my_ml_package.supervised.randomforest import RandomForest

@pytest.fixture
def sample_data():
    """Create a simple dataset for testing."""
    X = np.random.rand(20, 5)  # 20 samples, 5 features
    y = np.random.randint(0, 2, size=20)  # Binary target
    return X, y

def test_rf_initialization():
    """Check if parameters are stored correctly."""
    model = RandomForest(n_estimators=50, random_state=10)
    assert model.n_estimators == 50
    assert model.random_state == 10
    assert model.feature_importances_ is None

def test_rf_fit_predict(sample_data):
    """Check if fit and predict work and return the right shapes."""
    X, y = sample_data
    model = RandomForest(n_estimators=10)
    
    # Test Fit
    model.fit(X, y)
    assert model.feature_importances_ is not None
    assert len(model.feature_importances_) == 5
    
    # Test Predict
    y_pred = model.predict(X)
    assert y_pred.shape == (20,)
    assert np.all((y_pred == 0) | (y_pred == 1))

def test_rf_score(sample_data):
    """Check if the score method returns a valid accuracy."""
    X, y = sample_data
    model = RandomForest(n_estimators=10)
    model.fit(X, y)
    
    accuracy = model.score(X, y)
    assert 0.0 <= accuracy <= 1.0