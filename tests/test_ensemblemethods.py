import pytest
import numpy as np
from my_ml_package.supervised.ensemblemethods import AdaBoost

@pytest.fixture
def sample_data():
    """Create a small synthetic dataset for testing."""
    # 20 samples, 4 features
    X = np.random.rand(20, 4)
    # Binary target (0 or 1)
    y = np.random.randint(0, 2, size=20)
    return X, y

def test_adaboost_initialization():
    """Check if the parameters are stored correctly in the class."""
    model = AdaBoost(n_estimators=100, learning_rate=0.1, random_state=123)
    assert model.n_estimators == 100
    assert model.learning_rate == 0.1
    assert model.random_state == 123
    assert model.feature_importances_ is None

def test_adaboost_fit_predict(sample_data):
    """Check if the model can train and predict on synthetic data."""
    X, y = sample_data
    model = AdaBoost(n_estimators=10)
    
    # Test training
    model.fit(X, y)
    assert model.feature_importances_ is not None
    # Check that it produces one importance value per feature
    assert len(model.feature_importances_) == X.shape[1]
    
    # Test prediction
    y_pred = model.predict(X)
    assert y_pred.shape == (20,)
    # Ensure values are binary 0 or 1
    assert np.all(np.isin(y_pred, [0, 1]))

def test_adaboost_score(sample_data):
    """Verify the accuracy score method returns a valid float."""
    X, y = sample_data
    model = AdaBoost(n_estimators=5)
    model.fit(X, y)
    
    accuracy = model.score(X, y)
    assert isinstance(accuracy, float)
    assert 0.0 <= accuracy <= 1.0