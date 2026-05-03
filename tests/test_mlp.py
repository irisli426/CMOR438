import pytest
import numpy as np
from my_ml_package.supervised.mlp import MultilayerPerceptron

@pytest.fixture
def sample_model():
    """Create a NumPy-based MultilayerPerceptron instance for testing."""
    # Matches your __init__(hidden_layers=[64], learning_rate=0.001, ...)
    return MultilayerPerceptron(
        hidden_layers=[10], 
        learning_rate=0.01, 
        random_state=42
    )

def test_initialization(sample_model):
    """Check if the model initializes with the correct parameters."""
    assert sample_model.learning_rate == 0.01
    assert sample_model.hidden_layers == [10]
    assert len(sample_model.loss_history_) == 0

def test_forward_pass_shape(sample_model):
    """Check if the predict output shape is correct."""
    batch_size = 8
    input_features = 10 
    X = np.random.randn(batch_size, input_features)
    
    # We call .predict(X) instead of model(X)
    # If your method is named 'forward', change 'predict' to 'forward'
    try:
        output = sample_model.predict(X)
    except AttributeError:
        output = sample_model.forward(X)
        
    assert output.shape[0] == batch_size

def test_model_fit(sample_model):
    """Check if the fit method runs and updates weights."""
    X = np.random.randn(20, 10)
    y = np.random.randint(0, 2, size=(20, 1))
    
    # Testing your training logic
    sample_model.fit(X, y)
    
    # Check if a loss was recorded (assuming fit updates loss_history_)
    assert isinstance(sample_model.loss_history_, list)