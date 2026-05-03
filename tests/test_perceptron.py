import numpy as np
from my_ml_package.supervised.perceptron import Perceptron

def test_perceptron_basic():
    # Simple data: [1, 1] should be class 1, [-1, -1] should be class 0
    X = np.array([[1, 1], [2, 2], [-1, -1], [-2, -2]])
    y = np.array([1, 1, 0, 0])
    
    model = Perceptron(learning_rate=0.1, epochs=10)
    model.fit(X, y)
    
    predictions = model.predict(np.array([[1.5, 1.5], [-1.5, -1.5]]))
    
    assert predictions[0] == 1
    assert predictions[1] == 0

def test_perceptron_weights_initialization():
    model = Perceptron()
    X = np.array([[1, 2], [3, 4]])
    y = np.array([1, 0])
    model.fit(X, y)
    assert model.weights is not None
    assert len(model.weights) == 2