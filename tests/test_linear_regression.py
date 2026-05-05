import numpy as np
import pytest
from my_ml_package.supervised.linear_regression import LinearRegression
#from src.my_ml_package.supervised.linear_regression import LinearRegression


def test_model_fit():
    # 1. Setup: Create perfect data (y = 2x)
    X = np.array([[1], [2], [3]])
    y = np.array([2, 4, 6])
    
    # 2. Action: Train the model
    model = LinearRegression(lr=0.01, epochs=2000)
    model.fit(X, y)
    
    # 3. Assert: Check if the prediction for 4 is close to 8
    prediction = model.predict(np.array([[4]]))
    
    # We use np.allclose because floating point math isn't always exact
    assert np.allclose(prediction, [8], atol=0.1)

def test_initialization():
    model = LinearRegression(lr=0.5)
    assert model.lr == 0.5
    assert model.weights is None

import numpy as np
import pytest
from my_ml_package.supervised.linear_regression import LinearRegression

def test_linear_regression_coefficients():
    """
    Verify the model learns the actual slope and intercept,
    not just a close-enough prediction.
    """
    # y = 3x + 5
    X = np.array([[1], [2], [3], [4]])
    y = np.array([8, 11, 14, 17])
    
    model = LinearRegression(lr=0.01, epochs=5000)
    model.fit(X, y)
    
    # Check slope (weight) and intercept (bias)
    # Assuming your model stores them as .weights and .bias
    assert np.isclose(model.weights[0], 3.0, atol=0.1)
    assert np.isclose(model.bias, 5.0, atol=0.1)

