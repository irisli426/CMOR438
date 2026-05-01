import numpy as np
from src.my_ml_package.supervised.knn import KNearestNeighbors

def test_knn_logic():
    """
    Test if KNN correctly classifies a simple 2D dataset.
    Group A is near [0,0], Group B is near [10,10].
    """
    # 1. Create simple toy data
    X_train = np.array([[0, 0], [1, 1], [9, 9], [10, 10]])
    y_train = np.array(["Low", "Low", "High", "High"])
    
    # 2. Initialize model
    model = KNearestNeighbors(k_neighbors=3)
    model.fit(X_train, y_train)
    
    # 3. Test a point that is clearly 'Low'
    point_low = np.array([[0.5, 0.5]])
    pred_low = model.predict(point_low)
    
    # 4. Test a point that is clearly 'High'
    point_high = np.array([[9.5, 9.5]])
    pred_high = model.predict(point_high)
    
    assert pred_low[0] == "Low"
    assert pred_high[0] == "High"

def test_knn_k_value():
    """Ensure the model respects the k_neighbors parameter."""
    X = np.array([[1, 1], [2, 2], [3, 3]])
    y = np.array([0, 0, 1])
    model = KNearestNeighbors(k_neighbors=1)
    model.fit(X, y)
    # A point at [2.1, 2.1] should be class 0 if K=1 (closest to [2,2])
    assert model.predict([[2.1, 2.1]])[0] == 0