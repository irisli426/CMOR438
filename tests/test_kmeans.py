import numpy as np
from my_ml_package.unsupervised.kmeans import KMeans
#from src.my_ml_package.unsupervised.kmeans import KMeans  # <--- Note the ".unsupervised"

def test_kmeans_fit():
    X = np.array([[1, 2], [1, 4], [1, 0], [10, 2], [10, 4], [10, 0]])
    model = KMeans(k=2)
    labels = model.fit(X)
    # Check that we got a label for every data point
    assert len(labels) == len(X)
    # Check that we have exactly 2 clusters
    assert len(np.unique(labels)) == 2

def test_kmeans_thorough():
    # 1. Setup clearly separable data
    # Group A centered around [1, 2], Group B centered around [10, 2]
    X = np.array([[1, 2], [1, 1], [2, 2], [10, 2], [10, 1], [9, 2]])
    k = 2
    model = KMeans(k=k)
    labels = model.fit(X)

    # 2. Basic Shape & Type Checks
    assert len(labels) == len(X), "Should return one label per sample"
    assert len(np.unique(labels)) == k, "Should identify exactly K clusters"
    
    # 3. Logic Check: Separation
    # The first 3 points should have the same label, 
    # and the last 3 points should have the same (but different) label.
    assert labels[0] == labels[1] == labels[2]
    assert labels[3] == labels[4] == labels[5]
    assert labels[0] != labels[3]

    # 4. Centroid Check (If your class stores them as model.centroids)
    # The centroids should be close to the means of our groups ([1.33, 1.66] and [9.66, 1.66])
    for center in model.centroids:
        # Check if the center is near one of our expected midpoints
        is_near_a = np.allclose(center, [1.33, 1.66], atol=1.0)
        is_near_b = np.allclose(center, [9.66, 1.66], atol=1.0)
        assert is_near_a or is_near_b