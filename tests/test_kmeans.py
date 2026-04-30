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