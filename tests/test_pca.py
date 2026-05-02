import numpy as np
import pytest
from src.my_ml_package.unsupervised.pca import PCA

def test_pca_dimensions():
    # Create random 10-feature data
    X = np.random.rand(100, 10)
    k = 3
    pca = PCA(n_components=k)
    X_reduced = pca.fit_transform(X)
    
    assert X_reduced.shape == (100, k)
    assert len(pca.variance_ratio) == k

def test_pca_variance():
    # Data with clear variance on one axis
    X = np.array([[1, 2], [2, 4], [3, 6], [4, 8]])
    pca = PCA(n_components=1)
    pca.fit(X)
    # The first component should explain almost all variance
    assert pca.variance_ratio[0] > 0.9