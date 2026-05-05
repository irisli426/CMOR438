import unittest
import numpy as np
from my_ml_package.unsupervised.dbscan import DBSCAN

class TestDBSCAN(unittest.TestCase):
    def test_clustering_and_noise(self):
        # Data: Two points in a tight group, one point in the middle of nowhere
        X = np.array([[1, 1], [1.1, 1.1], [50, 50]])
        
        # eps=2 is plenty of room for the first two, but too small for the third
        model = DBSCAN(eps=2, min_samples=2)
        labels = model.fit_predict(X)
        
        # The first two should share a cluster ID (usually 0)
        self.assertEqual(labels[0], labels[1], "Close points were not clustered together")
        # The outlier should be exactly -1
        self.assertEqual(labels[2], -1, "Isolated point was not labeled as noise")

if __name__ == '__main__':
    unittest.main()

import numpy as np
import pytest
from my_ml_package.unsupervised.dbscan import DBSCAN

def test_dbscan_chain_reaction():
    """
    Test that DBSCAN can follow a 'chain' of points.
    Even if the first and last points are far apart, they should
    belong to the same cluster if connected by intermediate points.
    """
    # Points are 1.0 apart. If eps=1.1, they should all connect.
    X = np.array([
        [0, 0], [1, 0], [2, 0], [3, 0], [4, 0]
    ])
    model = DBSCAN(eps=1.1, min_samples=2)
    labels = model.fit_predict(X)
    
    assert len(np.unique(labels[labels != -1])) == 1, "The chain should form exactly one cluster"
    assert labels[0] == labels[4], "Ends of the chain should have the same label"

def test_dbscan_min_samples_threshold():
    """
    Test that a group only becomes a cluster if it meets min_samples.
    """
    X = np.array([[0,0], [0.1, 0.1]]) # Only 2 points
    
    # If min_samples is 3, these 2 points should be considered noise
    model = DBSCAN(eps=0.5, min_samples=3)
    labels = model.fit_predict(X)
    
    assert np.all(labels == -1), "Points should be noise if min_samples isn't met"

def test_dbscan_border_points():
    """
    Test that border points are included in the cluster.
    Point [2,0] has enough neighbors to be a 'Core Point'.
    Point [2.5, 0] is within eps but doesn't have enough neighbors itself.
    """
    X = np.array([
        [2, 0], [2.1, 0], [1.9, 0], # Core point group
        [2.6, 0]                    # Border point (near core, but isolated)
    ])
    # Core needs 3 neighbors; [2.6, 0] only has 1 neighbor.
    model = DBSCAN(eps=0.7, min_samples=3)
    labels = model.fit_predict(X)
    
    assert labels[3] != -1, "Border point should be part of the cluster"
    assert labels[3] == labels[0], "Border point should share the label of its core point"