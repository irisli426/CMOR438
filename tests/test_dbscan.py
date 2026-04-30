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