import numpy as np
from my_ml_package.supervised.decision_tree import DecisionTree

def test_decision_tree_simple_split():
    # Create data where if X is > 5, Y is 100. If X is <= 5, Y is 0.
    X = np.array([[1], [2], [9], [10]])
    y = np.array([0, 0, 100, 100])
    
    model = DecisionTree(max_depth=1)
    model.fit(X, y)
    
    # Predict on a new value (7) which is > 5
    prediction = model.predict(np.array([[7]]))
    assert prediction[0] == 100  # The tree should have learned the split

import unittest
import numpy as np
from my_ml_package.supervised.decision_tree import DecisionTree

class TestDecisionTree(unittest.TestCase):
    def test_basic_learning(self):
        # Fake data: 0 and 1 are 'Small' (0), 10 and 11 are 'Large' (1)
        X = np.array([[0], [1], [10], [11]])
        y = np.array([0, 0, 1, 1])
        
        model = DecisionTree(max_depth=2)
        model.fit(X, y)
        predictions = model.predict(X)
        
        # We use np.allclose because the model returns floats (0.0, 1.0)
        self.assertTrue(np.allclose(predictions, y), "Decision Tree failed to learn simple split!")

if __name__ == '__main__':
    unittest.main()