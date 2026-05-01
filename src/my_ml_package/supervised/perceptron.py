import numpy as np

class Perceptron:
    def __init__(self, learning_rate=0.01, epochs=1000):
        self.lr = learning_rate
        self.epochs = epochs
        self.weights = None
        self.bias = None

    def fit(self, X, y):
        """
        Train the perceptron using the update rule: 
        w = w + lr * (y_true - y_pred) * x
        """
        n_samples, n_features = X.shape
        
        # 1. Initialize weights and bias to zeros
        self.weights = np.zeros(n_features)
        self.bias = 0

        # 2. Convert labels to 0 and 1 (binary classification)
        y_ = np.array([1 if i > 0 else 0 for i in y])

        # 3. Training loop
        for _ in range(self.epochs):
            for idx, x_i in enumerate(X):
                # Linear model: (w * x) + b
                linear_output = np.dot(x_i, self.weights) + self.bias
                
                # Activation: Step Function
                y_predicted = 1 if linear_output >= 0 else 0

                # Perceptron Update Rule
                update = self.lr * (y_[idx] - y_predicted)
                
                self.weights += update * x_i
                self.bias += update

    def predict(self, X):
        """Predict labels for new data"""
        linear_output = np.dot(X, self.weights) + self.bias
        # Apply the same step function to return 0 or 1
        return np.where(linear_output >= 0, 1, 0)