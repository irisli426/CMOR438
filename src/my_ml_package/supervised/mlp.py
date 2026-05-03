import numpy as np

class MultilayerPerceptron:
    def __init__(self, hidden_layers=[64], learning_rate=0.001, max_iter=1000, random_state=42):
        self.hidden_layers = list(hidden_layers)
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.random_state = random_state
        self.loss_history_ = []
        self._rng = np.random.default_rng(random_state)
        self.weights_ = []
        self.biases_ = []

    def fit(self, X, y):
        """Standard API for training the model."""
        X = np.array(X)
        y = np.array(y).reshape(-1, 1)
        
        # 1. Initialize weights and biases based on input feature size
        self._initialize_parameters(X.shape[1])
        
        # 2. Gradient Descent Loop
        for epoch in range(self.max_iter):
            # Forward pass: get activations for all layers
            activations = self._forward(X)
            
            # Calculate Binary Cross-Entropy Loss for history
            loss = self._calculate_loss(y, activations[-1])
            self.loss_history_.append(loss)
            
            # Backward pass: compute gradients
            grads_W, grads_b = self._backward(activations, y)
            
            # Update parameters
            for i in range(len(self.weights_)):
                self.weights_[i] -= self.learning_rate * grads_W[i]
                self.biases_[i] -= self.learning_rate * grads_b[i]

    def predict(self, X, threshold=0.5):
        """Standard API for making binary predictions."""
        X = np.array(X)
        probabilities = self._forward(X)[-1]
        return (probabilities >= threshold).astype(int).reshape(X.shape[0], -1)

    def _initialize_parameters(self, n_features):
        """Sets up the weight matrices and bias vectors using Xavier initialization."""
        layer_sizes = [n_features] + self.hidden_layers + [1]
        
        for i in range(len(layer_sizes) - 1):
            # Xavier/Glorot Initialization for signal stability
            limit = np.sqrt(6 / (layer_sizes[i] + layer_sizes[i+1]))
            W = self._rng.uniform(-limit, limit, (layer_sizes[i], layer_sizes[i+1]))
            b = np.zeros((1, layer_sizes[i+1]))
            
            self.weights_.append(W)
            self.biases_.append(b)

    def _forward(self, X):
        """Propagates inputs through the network layers."""
        activations = [X]
        
        for i in range(len(self.weights_)):
            # Linear Transformation: Z = WX + b
            Z = activations[-1] @ self.weights_[i] + self.biases_[i]
            
            # Activation: Sigmoid for output, ReLU for hidden layers
            if i == len(self.weights_) - 1:
                A = 1 / (1 + np.exp(-np.clip(Z, -20, 20))) # Sigmoid
            else:
                A = np.maximum(0, Z) # ReLU
                
            activations.append(A)
        return activations

    def _backward(self, activations, y):
        """Calculates gradients using the chain rule (backpropagation)."""
        m = y.shape[0]
        grads_W = []
        grads_b = []
        
        # Initial error at the output layer
        delta = activations[-1] - y
        
        # Work backward from output to input
        for i in reversed(range(len(self.weights_))):
            dW = (activations[i].T @ delta) / m
            db = np.sum(delta, axis=0, keepdims=True) / m
            
            grads_W.insert(0, dW)
            grads_b.insert(0, db)
            
            # Calculate error for the next layer (ReLU derivative)
            if i > 0:
                delta = (delta @ self.weights_[i].T) * (activations[i] > 0)
        
        return grads_W, grads_b

    def _calculate_loss(self, y, y_pred):
        """Computes the Logistic Loss (Binary Cross-Entropy)."""
        epsilon = 1e-15 # Prevent log(0)
        return -np.mean(y * np.log(y_pred + epsilon) + (1 - y) * np.log(1 - y_pred + epsilon))