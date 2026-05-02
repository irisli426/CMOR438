# Supervised Learning Modules

This directory contains implementations of supervised learning algorithms where the model learns from labeled training data.

---

## 1. K-Nearest Neighbors (KNN)
KNN is a non-parametric classification algorithm. It predicts the class of a sample by finding the $k$ most frequent labels among its closest neighbors.

* **Distance Metric**: Euclidean Distance ($L2$ norm).
* **Best Practice**: Always scale features using `StandardScaler` before fitting.

## 2. Perceptron
A fundamental linear binary classifier inspired by biological neurons. It updates its weights based on misclassified points until a separating hyperplane is found.

* **Activation**: Heaviside Step Function.
* **Requirement**: Data must be linearly separable for guaranteed convergence.

## 3. Logistic Regression
A probabilistic linear model used for binary classification. Unlike the Perceptron, it outputs a probability between 0 and 1.

* **Activation**: Sigmoid Function $\sigma(z) = \frac{1}{1 + e^{-z}}$.
* **Loss Function**: Log-Loss (Binary Cross-Entropy).