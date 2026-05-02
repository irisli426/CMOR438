# Unsupervised Learning Modules

This directory contains algorithms that identify patterns and structures within unlabeled data.

---

## 1. Principal Component Analysis (PCA)
PCA is a dimensionality reduction technique that transforms high-dimensional datasets into a lower-dimensional form while retaining as much variance as possible.

### Key Features:
* **Variance Maximization**: Identifies the "Principal Components" (eigenvectors) that capture the most information.
* **Data Compression**: Reduces features from $N$ to $K$ dimensions to simplify models or for 2D/3D visualization.
* **Math Foundation**: Built on the Eigen-decomposition of the data's Covariance Matrix.

> **Note**: PCA is highly sensitive to the scale of features. Ensure data is centered and scaled before applying PCA.