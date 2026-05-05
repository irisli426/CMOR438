# Machine Learning Example Notebooks

This directory contains executable Jupyter Notebooks (`.ipynb`) and supporting documentation that demonstrate the usage, implementation, and core concepts of the models available in this library.

Each algorithm folder contains a demonstration notebook and a dedicated README explaining the specific parameters and mathematical approach for that model.

---

## Primary Goals

- **Demonstration** — Show how to initialize, train, and predict using every core model.
- **Best Practices** — Emphasize critical preprocessing steps (like standardization) and proper evaluation to prevent common errors like data leakage.
- **Visualization** — Provide clear visual outputs — such as decision boundaries, convergence curves, and cluster plots — to illustrate model behavior.

---

## Directory Structure Overview

The examples are organized by learning paradigm to help you quickly find the relevant algorithm:

### 1. `Supervised_Learning/`

Models that learn from labeled data to predict specific targets.

| Directory | Core Model | Key Demonstration Focus |
|---|---|---|
| `Decision_Tree/` | `DecisionTreeClassifier` | Tree depth, Gini impurity, and pruning strategies. |
| `Ensemble_Methods/` | `RandomForest`, `AdaBoost` | Boosting vs. bagging and out-of-bag error estimation. |
| `KNN/` | `KNearestNeighbors` | Distance-based logic for classification and regression. |
| `Linear_Regression/` | `LinearRegression` | Ordinary Least Squares and Gradient Descent optimization. |
| `Logistic_Regression/` | `LogisticRegression` | Binary classification using the Sigmoid function. |
| `MLP/` | `MultilayerPerceptron` | Neural network training via backpropagation. |
| `Perceptron/` | `Perceptron` | The fundamental building block of neural architectures. |
| `Random_Forest/` | `RandomForest` | Robust classification through bagging and feature sampling. |

### 2. `Unsupervised_Learning/`

Models that discover hidden patterns or structures in unlabeled data.

| Directory | Core Model | Key Demonstration Focus |
|---|---|---|
| `DBSCAN/` | `DBSCAN` | Density-based clustering and automated noise detection. |
| `KMeans/` | `KMeans` | Centroid-based clustering and inertia minimization. |
| `PCA/` | `PCA` | Dimensionality reduction while preserving maximum variance. |

---

## Essential Preprocessing Note

A robust machine learning workflow requires proper data handling. In nearly all examples — especially those involving distance calculations (`KNN`, `KMeans`, `DBSCAN`) or Gradient Descent (`Linear`/`Logistic Regression`, `MLP`) — **feature scaling is mandatory**.

- **Standardization** — Features should be standardized (Z-score scaling) before training.
- **The Golden Rule** — Always `fit` your scaler **only on the training data**, then `transform` both training and test data. This prevents data leakage and ensures your model generalizes to unseen points.

---

## How to Use These Examples

1. **Navigate** — Choose an algorithm folder (e.g., `Supervised_Learning/Random_Forest/`).
2. **Read** — Check the local `README.md` in that folder for specific implementation details.
3. **Execute** — Open the `.ipynb` file and run the cells sequentially.
4. **Experiment** — Feel free to modify hyperparameters (like `learning_rate` or `max_depth`) to see how the visualizations and accuracy metrics change in real-time.