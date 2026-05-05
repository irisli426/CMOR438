Source (src) Directory
This directory contains the core implementation of the my_ml_package library. These algorithms are written from scratch using NumPy to mirror the scikit-learn API, prioritizing mathematical clarity and educational value.

Installation & Setup
This project is designed to be installed locally as a Python package to allow for seamless imports across your notebooks and test files.

Create and activate a virtual environment (Recommended)

Bash
python3 -m venv .venv
source .venv/bin/activate  # macOS / Linux
Install required dependencies

Bash
pip install numpy pandas matplotlib pytest
Install my_ml_package in editable mode
From the project root (where the src/ folder lives):

Bash
pip install -e .
This allows you to import your custom modules from anywhere on your system:

Python
from my_ml_package.supervised.knn import KNN
from my_ml_package.unsupervised.pca import PCA
Design Goals
API Consistency: Mimic scikit-learn conventions (.fit(), .predict(), .transform()).

NumPy-Centric: Use only NumPy and core Python to demonstrate the linear algebra behind the models.

Interpretability: Keep code modular so that the relationship between math and code is easy to follow.

Strict Testing: Ensure every model passes a suite of unit tests for numerical correctness.

Standardized Interface
All models in this package follow a consistent workflow:

fit(X, y): Trains the model on labeled data (Supervised).

fit(X): Finds patterns or clusters in data (Unsupervised).

predict(X): Performs inference on new data points.

transform(X): Projects data into new spaces (e.g., PCA).

Supervised Learning (my_ml_package/supervised)
Algorithms designed for classification and regression using historical labels:

Linear & Logistic Regression: Baseline models for continuous and categorical targets.

Perceptron & MLP: Neural architectures ranging from single-layer to multi-layer feedforward networks.

KNN: Distance-based classification.

Decision Trees & Random Forests: Tree-based logic and ensemble bagging techniques.

Unsupervised Learning (my_ml_package/unsupervised)
Algorithms that identify structure in unlabeled data:

K-Means & DBSCAN: Centroid-based and density-based clustering techniques.

PCA: Linear dimensionality reduction used to compress features while maintaining variance.

Testing Philosophy
Every algorithm in this source directory is validated against a corresponding test file in the tests/ directory. Our testing covers:

Mathematical Correctness: Comparing output against known theoretical results.

Input Validation: Handling edge cases like empty arrays or mismatched dimensions.

Convergence: Ensuring iterative models (like Logistic Regression) reach a stable solution.

Run the tests using:

Bash
python3 -m pytest tests/
Intended Use
my_ml_package is built for:

Algorithmic Deep-Dives: Understanding the "black box" of machine learning.

Reproducible Research: Using custom implementations on environmental and agricultural datasets.

Technical Portfolios: Demonstrating the ability to translate mathematical formulas into functional Python code.