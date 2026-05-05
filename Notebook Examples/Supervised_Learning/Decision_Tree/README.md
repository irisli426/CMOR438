# Decision Tree Classification: Identifying High-Emission Nations

## Algorithm Overview
The **Classification and Regression Tree (CART)** algorithm is a non-parametric supervised learning method. In this implementation, the tree builds a hierarchy of binary "questions" to partition the dataset into increasingly homogeneous groups.

### Core Logic: Variance Reduction
While typically used for regression, this implementation uses **Variance Reduction** as the splitting criterion. The goal at each node is to find the feature and threshold that maximizes the "Information Gain" by reducing the variance of the target labels in the resulting child nodes.

$$Gain = \text{Var}(y_{parent}) - \sum_{i \in \{left, right\}} \frac{N_i}{N_{total}} \text{Var}(y_i)$$

---

## Model Strategy: High vs. Low Emitters
Instead of predicting exact GHG tonnage, we transformed the problem into a **Binary Classification** task to identify "High Emitter" countries.

| Label | Definition | Context |
| :--- | :--- | :--- |
| **1 (High)** | Total GHG > Global Median | Targets for strict regulation. |
| **0 (Low)** | Total GHG < Global Median | Developing or low-industrial nations. |

### Hyperparameters
* **`max_depth=3`**: We restricted the tree depth to 3 levels to ensure **interpretability**. This prevents the model from over-specializing (overfitting) and allows us to see the primary "rules" governing global emissions.

---

## Key Results & Visualization
The model achieved high accuracy by identifying a clear threshold in the feature space.

### Decision Boundary Analysis
Using a log-log scale for visualization (Total GHG vs. Per Capita), the model reveals:
1. **The Primary Split:** The most significant predictor of being a "High Emitter" is the absolute Total GHG volume, rather than Per Capita efficiency.
2. **Linear Separability:** In log-space, the boundary between high and low emitters is nearly a straight vertical line, which explains why the Decision Tree can classify these nations with such high precision.

---

## Setup & Execution
To run this notebook, ensure the following nested project structure:

```text
notebooks/
└── Supervised_Learning/
    └── Decision_Tree/
        ├── decision_tree.ipynb
        └── README.md