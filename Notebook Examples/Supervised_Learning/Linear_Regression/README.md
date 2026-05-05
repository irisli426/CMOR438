# Linear Regression: Global Emissions Forecasting

## Algorithm Overview
Linear Regression is a fundamental supervised learning algorithm used to model the relationship between a scalar dependent variable ($y$) and one or more independent variables ($X$). This specific implementation uses **Batch Gradient Descent** to iteratively minimize the Mean Squared Error (MSE).

### The Math Behind the Model
The model predicts emissions using the linear combination:

$$\hat{y} = X\mathbf{w} + b$$

**Where:**
* **$\mathbf{w}$ (Weights):** Represent the impact of each feature (e.g., Year, GDP, Country dummies) on total emissions.
* **$b$ (Bias):** The intercept term.
* **Gradient Descent:** The weights are updated in the direction of the steepest descent of the cost function:
    
    $$\mathbf{w} = \mathbf{w} - \eta \cdot \frac{\partial J}{\partial \mathbf{w}}$$
    $$\text{where } \frac{\partial J}{\partial \mathbf{w}} = \frac{1}{n} X^T(\hat{y} - y)$$

---

## Implementation Details

| Feature | Description |
| :--- | :--- |
| **Optimizer** | Custom Batch Gradient Descent. |
| **Loss Function** | Mean Squared Error (MSE). |
| **Scaling** | **Mandatory.** Features are standardized using `StandardScaler` to ensure Gradient Descent convergence. |
| **Categorical Data** | Handled via One-Hot Encoding (converting Country/ISO codes into ~200+ numeric features). |

### Key Hyperparameters
* **Learning Rate ($\eta$):** Set to `0.001` in this notebook to maintain stability given the high dimensionality of encoded data.
* **Epochs:** Set to `10,000` to allow the custom model enough "steps" to reach the global minimum.

---

## Performance Benchmark
A core objective of this notebook is to validate the custom `my_ml_package` against the industry-standard `scikit-learn` library to ensure mathematical correctness.

| Model Implementation | R² Score |
| :--- | :--- |
| **Scikit-Learn (Baseline)** | ~0.7945 |
| **Custom Package Model** | ~0.7944 | 

**Analysis:** The near-identical $R^2$ scores demonstrate that our custom Gradient Descent logic successfully converges to the same optimal parameters as professional-grade libraries.

---

