# Multilayer Perceptron (MLP) for Climate Trend Classification

This directory contains a Jupyter notebook demonstrating a **Multilayer Perceptron (MLP)** classifier implemented from scratch using `NumPy`. The project applies this neural network to global temperature anomaly data to classify years into "High Warming" or "Low Warming" categories based on regional temperature signatures.

## What This Notebook Covers
* **Data Synthesis:** Managing a high-dimensional dataset with 4,811 regional anomalies.
* **From-Scratch MLP Implementation:** A complete object-oriented approach to building a neural network without using `scikit-learn`'s MLP module.
* **Backpropagation & Gradient Descent:** Manual implementation of the chain rule to update weights and biases.
* **Nonlinear Classification:** Using ReLU and Sigmoid activations to learn patterns that a simple Perceptron cannot.
* **Loss Analysis:** Tracking Binary Cross-Entropy loss over 1,500+ iterations to ensure model stability.

## Model Overview: Multilayer Perceptron

An MLP is a feedforward artificial neural network that generates an output from a set of inputs. It consists of:
1. **Input Layer:** Receives the scaled regional temperature data.
2. **Hidden Layer(s):** Applies weights and non-linear transformations (ReLU) to extract complex features.
3. **Output Layer:** Uses a Sigmoid activation to produce a probability (0.0 to 1.0) for binary classification.

### Activation Functions
* **ReLU (Hidden):** `f(z) = max(0, z)`. This allows the network to learn non-linear boundaries.
* **Sigmoid (Output):** `σ(z) = 1 / (1 + exp(-z))`. This maps the final prediction to a probability range, where > 0.5 is classified as "High Warming."

### Loss Function
The model minimizes **Binary Cross-Entropy Loss**, which penalizes the difference between the predicted probability and the actual binary label:
`Loss = -[y · log(y_hat) + (1 − y) · log(1 − y_hat)]`

## Dataset: Environment Temperature Change
The model analyzes the **Environment_Temperature_change_E_All_Data_NOFLAG.csv** dataset.
* **Features:** Annual temperature updates for every country and region globally.
* **Target:** A binary label (1 if the global temperature change is above the historical median, 0 otherwise).
* **Pre-processing:** Data is standardized using `StandardScaler` to ensure the high-dimensional features (thousands of regions) contribute equally to the gradient updates.

## Results & Evaluation
The performance is measured using:
* **Accuracy Score:** The percentage of years correctly classified in the test set.
* **Classification Report:** Detailed breakdown of Precision, Recall, and F1-score for both "Low Warming" and "High Warming" classes.

The MLP successfully recognizes the "mathematical signature" of global warming by synthesizing thousands of tiny regional changes simultaneously, outperforming simpler linear models.

## How to Use
1. Ensure `Environment_Temperature_change_E_All_Data_NOFLAG.csv` is in the same directory as the notebook.
2. Run the `MultilayerPerceptron` class definition cell.
3. Execute the training cell to see the loss curve visualization.
4. Review the final Classification Report to see the model's predictive power.

## Technical Notes
* Implementation relies on `NumPy` for matrix operations.
* **Xavier Initialization** is used to set the initial weights, preventing the signal from exploding or vanishing as it passes through the layers.
* **Numerical Stability:** A "clipped" sigmoid function is used to prevent overflow errors during training.