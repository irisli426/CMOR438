# Perceptron: Linear Decision Boundaries in Crop Classification

This directory contains a Jupyter notebook implementing the **Perceptron**, the fundamental building block of neural networks, applied to binary agricultural classification tasks.

## Project Overview
The Perceptron is used here to identify a linear "split" in agricultural data—for example, determining if a region is "Optimal" or "Sub-optimal" for a specific crop based on a weighted sum of inputs like rainfall and temperature.

### Key Objectives
* **Linear Separability**: Identify whether agricultural features can be cleanly divided by a single straight line (hyperplane).
* **Weight Optimization**: Observe how the model updates its weights iteratively when it encounters a classification error.
* **Convergence**: Test the model's ability to reach a stable state on linearly separable datasets.

## The Model: Perceptron
The Perceptron is a binary linear classifier that uses a step function to convert input signals into a 0 or 1 output.

### Why Perceptron for this Dataset?
* **Baseline Modeling**: It serves as the perfect "simple" baseline to see if complex data can be solved with a basic linear rule.
* **Efficiency**: It is computationally lightweight and provides immediate feedback on whether a linear relationship exists.