# Principal Component Analysis (PCA): Dimensionality Reduction of Agricultural Features

This directory contains a Jupyter notebook demonstrating **Principal Component Analysis (PCA)** to simplify and visualize high-dimensional agricultural datasets.

## Project Overview
Agricultural data often contains dozens of correlated features. PCA is used here to "compress" these features into two or three **Principal Components**, allowing us to visualize 8D data on a 2D scatter plot.

### Key Objectives
* **Dimensionality Reduction**: Transform a large set of variables into a smaller one that still contains most of the original information.
* **Variance Analysis**: Use "Scree Plots" to determine how many components are needed to explain 95% of the data's behavior.
* **Cluster Visualization**: Project the `agriculture.csv` dataset into 2D space to see if different crop types naturally group together.

## The Model: PCA
PCA is an unsupervised technique that performs an orthogonal transformation (Eigen-decomposition) to convert correlated variables into a set of linearly uncorrelated principal components.

### Why PCA for this Dataset?
* **Visualization**: You cannot graph 8 variables at once; PCA allows us to "see" the relationships in 2D.
* **Noise Reduction**: By focusing on the components with the highest variance, we effectively filter out "noise" from the dataset.