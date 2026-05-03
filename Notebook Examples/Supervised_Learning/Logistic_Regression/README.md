# Logistic Regression: Probabilistic Modeling of Yield Success

This directory contains a Jupyter notebook demonstrating **Logistic Regression** to predict the probability of crop success across different geographic sectors.

## Project Overview
Unlike hard classifiers, Logistic Regression provides a "confidence score" (probability). This notebook explores how environmental variables correlate with the likelihood of a high-yield harvest.

### Key Objectives
* **Probability Mapping**: Use the Sigmoid function to map real-valued numbers into a [0, 1] range.
* **Decision Thresholds**: Learn how to adjust the classification threshold (e.g., 0.5) to balance precision and recall.
* **Feature Importance**: Analyze the coefficients to see which environmental factors (e.g., Nitrogen levels) most heavily influence the outcome.

## The Model: Logistic Regression
Logistic Regression is a linear model for binary classification that estimates probabilities using the logistic function.

### Why Logistic Regression for this Dataset?
* **Risk Assessment**: In agriculture, knowing the *probability* of success is often more valuable than a simple Yes/No.
* **Interpretability**: The model coefficients directly tell us how much a unit change in a feature (like pH) affects the odds of the target class.