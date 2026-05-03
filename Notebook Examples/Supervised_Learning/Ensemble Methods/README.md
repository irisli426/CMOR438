# Ensemble Methods: AdaBoost for Sequential Climate Learning

This directory contains a Jupyter notebook demonstrating the application of **AdaBoost (Adaptive Boosting)** to identify robust global climate predictors. By combining multiple "weak learners," the model focuses iteratively on the most difficult-to-classify years in the temperature record.

## Project Overview

AdaBoost provides a unique perspective on climate data by prioritizing "edge cases"—years where regional anomalies don't immediately follow the expected global trend. This notebook uses the `Environment_Temperature_change_E_All_Data_NOFLAG.csv` dataset to build a high-accuracy classifier through iterative error correction.

### Key Objectives
* **Sequential Learning:** Understand how AdaBoost assigns higher weights to misclassified data points in each subsequent round.
* **Weak to Strong:** Observe how simple decision stumps (basic geographic thresholds) can be combined into a sophisticated global predictor.
* **Regional Synthesis:** Process 4,811 features to find the specific geographic markers that are hardest for standard models to classify.

## The Model: AdaBoost

AdaBoost is an ensemble learning algorithm that operates sequentially rather than in parallel (like Random Forest).


### Why AdaBoost for this Dataset?
* **Focused Learning:** Since climate data can be noisy due to local weather events, AdaBoost forces the model to "learn from its mistakes" and focus on years where the signal-to-noise ratio is low.
* **Feature Weighting:** It effectively identifies which combination of regional anomalies serves as a "deal-breaker" for accurate global trend classification.
* **Precision:** It is highly effective at reducing bias and creating a tight fit for complex datasets.

## Technical Workflow

1.  **Time-Series Preparation:** Transposing the dataset to align 60+ years of observations with thousands of regional features.
2.  **Binary Target Engineering:** Generating labels based on the median "World" temperature change to define "High Warming" (1) and "Low Warming" (0) states.
3.  **Iterative Training:** Utilizing `AdaBoostClassifier` with Decision Stumps as base estimators.
4.  **Performance Metrics:** Calculating accuracy and generating a classification report to evaluate the precision-recall balance.

## Results & Insights
The notebook demonstrates that AdaBoost can achieve high stability by focusing on the geographic markers that other models might overlook. By the final iteration, the ensemble has created a weighted "voting" system where each regional feature contributes to a collective global prediction.

## Usage
1.  Verify the dataset `Environment_Temperature_change_E_All_Data_NOFLAG.csv` is present in the root folder.
2.  Install required libraries: `pip install pandas numpy matplotlib scikit-learn`.
3.  Open `Ensemble_Methods_Demo.ipynb` and execute the cells to observe the sequential improvement of the model.

## Notes
* **Core Packages:** `pandas`, `numpy`, `sklearn`.
* **Methodology:** Sequential Ensemble Boosting.
* **Focus:** Reducing bias and improving classification of difficult-to-predict climate anomalies.