# Random Forest: Geographic Indicators of Climate Change

This directory contains a Jupyter notebook demonstrating the application of the **Random Forest Algorithm** to identify regional predictors of global climate shifts. The analysis uses an ensemble learning approach to determine which geographic areas are the most significant indicators of global temperature trends.

## Project Overview
The notebook implements a robust classification pipeline to move beyond simple trend analysis and uncover the "feature importance" of different countries and regions in the context of global warming.

### Key Objectives
* **Ensemble Classification:** Predicting "High Warming" vs. "Low Warming" years by aggregating the decisions of multiple decision trees.
* **Feature Importance:** Ranking 4,811 geographic regions to see which local temperature changes most closely mirror or predict global shifts.
* **Noise Reduction:** Utilizing the Random Forest's inherent stability to handle the high variance found in meteorological datasets.

## The Model: Random Forest

Random Forest is an ensemble method that constructs a multitude of decision trees during training. 

### Why Random Forest for Climate Data?
* **Handling High Dimensionality:** With thousands of regional features, Random Forest can effectively isolate the most important variables without the need for manual feature selection.
* **Robustness:** Unlike a single decision layer, the forest averages results to prevent outliers (extreme weather years) from skewing the overall model.
* **Non-Linearity:** It naturally captures non-linear relationships between local anomalies and the global state.

## Technical Workflow

1.  **Data Reshaping:** The `Environment_Temperature_change` dataset is transposed so that geographic areas serve as features (columns) and years serve as observations (rows).
2.  **Target Definition:** The "World" temperature series is used to create binary labels based on the historical median.
3.  **Model Training:** Utilizing `scikit-learn`'s `RandomForestClassifier` with 100 estimators.
4.  **Feature Importance Extraction:** Using the `.feature_importances_` attribute to extract and rank the weight of every country in the final model.

## Evaluation & Results
The notebook evaluates the model using:
* **Accuracy Score:** Measuring the model's ability to classify unseen years.
* **Visual Analysis:** A horizontal bar chart visualizing the Top 10 countries that act as the strongest "geographic indicators" for global temperature change.

## Usage
1.  Place the `Environment_Temperature_change_E_All_Data_NOFLAG.csv` file in the project directory.
2.  Install dependencies: `pip install pandas numpy matplotlib seaborn scikit-learn`.
3.  Run `Random_Forest_Demo.ipynb` to generate the importance rankings and performance metrics.

## Notes
* **Package Dependencies:** `pandas`, `numpy`, `matplotlib`, `seaborn`, and `sklearn`.
* **Data Source:** FAO Environment Temperature Change dataset.