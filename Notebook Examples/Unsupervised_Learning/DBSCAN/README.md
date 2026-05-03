# DBSCAN: Density-Based Outlier Detection in Global Emissions

## Algorithm Overview
**DBSCAN** (Density-Based Spatial Clustering of Applications with Noise) is a powerful unsupervised clustering algorithm that groups data points based on their spatial density. Unlike K-Means, DBSCAN does not require a pre-defined number of clusters and is highly effective at identifying outliers (noise).

### Core Mechanics: Density Connectivity
The algorithm classifies every point into one of three categories:
1. **Core Points:** Points that have at least `min_samples` within a distance of `eps`.
2. **Border Points:** Points within `eps` distance of a Core Point but with fewer than `min_samples` neighbors themselves.
3. **Noise Points (-1):** Points that are neither Core nor Border points. These are the "Outliers."



---

## Model Strategy: Finding the "Super-Polluters"
In the context of climate data, we use DBSCAN to identify nations that do not follow the standard global emissions density. 

| Parameter | Value | Logic |
| :--- | :--- | :--- |
| **$\epsilon$ (Epsilon)** | `0.5` | The radius of the search area. Standardized features allow for a meaningful distance metric. |
| **`min_samples`** | `5` | The minimum number of countries needed to define a "standard" emissions profile. |

### Why DBSCAN?
While K-Means forced every country into a group, DBSCAN allows the world's largest emitters to exist in their own category: **Noise**. This mathematically proves that countries like China, the USA, and India are "statistically unique" in their environmental impact.

---

## Key Insights from the Data
* **Global Homogeneity:** Most nations (Cluster 0) cluster tightly together, showing a common baseline for emissions.
* **The Value of Noise:** The model identified 12 "Noise" nations. These are the global outliers whose industrial scale or per-capita intensity is so high that they cannot be grouped with the rest of the world.
* **Feature Sensitivity:** Outlier status is driven heavily by **Total GHG volume**, proving that size is the dominant factor in "uniqueness" for global pollution.

---

## Setup & Execution
To run this notebook, ensure the following nested project structure:

```text
notebooks/
└── Unsupervised_Learning/
    └── DBSCAN_Outliers/
        ├── dbscan.ipynb
        └── README.md