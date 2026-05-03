# K-Means Clustering: Defining Global Emissions Profiles

## Algorithm Overview
K-Means is a centroid-based unsupervised learning algorithm that partitions a dataset into $k$ distinct, non-overlapping clusters. It works by minimizing the **Inertia**, or the "Within-Cluster Sum of Squares" (WCSS).

### The Iterative Process
The algorithm follows two primary steps until convergence:
1. **Assignment:** Each data point is assigned to the nearest centroid based on the Euclidean distance:
   $$d(x, c) = \sqrt{\sum_{i=1}^{n} (x_i - c_i)^2}$$
2. **Update:** Centroids are recalculated as the geometric mean of all points assigned to that cluster:
   $$c_i = \frac{1}{|S_i|} \sum_{x_j \in S_i} x_j$$



---

## Strategy: Selecting the Optimal $k$
A core challenge of Unsupervised Learning is that there are no "correct" labels. We utilized the **Elbow Method** to find the balance between model simplicity and accuracy.

* **The Elbow Point:** By plotting Inertia vs. Number of Clusters, we identified $k=4$ as the optimal point where the "gain" in information begins to level off.

---

## Emissions Profiles (Cluster Results)
By clustering **Total GHG** and **Per Capita** emissions, we identified four distinct global categories:

| Cluster | Profile Name | Characterization |
| :--- | :--- | :--- |
| **0** | Developing/Low Impact | Minimal industrial footprint; majority of nations. |
| **1** | Industrial Giants | High-population nations driving global volume (e.g., China, USA). |
| **2** | Mid-Scale Industrial | Modern industrialized nations with high efficiency. |
| **3** | High-Intensity Outliers | Nations with extreme per-capita footprints (e.g., Oil producers). |

---

## Setup & Execution
To run this notebook, ensure the following nested project structure:

```text
notebooks/
└── Unsupervised_Learning/
    └── KMeans_Clustering/
        ├── kmeans.ipynb
        └── README.md