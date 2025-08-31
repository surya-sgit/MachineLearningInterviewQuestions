# K-Means Clustering – 100 Interview Questions & Answers

---

## Basics (Q1–Q20)

### Q1. What is K-Means clustering?  
K-Means is an unsupervised learning algorithm that groups data into k clusters based on feature similarity. Each data point belongs to the nearest cluster centroid. It aims to minimize the sum of squared distances between data points and their cluster centroid.

### Q2. Is K-Means supervised or unsupervised?  
It is an **unsupervised** algorithm. It doesn’t require labeled data. It discovers inherent groupings in the dataset.

### Q3. What is the cost function of K-Means?  
The cost is **Within-Cluster Sum of Squares (WCSS)**:  
\[
J = \sum_{i=1}^{k}\sum_{x \in C_i} ||x - \mu_i||^2
\]  

### Q4. What are the steps in K-Means?  
1. Choose k.  
2. Initialize centroids randomly.  
3. Assign points to nearest centroid.  
4. Recompute centroids.  
5. Repeat until convergence.  

### Q5. What is the role of centroids?  
They represent the cluster centers. They are recalculated as the mean of assigned points.  

### Q6. How do you choose k?  
- Elbow method  
- Silhouette score  
- Gap statistic  
- Domain expertise  

### Q7. What is the Elbow method?  
Plot WCSS vs k. The “elbow” point shows optimal k.  

### Q8. What is the Silhouette score?  
It measures cohesion and separation.  
- +1 → well clustered  
- 0 → boundary  
- Negative → wrong assignment  

### Q9. Difference between K-Means and K-Medoids?  
- K-Means: uses centroids (mean)  
- K-Medoids: uses medoids (actual points)  
Medoids are more robust to outliers.  

### Q10. Difference between K-Means and hierarchical clustering?  
- K-Means requires k, hierarchical doesn’t.  
- K-Means is efficient, hierarchical builds dendrograms.  

### Q11. What distance metric is used in K-Means?  
Mostly Euclidean, but can adapt to Manhattan or cosine.  

### Q12. What is inertia in K-Means?  
Total WCSS, measures compactness of clusters.  

### Q13. Why is K-Means iterative?  
Assignments depend on centroids, centroids depend on assignments. Hence iterative refinement until stable.  

### Q14. What are assumptions of K-Means?  
- Spherical clusters  
- Equal density  
- Features are continuous  

### Q15. Is K-Means deterministic?  
No, random initialization makes it stochastic. Fixing seed makes it deterministic.  

### Q16. What is K-Means++ initialization?  
Centroids chosen far apart to improve convergence and reduce bad local minima.  

### Q17. What happens if k=1?  
All points in one cluster.  

### Q18. What happens if k=N?  
Each point is its own cluster. WCSS = 0, meaningless.  

### Q19. What if clusters are non-spherical?  
K-Means fails. Alternatives: DBSCAN, spectral clustering.  

### Q20. Why is feature scaling important?  
Distances get dominated by large-range features. Standardization ensures fair clustering.  

---

## Advanced Concepts (Q21–Q40)

### Q21. Can K-Means handle categorical data?  
Not directly. Use K-Modes or K-Prototypes for categorical/mixed data.  

### Q22. What is the time complexity of K-Means?  
O(n × k × i × d), where n = samples, k = clusters, i = iterations, d = dimensions.  

### Q23. What is mini-batch K-Means?  
Variant that uses random subsets for faster clustering on large datasets.  

### Q24. Can K-Means find non-convex clusters?  
No, only spherical/convex.  

### Q25. How does outlier affect K-Means?  
Outliers pull centroids away from dense regions.  

### Q26. How to make K-Means robust to outliers?  
- Remove outliers  
- Use K-Medoids  
- Use trimmed K-Means  

### Q27. Difference between hard and soft clustering?  
- Hard: each point belongs to one cluster (K-Means).  
- Soft: probabilistic membership (e.g., GMM).  

### Q28. Can K-Means converge to local optima?  
Yes, due to initialization. K-Means++ helps.  

### Q29. Effect of dimensionality?  
High dimensions make distances meaningless (curse of dimensionality). Dimensionality reduction needed.  

### Q30. How to evaluate K-Means?  
- WCSS  
- Silhouette score  
- Davies-Bouldin index  
- Rand index (if ground truth exists)  

### Q31. What is Davies-Bouldin Index?  
Measures average similarity between clusters. Lower is better.  

### Q32. Applications of K-Means?  
- Market segmentation  
- Image compression  
- Topic modeling  
- Customer profiling  

### Q33. Can K-Means detect anomalies?  
Yes, indirectly. Outliers far from centroids can be anomalies.  

### Q34. Is K-Means good for text data?  
Yes, after vectorization (TF-IDF, embeddings).  

### Q35. How does K-Means work for image compression?  
Clusters pixel colors into k representative colors.  

### Q36. What is curse of dimensionality in K-Means?  
Distances become less informative in high-dim space.  

### Q37. Is K-Means scalable?  
Yes, linear time complexity. Mini-batch makes it more scalable.  

### Q38. When should you not use K-Means?  
- Non-spherical clusters  
- Categorical-only data  
- Highly imbalanced data  

### Q39. Is K-Means deterministic with same initialization?  
Yes, results are fixed with same initialization and data order.  

### Q40. What is stopping criterion?  
Centroid shifts < tolerance or max iterations reached.  

---

## Practical (Q41–Q60)

### Q41. What libraries implement K-Means?  
- scikit-learn  
- Spark MLlib  
- TensorFlow (custom ops)  

### Q42. How does scikit-learn implement K-Means?  
Default: K-Means++ initialization, Lloyd’s algorithm, inertia for cost.  

### Q43. What is Lloyd’s algorithm?  
The iterative refinement procedure used by K-Means.  

### Q44. Difference between online and batch K-Means?  
- Batch: uses full dataset per iteration.  
- Online: updates incrementally with new data.  

### Q45. What is bisecting K-Means?  
Hierarchical approach: recursively split clusters with K-Means into subclusters.  

### Q46. What are centroid update rules?  
Centroid = mean of all points in cluster.  

### Q47. Can K-Means be used for dimensionality reduction?  
Not directly, but cluster assignments can act as features.  

### Q48. How does K-Means handle missing values?  
It cannot handle missing data natively. Imputation required.  

### Q49. What is the role of random seed?  
Ensures reproducibility in initialization.  

### Q50. How many iterations does K-Means need?  
Depends on data. Typically 10–300 iterations, converges quickly.  

### Q51. Is K-Means guaranteed to converge?  
Yes, but only to a local minimum.  

### Q52. What is the geometric interpretation of K-Means?  
It partitions space into Voronoi cells defined by centroids.  

### Q53. What is a Voronoi diagram?  
Partition of space where each region is closest to a centroid.  

### Q54. Can K-Means be parallelized?  
Yes, centroid updates and assignments are parallelizable.  

### Q55. How to handle different scales of data?  
Normalize or standardize features.  

### Q56. Does K-Means work with binary data?  
Not well. Better: K-Modes or clustering with Hamming distance.  

### Q57. What is the effect of noise in data?  
Noise reduces clustering accuracy, centroids may shift.  

### Q58. What are convergence issues?  
- Empty clusters  
- Oscillations in assignments  
- Poor initialization  

### Q59. How to avoid empty clusters?  
Reinitialize empty cluster with a distant point.  

### Q60. How to decide tolerance parameter?  
Choose based on required precision vs computation cost.  

---

## Advanced Applications (Q61–Q80)

### Q61. What is Kernel K-Means?  
Extension that applies kernel trick to capture non-linear boundaries.  

### Q62. What is Fuzzy C-Means?  
Soft clustering where points have degrees of membership in clusters.  

### Q63. Difference between K-Means and GMM?  
- K-Means: hard clustering, spherical assumption  
- GMM: probabilistic, allows ellipsoidal clusters  

### Q64. How does PCA help K-Means?  
Reduces dimensions, removes noise, makes clustering more effective.  

### Q65. What is spherical K-Means?  
Uses cosine similarity instead of Euclidean distance, good for text data.  

### Q66. What is spectral clustering vs K-Means?  
Spectral clustering uses eigen-decomposition of similarity graph, then applies K-Means on reduced space.  

### Q67. How does K-Means handle imbalance?  
Poorly, since large clusters dominate centroid computation.  

### Q68. Can K-Means be used for semi-supervised tasks?  
Yes, by seeding some labels and constraining assignments.  

### Q69. How does initialization affect results?  
Bad initialization → poor local minima. Good initialization → stable clusters.  

### Q70. What is an example of bad initialization?  
Centroids too close → collapse into same cluster.  

### Q71. What is trimmed K-Means?  
Variant that ignores a percentage of farthest points when computing centroids.  

### Q72. How is K-Means related to EM algorithm?  
Both are iterative optimization. K-Means is a special case of EM with hard assignments.  

### Q73. What is soft K-Means?  
Probabilistic assignment of points to clusters.  

### Q74. What is the difference between K-Means and DBSCAN?  
- K-Means: partitioning, requires k, not robust to outliers  
- DBSCAN: density-based, no k, robust to outliers  

### Q75. What is the effect of scaling k too high?  
Clusters become too small and meaningless.  

### Q76. What is global vs local minimum in K-Means?  
K-Means converges to local minima of cost function, not guaranteed global.  

### Q77. How can ensemble clustering help?  
Run K-Means multiple times, combine results (consensus clustering).  

### Q78. What is stability of clustering?  
Consistency of results under perturbations of data or parameters.  

### Q79. What is the role of cosine similarity in clustering?  
Helps cluster high-dimensional sparse vectors (e.g., text).  

### Q80. What is streaming K-Means?  
Handles continuously arriving data by updating centroids incrementally.  

---

## Expert-Level (Q81–Q100)

### Q81. What is scalable K-Means++?  
Initialization optimized for distributed/parallel systems.  

### Q82. Can K-Means handle mixed data types?  
Not directly. Use K-Prototypes.  

### Q83. What is the effect of correlated features?  
They distort distance calculations. PCA helps.  

### Q84. Can K-Means be used in recommendation systems?  
Yes, for user/item segmentation.  

### Q85. How is K-Means used in anomaly detection for networks?  
Clusters normal traffic, outliers flagged as anomalies.  

### Q86. What are limitations of K-Means?  
- Requires k  
- Sensitive to outliers  
- Assumes spherical clusters  
- Not for categorical data  

### Q87. What is initialization sensitivity?  
Different initial centroids → different clusters.  

### Q88. How to evaluate K-Means without labels?  
Silhouette, Davies-Bouldin, Calinski-Harabasz.  

### Q89. What is Calinski-Harabasz Index?  
Measures variance ratio between clusters. Higher = better.  

### Q90. What is a cluster centroid vs medoid?  
- Centroid = mean  
- Medoid = representative data point  

### Q91. What is the geometric shape of K-Means clusters?  
Convex (Voronoi cells).  

### Q92. Can K-Means be used in gene expression analysis?  
Yes, for grouping genes or samples with similar expression.  

### Q93. How is K-Means used in finance?  
- Customer segmentation  
- Fraud detection  
- Portfolio grouping  

### Q94. How does initialization affect runtime?  
Better initialization reduces iterations to converge.  

### Q95. What is weighted K-Means?  
Each data point has a weight, centroid computation accounts for it.  

### Q96. What is constraint-based K-Means?  
Incorporates constraints (must-link, cannot-link) during clustering.  

### Q97. Can K-Means work on graph data?  
Not directly. Graph embeddings used first, then K-Means.  

### Q98. What is centroid drift in streaming K-Means?  
Centroids change over time as new data arrives.  

### Q99. Can K-Means be used in deep learning?  
Yes, for feature learning, cluster assignments as pseudo-labels.  

### Q100. Summarize strengths and weaknesses of K-Means.  
**Strengths**: simple, fast, scalable, widely used.  
**Weaknesses**: needs k, sensitive to outliers, assumes spherical clusters, poor for categorical/mixed data.  

---
