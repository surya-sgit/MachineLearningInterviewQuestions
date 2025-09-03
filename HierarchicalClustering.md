# Hierarchical Clustering – 100 Interview Questions & Answers

---

### Q1. What is hierarchical clustering?  
A clustering method that builds a hierarchy of clusters either by progressively merging smaller clusters (agglomerative) or splitting larger clusters (divisive).

---

### Q2. What are the two main types of hierarchical clustering?  
1. **Agglomerative (bottom-up)** – start with single points, merge clusters iteratively.  
2. **Divisive (top-down)** – start with all points in one cluster, recursively split.

---

### Q3. Which is more commonly used: agglomerative or divisive?  
Agglomerative clustering is more common because it’s computationally simpler and widely implemented.

---

### Q4. What is a dendrogram?  
A tree-like diagram that records the sequence of merges or splits in hierarchical clustering.

---

### Q5. How do you interpret a dendrogram?  
By cutting the dendrogram at a specific height, you determine the number of clusters.

---

### Q6. What is the difference between hierarchical and K-Means clustering?  
- **K-Means**: requires predefined k, optimizes centroids, works well on large datasets.  
- **Hierarchical**: no need to predefine k, builds a dendrogram, works better on small datasets.

---

### Q7. How do you decide the number of clusters in hierarchical clustering?  
By cutting the dendrogram at the point where the distance between merged clusters increases sharply (elbow method).

---

### Q8. What are linkage methods in hierarchical clustering?  
They define how distances between clusters are computed:  
- Single linkage  
- Complete linkage  
- Average linkage  
- Ward’s method

---

### Q9. What is single linkage?  
The distance between two clusters is the shortest distance between any two points in the clusters.

---

### Q10. What is complete linkage?  
The distance between two clusters is the longest distance between any two points in the clusters.

---

### Q11. What is average linkage?  
The distance between clusters is the average distance between all pairs of points in the two clusters.

---

### Q12. What is Ward’s method?  
It merges clusters that result in the minimum increase in total within-cluster variance.

---

### Q13. Which linkage method is most commonly used?  
Ward’s method, as it tends to produce more compact, spherical clusters.

---

### Q14. What is the time complexity of agglomerative clustering?  
Typically **O(n³)** for naive implementations, but optimized versions can reduce this.

---

### Q15. What is the space complexity of hierarchical clustering?  
It usually requires **O(n²)** space to store the distance matrix.

---

### Q16. Can hierarchical clustering handle large datasets efficiently?  
Not well, because of its high computational cost compared to K-Means.

---

### Q17. What are the advantages of hierarchical clustering?  
- No need to predefine the number of clusters.  
- Produces a dendrogram for visualization.  
- Captures nested clusters.  

---

### Q18. What are the disadvantages of hierarchical clustering?  
- Computationally expensive.  
- Sensitive to noise and outliers.  
- Once merged/split, clusters cannot be undone.

---

### Q19. What distance metrics can be used in hierarchical clustering?  
- Euclidean distance  
- Manhattan distance  
- Cosine similarity  
- Correlation distance  

---

### Q20. Which libraries in Python support hierarchical clustering?  
- `scipy.cluster.hierarchy`  
- `sklearn.cluster.AgglomerativeClustering`  

---

### Q21. How do you plot a dendrogram in Python?  
Using `scipy.cluster.hierarchy.dendrogram` after linkage computation.

---

### Q22. What is cophenetic correlation coefficient?  
It measures how faithfully a dendrogram preserves pairwise distances between original data points.

---

### Q23. What happens if you cut a dendrogram at different heights?  
You get different numbers of clusters; lower cuts → more clusters, higher cuts → fewer clusters.

---

### Q24. Can hierarchical clustering find non-spherical clusters?  
Yes, depending on the linkage method and distance metric used.

---

### Q25. What is the role of distance matrix in hierarchical clustering?  
It is the foundation; hierarchical clustering starts with pairwise distances and builds from there.

---

### Q26. What’s the main drawback of divisive hierarchical clustering?  
It is computationally more complex and rarely implemented in practice.

---

### Q27. Is hierarchical clustering deterministic?  
Yes, given the same data, distance metric, and linkage method, the output is deterministic.

---

### Q28. What are ultrametric distances?  
Distances that satisfy the ultrametric inequality, commonly arising from hierarchical clustering dendrograms.

---

### Q29. How to choose between linkage methods?  
- **Single linkage**: detects elongated clusters.  
- **Complete linkage**: compact, round clusters.  
- **Average linkage**: balance of both.  
- **Ward’s method**: variance-minimizing clusters.

---

### Q30. What are nested clusters?  
Clusters within clusters, which hierarchical clustering naturally represents through dendrograms.

---

### Q31. Why is hierarchical clustering sensitive to outliers?  
Outliers can create individual branches in the dendrogram, affecting cluster interpretation.

---

### Q32. What is the role of Z-matrix in hierarchical clustering?  
It stores information about merges (clusters, distance, number of observations).

---

### Q33. What is the elbow method in hierarchical clustering?  
A method to choose the number of clusters by looking at the “elbow” in the dendrogram or distance plot.

---

### Q34. Can hierarchical clustering work with categorical data?  
Yes, but requires appropriate distance measures (e.g., Hamming distance, Gower distance).

---

### Q35. What is the difference between flat clustering and hierarchical clustering?  
- **Flat clustering**: assigns each point to one of k clusters.  
- **Hierarchical clustering**: builds a hierarchy/tree of clusters.

---

### Q36. What is the Lance–Williams formula?  
A recursive formula used to update distances in hierarchical clustering.

---

### Q37. Can hierarchical clustering be parallelized?  
Partially, but it is harder compared to K-Means due to dependencies in merging.

---

### Q38. How does noise affect hierarchical clustering?  
It can distort dendrogram structures and lead to spurious small clusters.

---

### Q39. What’s the difference between monothetic and polythetic clustering?  
- **Monothetic**: splits based on one attribute.  
- **Polythetic**: uses multiple attributes simultaneously (like hierarchical clustering).

---

### Q40. When should you prefer hierarchical clustering over K-Means?  
- Small datasets.  
- Unknown number of clusters.  
- When a dendrogram/tree is useful for analysis.

---

### Q41. How do you evaluate hierarchical clustering performance?  
Metrics include silhouette score, Davies–Bouldin index, cophenetic correlation, adjusted Rand index.

---

### Q42. What is a limitation of using Euclidean distance in hierarchical clustering?  
It assumes isotropic clusters and is sensitive to scale of features.

---

### Q43. What preprocessing steps are important for hierarchical clustering?  
- Feature scaling  
- Removing noise/outliers  
- Choosing appropriate distance metric  

---

### Q44. What is truncation in dendrogram plotting?  
Displaying only part of the dendrogram (e.g., last p merges) to simplify visualization.

---

### Q45. What is a cluster hierarchy?  
A multi-level clustering structure where clusters are nested within larger clusters.

---

### Q46. Can hierarchical clustering be used for document clustering?  
Yes, by computing similarity between TF-IDF representations.

---

### Q47. What is the difference between agglomerative clustering in sklearn vs scipy?  
- `scipy`: provides more flexibility in dendrogram visualization.  
- `sklearn`: integrates with ML pipelines, but doesn’t directly plot dendrograms.

---

### Q48. What is the chaining phenomenon in single linkage?  
When clusters are formed as long chains due to merging nearest neighbors one by one.

---

### Q49. Which linkage method is least affected by chaining?  
Complete linkage, since it uses the farthest distance between clusters.

---

### Q50. What’s the main advantage of complete linkage?  
It tends to produce compact, equally sized clusters.

---

### Q51. What’s the drawback of complete linkage?  
Sensitive to outliers because it uses the farthest pair.

---

### Q52. Which linkage is best for noisy datasets?  
Average linkage or Ward’s method.

---

### Q53. What is hierarchical density-based clustering?  
A variant that merges density-based clusters hierarchically, like HDBSCAN.

---

### Q54. Can hierarchical clustering be combined with other clustering methods?  
Yes, hybrid approaches exist (e.g., hierarchical clustering to estimate k for K-Means).

---

### Q55. What are ultrametric trees?  
Hierarchical trees where distances satisfy ultrametric inequality, similar to dendrograms.

---

### Q56. What are tie situations in hierarchical clustering?  
When two or more pairs of clusters have the same distance; resolved by tie-breaking rules.

---

### Q57. How does feature scaling affect hierarchical clustering?  
Unscaled features can dominate distance metrics, leading to biased clusters.

---

### Q58. What is the Silhouette Score?  
A metric to measure cluster quality by comparing intra-cluster and inter-cluster distances.

---

### Q59. Can hierarchical clustering handle non-linear boundaries?  
Yes, especially with appropriate distance metrics, though not as well as DBSCAN.

---

### Q60. What is a hybrid hierarchical-KMeans approach?  
Use hierarchical clustering to estimate number of clusters, then refine with K-Means.

---

### Q61. What is the computational bottleneck in hierarchical clustering?  
Distance matrix computation and updates during cluster merging.

---

### Q62. How do you prune a dendrogram?  
Cut it at a height threshold to define flat clusters.

---

### Q63. What’s the role of `AgglomerativeClustering` in sklearn?  
Implements bottom-up hierarchical clustering with different linkage options.

---

### Q64. What is the impact of high dimensionality on hierarchical clustering?  
The curse of dimensionality makes distances less informative, degrading performance.

---

### Q65. What is centroid linkage?  
Distance between cluster centroids is used to merge clusters.

---

### Q66. How does Ward’s method differ from centroid linkage?  
Ward minimizes variance; centroid linkage minimizes distance between centroids.

---

### Q67. What’s the relationship between hierarchical clustering and phylogenetics?  
Hierarchical clustering is widely used to build evolutionary trees in biology.

---

### Q68. Can hierarchical clustering detect anomalies?  
Yes, outliers often appear as small, separate branches in dendrograms.

---

### Q69. What is the importance of initialization in hierarchical clustering?  
Unlike K-Means, it doesn’t require random initialization—deterministic process.

---

### Q70. What is the role of `linkage()` in scipy?  
It performs hierarchical clustering and generates a linkage matrix for dendrograms.

---

### Q71. What is fcluster in scipy?  
A function that extracts flat clusters from hierarchical clustering results.

---

### Q72. What is the effect of using cosine similarity?  
Helps when dealing with text/high-dimensional sparse data.

---

### Q73. What are chained vs complete merges?  
- **Chained**: sequential nearest neighbor merges.  
- **Complete**: merge based on max distances, avoiding chaining.

---

### Q74. What is the difference between dendrogram height and cluster distance?  
They represent the same concept: the dissimilarity at which clusters merge.

---

### Q75. How do you scale hierarchical clustering for big data?  
- Use approximate nearest neighbors  
- Apply mini-batch clustering first  
- Use BIRCH algorithm  

---

### Q76. What is the BIRCH algorithm?  
Balanced Iterative Reducing and Clustering using Hierarchies – a scalable hierarchical clustering method.

---

### Q77. How is BIRCH different from standard hierarchical clustering?  
It builds a clustering feature tree incrementally, making it suitable for large datasets.

---

### Q78. Can hierarchical clustering be applied to streaming data?  
Yes, with incremental algorithms like BIRCH.

---

### Q79. How does noise reduction help hierarchical clustering?  
Removes spurious small clusters and improves dendrogram clarity.

---

### Q80. What is centroid distance in hierarchical clustering?  
The Euclidean distance between cluster centroids.

---

### Q81. What is the difference between dendrogram cut and flat clustering output?  
Dendrogram cut: visual/manual.  
Flat clustering: programmatic extraction of cluster labels.

---

### Q82. What is the limitation of dendrograms with large datasets?  
They become cluttered and unreadable.

---

### Q83. What is the “curse of dimensionality” in hierarchical clustering?  
In high dimensions, distances become similar, making clustering unreliable.

---

### Q84. Can PCA help hierarchical clustering?  
Yes, reducing dimensionality before clustering improves performance and interpretability.

---

### Q85. What is hybrid hierarchical density clustering?  
Combines hierarchical structure with density-based clustering (e.g., HDBSCAN).

---

### Q86. What is monotonicity in hierarchical clustering?  
Cluster distances should not decrease as we move up the dendrogram.

---

### Q87. What is proximity matrix?  
Matrix of pairwise distances used in hierarchical clustering.

---

### Q88. How do you validate dendrogram quality?  
Using cophenetic correlation coefficient.

---

### Q89. What is the relationship between hierarchical clustering and taxonomy?  
It’s used to build taxonomic classifications (e.g., biology, linguistics).

---

### Q90. Can hierarchical clustering be used with time series?  
Yes, if you define appropriate distance measures (e.g., DTW distance).

---

### Q91. What’s the difference between hierarchical clustering and DBSCAN?  
- Hierarchical: builds tree structure, needs distance matrix.  
- DBSCAN: density-based, detects arbitrary shapes, good for large datasets.

---

### Q92. What is inconsistency coefficient in dendrograms?  
Measures variability of cluster distances; helps decide cut points.

---

### Q93. Can hierarchical clustering handle missing values?  
Not directly; requires imputation or distance metrics that handle missingness.

---

### Q94. How is hierarchical clustering used in gene expression analysis?  
To cluster genes/samples with similar expression profiles.

---

### Q95. What’s the effect of scaling features differently?  
Dominant features skew distance calculations; always scale features appropriately.

---

### Q96. What is the role of clustering tendency tests?  
They assess whether data has meaningful clusters before applying clustering.

---

### Q97. How to handle categorical + numerical data in hierarchical clustering?  
Use mixed-distance metrics like Gower distance.

---

### Q98. What’s the difference between Ward’s and average linkage?  
Ward: minimizes variance.  
Average linkage: minimizes average distances.

---

### Q99. What are applications of hierarchical clustering?  
- Market segmentation  
- Document clustering  
- Phylogenetics  
- Image segmentation  
- Social network analysis  

---

### Q100. Summarize the key advantages of hierarchical clustering.  
- No need to predefine cluster number.  
- Produces interpretable dendrograms.  
- Captures nested/natural hierarchies.  
- Works with various distance metrics.

---
