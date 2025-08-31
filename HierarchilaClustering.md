# Hierarchical Clustering – 100 Interview Questions & Answers

---

## Basics (Q1–Q20)

### Q1. What is Hierarchical Clustering?  
Hierarchical Clustering is an **unsupervised learning algorithm** that builds a hierarchy of clusters either in a **bottom-up (agglomerative)** or **top-down (divisive)** manner. It doesn’t require the number of clusters upfront and produces a dendrogram for visualization.

### Q2. Difference between Agglomerative and Divisive clustering?  
- **Agglomerative (bottom-up):** Each point starts as its own cluster; clusters are merged iteratively.  
- **Divisive (top-down):** All points start in one cluster; splits are performed iteratively.  

### Q3. What is a dendrogram?  
A dendrogram is a tree-like diagram that shows cluster merges (agglomerative) or splits (divisive). The height of branches indicates distance or dissimilarity at which clusters are merged/split.  

### Q4. What is linkage in Hierarchical Clustering?  
Linkage defines how distance between clusters is computed. Common types:  
- **Single linkage:** Minimum distance between points in clusters.  
- **Complete linkage:** Maximum distance between points in clusters.  
- **Average linkage:** Average distance between all points.  
- **Ward’s method:** Minimizes variance increase when merging clusters.  

### Q5. What is Euclidean distance in hierarchical clustering?  
Euclidean distance is the straight-line distance between two points in n-dimensional space. It’s the most common distance metric in clustering.

### Q6. Can Hierarchical Clustering handle categorical data?  
Yes, using **Hamming distance** or **Gower’s distance** for categorical/mixed data.  

### Q7. Difference between Hierarchical Clustering and K-Means?  
- Hierarchical: No need to predefine k, creates dendrogram.  
- K-Means: Requires k upfront, partitions data iteratively.  

### Q8. What is the time complexity of Hierarchical Clustering?  
- Agglomerative: O(n³) in naive implementation, O(n² log n) with optimized algorithms.  
- Divisive: Exponential, less common for large datasets.  

### Q9. When to prefer Hierarchical Clustering over K-Means?  
- When you don’t know k in advance.  
- For small to medium datasets.  
- When dendrogram visualization is helpful.  

### Q10. How do you cut a dendrogram to form clusters?  
Draw a horizontal line across the dendrogram at desired distance threshold. Clusters are defined by connected branches below the line.  

### Q11. What is cophenetic correlation coefficient?  
It measures how faithfully a dendrogram preserves pairwise distances. Value close to 1 indicates good representation.  

### Q12. Difference between bottom-up and top-down approaches?  
- Bottom-up: start with points, merge iteratively (agglomerative).  
- Top-down: start with one cluster, split iteratively (divisive).  

### Q13. What is single linkage problem (chaining)?  
Single linkage may produce elongated clusters (“chains”) due to minimum distance criterion.  

### Q14. What is complete linkage advantage?  
Produces compact, spherical clusters; avoids chaining problem.  

### Q15. What is Ward’s method?  
Merges clusters that minimize increase in total within-cluster variance. Produces well-balanced clusters.  

### Q16. Can you use different distance metrics?  
Yes, e.g., Manhattan, Cosine, Correlation. Metric choice affects cluster structure.  

### Q17. What are dendrogram cutting criteria?  
- Fixed number of clusters  
- Distance threshold  
- Maximum cluster height  

### Q18. How to visualize dendrograms?  
Use libraries: `scipy.cluster.hierarchy` in Python or `hclust` in R. Visualize with matplotlib or seaborn.  

### Q19. How do you evaluate hierarchical clusters?  
- Cophenetic correlation coefficient  
- Silhouette score  
- Davies-Bouldin index  
- Comparing clusters with ground truth (if available)  

### Q20. Can hierarchical clustering scale to large datasets?  
Not efficiently; time and space complexity are high. Approximate methods or sampling may be needed.  

---

## Advanced Concepts (Q21–Q40)

### Q21. What is agglomerative clustering algorithm?  
Iterative merging of the closest clusters until all points are in one cluster or stopping criterion met.  

### Q22. What is divisive clustering algorithm?  
Iterative splitting of clusters starting from all points in one cluster. Less common due to high computational cost.  

### Q23. How to choose linkage method?  
Depends on cluster shape and size:  
- Single → elongated clusters  
- Complete → compact clusters  
- Average → balanced  
- Ward → minimizes variance  

### Q24. What is the difference between linkage and distance metric?  
Distance metric measures point-to-point dissimilarity. Linkage defines cluster-to-cluster distance.  

### Q25. Can hierarchical clustering handle high-dimensional data?  
Yes, but distances may lose meaning (curse of dimensionality). Dimensionality reduction (PCA, t-SNE) recommended.  

### Q26. How does noise affect hierarchical clustering?  
Outliers may merge late, producing singleton clusters or affecting cluster structure.  

### Q27. What is a cut-off threshold?  
A distance at which dendrogram is cut to produce clusters.  

### Q28. Difference between agglomerative and centroid linkage?  
- Agglomerative merges nearest points or clusters.  
- Centroid linkage merges based on centroid distances. Can cause inversions in dendrogram.  

### Q29. What are inversions in dendrogram?  
When a cluster merges at a lower height than its sub-clusters; happens with centroid linkage.  

### Q30. How to determine number of clusters without k?  
Use:  
- Dendrogram height cut  
- Gap statistic  
- Silhouette analysis  

### Q31. What is cophenetic distance?  
Distance at which two points are first joined in dendrogram. Used for cophenetic correlation.  

### Q32. What is clustering tendency?  
Whether data has inherent clusters. Measured by Hopkins statistic.  

### Q33. What is Hopkins statistic?  
A value between 0–1: closer to 0 → highly clusterable, closer to 0.5 → random, 1 → uniform.  

### Q34. How does hierarchical clustering compare to density-based clustering?  
- Hierarchical: tree-based, no k needed, high complexity  
- DBSCAN: density-based, handles noise, non-convex clusters  

### Q35. Can hierarchical clustering work with mixed data?  
Yes, using Gower distance or other similarity metrics for mixed features.  

### Q36. How to handle categorical data?  
Encode as numeric or use specialized distance metrics (e.g., Hamming).  

### Q37. Difference between hierarchical and partitional clustering?  
- Hierarchical: tree, no k needed, nested clusters  
- Partitional (K-Means): requires k, non-nested clusters  

### Q38. What is dynamic tree cut?  
Method to automatically detect clusters from dendrogram based on shape and height criteria.  

### Q39. What is silhouette score in hierarchical clustering?  
Same as general clustering: evaluates cohesion and separation of clusters formed.  

### Q40. Can hierarchical clustering handle streaming data?  
Not directly; incremental or online hierarchical methods required.  

---

## Practical Applications (Q41–Q60)

### Q41. Libraries for hierarchical clustering?  
- Python: `scipy.cluster.hierarchy`, `sklearn.cluster.AgglomerativeClustering`  
- R: `hclust`  

### Q42. How to plot dendrogram in Python?  
`from scipy.cluster.hierarchy import dendrogram, linkage` → `dendrogram(linkage_matrix)`  

### Q43. How to perform agglomerative clustering in scikit-learn?  
`AgglomerativeClustering(n_clusters=k, linkage='ward')`  

### Q44. Difference between Ward linkage and average linkage?  
Ward minimizes variance increase, average linkage uses mean pairwise distances.  

### Q45. How to preprocess data for hierarchical clustering?  
- Standardize/normalize features  
- Handle missing values  
- Optionally reduce dimensions  

### Q46. Can hierarchical clustering be used in customer segmentation?  
Yes, forms nested groups to identify sub-segments.  

### Q47. Can hierarchical clustering be used in bioinformatics?  
Yes, e.g., clustering genes or protein expression patterns.  

### Q48. Can hierarchical clustering be used in document clustering?  
Yes, after vectorizing text (TF-IDF or embeddings).  

### Q49. How to deal with outliers?  
Remove them or use robust distance metrics.  

### Q50. How to speed up computation?  
- Use optimized algorithms  
- Reduce dataset size  
- Use approximate methods  

### Q51. Difference between agglomerative clustering and K-Means in application?  
Agglomerative is better for unknown k and hierarchical patterns; K-Means is faster for large datasets.  

### Q52. Can hierarchical clustering handle non-numeric data?  
Yes, with similarity/distance measures suitable for categorical/mixed data.  

### Q53. What is a practical application in marketing?  
Segment customers by purchase behavior hierarchically.  

### Q54. How to visualize hierarchical clustering results?  
- Dendrograms  
- Heatmaps with clustered rows/columns  

### Q55. How to combine hierarchical clustering with PCA?  
Reduce dimensions first with PCA, then cluster for efficiency and better separation.  

### Q56. How to choose between agglomerative and divisive clustering?  
Agglomerative is simpler and more common. Divisive is computationally expensive, used rarely.  

### Q57. How does distance metric choice affect clusters?  
Different metrics lead to different merges/splits and cluster shapes.  

### Q58. Can hierarchical clustering find non-convex clusters?  
Sometimes, better than K-Means, but still limited by linkage choice.  

### Q59. What is the difference between hard and soft hierarchical clustering?  
Standard hierarchical is hard (points belong to one cluster), soft hierarchical methods assign probabilities.  

### Q60. How to validate clusters without labels?  
Use silhouette, Davies-Bouldin index, cophenetic correlation, or dendrogram inspection.  

---

## Advanced/Expert Level (Q61–Q100)

### Q61. What is ultrametric distance?  
Distance measure satisfying triangle inequality with equality for hierarchical structure. Used in dendrogram construction.  

### Q62. Can hierarchical clustering handle large datasets?  
Not efficiently; memory and time complexity grows quadratically/cubically. Use sampling or approximation.  

### Q63. What is the Lance-Williams formula?  
Recursive formula to update distances in agglomerative clustering for different linkages.  

### Q64. What is dynamic hierarchical clustering?  
Automatically chooses cut points and clusters without manual threshold.  

### Q65. What are limitations of hierarchical clustering?  
- High complexity  
- Sensitive to noise/outliers  
- Cannot undo merges/splits  
- Not scalable  

### Q66. Difference between agglomerative and graph-based clustering?  
Agglomerative merges clusters by distance; graph-based partitions data using connectivity.  

### Q67. How to combine hierarchical clustering with K-Means?  
Use hierarchical clustering to determine k, then refine clusters with K-Means.  

### Q68. What is ultrametric tree property?  
All leaves are equidistant from root. Common in evolutionary biology dendrograms.  

### Q69. Difference between hierarchical clustering and GMM?  
Hierarchical: deterministic merges, no probability.  
GMM: probabilistic, assumes Gaussian distribution.  

### Q70. Can hierarchical clustering be used in image segmentation?  
Yes, cluster pixel colors or features hierarchically for segmentation.  

### Q71. What is cluster dendrogram inversion?  
Merging at a lower height than previous merges due to centroid linkage.  

### Q72. How does hierarchical clustering differ from agglomerative clustering in practice?  
Agglomerative is a type of hierarchical clustering (bottom-up).  

### Q73. How to deal with empty clusters?  
Not an issue in hierarchical clustering; all points belong to some cluster until merge/split.  

### Q74. How does hierarchical clustering handle mixed numeric and categorical data?  
Use Gower distance or similar mixed-type similarity metrics.  

### Q75. Can hierarchical clustering be used in bioinformatics?  
Yes, for gene expression, protein families, evolutionary trees.  

### Q76. Can hierarchical clustering be used for anomaly detection?  
Yes, singleton clusters or small clusters may indicate anomalies.  

### Q77. How to choose between single, complete, average, or Ward linkage?  
- Single: elongated clusters  
- Complete: compact clusters  
- Average: balanced  
- Ward: minimizes variance  

### Q78. How does hierarchical clustering deal with cluster size imbalance?  
It merges based on distance; large clusters may absorb small ones. Sensitive to scaling.  

### Q79. Can hierarchical clustering be parallelized?  
Partial parallelization possible for distance computations. Full hierarchical process is sequential.  

### Q80. Difference between hard and soft dendrograms?  
Hard: each leaf in one cluster  
Soft: probabilistic membership  

### Q81. What is a height in dendrogram?  
Represents the distance or dissimilarity at which clusters are merged.  

### Q82. What is tree cut method?  
A way to define clusters by cutting dendrogram at a specified height or number of clusters.  

### Q83. How does hierarchical clustering relate to phylogenetics?  
Used to build evolutionary trees based on genetic similarity.  

### Q84. Can hierarchical clustering detect nested clusters?  
Yes, dendrogram shows hierarchy of nested clusters.  

### Q85. How to preprocess for hierarchical clustering?  
- Normalize data  
- Handle missing values  
- Optionally reduce dimensions  

### Q86. How to handle categorical variables?  
Encode or use distance measures suitable for categorical data.  

### Q87. How to compute distances for hierarchical clustering?  
Use Euclidean, Manhattan, Cosine, Correlation, Gower, or Hamming depending on data type.  

### Q88. How is clustering quality evaluated?  
Silhouette, cophenetic correlation, Davies-Bouldin index, dendrogram visualization.  

### Q89. Difference between hierarchical and partitional clustering?  
Hierarchical: nested, no k needed  
Partitional: flat clusters, k required  

### Q90. How to interpret dendrogram?  
Height → distance at merge  
Branch structure → hierarchy of clusters  

### Q91. What is agglomerative coefficient?  
Measures clustering strength; closer to 1 → strong cluster structure.  

### Q92. How does noise affect hierarchical clustering?  
Outliers may form singleton clusters, skew distances, or affect merge order.  

### Q93. Can hierarchical clustering be used in NLP?  
Yes, for document clustering, topic modeling, word embeddings grouping.  

### Q94. Can hierarchical clustering be combined with K-Means?  
Yes, to determine k, then refine clusters with K-Means.  

### Q95. What is adaptive hierarchical clustering?  
Automatically adjusts distance thresholds or stopping criteria based on data.  

### Q96. How to handle very large datasets?  
- Sampling  
- Mini-batch  
- Approximate nearest neighbors  

### Q97. Can hierarchical clustering detect anomalies?  
Yes, singleton or small clusters may indicate rare events.  

### Q98. Difference between hierarchical clustering and DBSCAN?  
Hierarchical: tree-based, no density assumptions  
DBSCAN: density-based, robust to noise, finds non-spherical clusters  

### Q99. Strengths of hierarchical clustering?  
- No need for k  
- Produces dendrogram  
- Handles nested clusters  
- Useful for small to medium datasets  

### Q100. Weaknesses of hierarchical clustering?  
- High time and space complexity  
- Sensitive to outliers  
- Not scalable for large datasets  
- Cannot undo merges or splits once done  

---
