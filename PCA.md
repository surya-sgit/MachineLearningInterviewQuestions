# Principal Component Analysis (PCA) - 100 Interview Questions & Answers

---

### Q1. What is PCA?
Principal Component Analysis (PCA) is a dimensionality reduction technique that transforms correlated features into a smaller set of uncorrelated variables called principal components.

---

### Q2. Why do we use PCA?
To reduce dimensionality, remove redundancy, speed up computation, reduce noise, and visualize high-dimensional data.

---

### Q3. What are principal components?
Linear combinations of the original variables that maximize variance while being orthogonal to each other.

---

### Q4. What is dimensionality reduction?
Reducing the number of input variables while preserving important information.

---

### Q5. What are the steps in PCA?
1. Standardize data  
2. Compute covariance matrix  
3. Calculate eigenvalues & eigenvectors  
4. Select top k eigenvectors  
5. Transform data  

---

### Q6. Why do we standardize before PCA?
To ensure all features contribute equally, as PCA is sensitive to scale.

---

### Q7. What is a covariance matrix in PCA?
A square matrix capturing pairwise feature relationships (variance on diagonals, covariance off-diagonals).

---

### Q8. Why use eigenvalues in PCA?
Eigenvalues indicate the amount of variance explained by each principal component.

---

### Q9. Why use eigenvectors in PCA?
Eigenvectors define the directions of maximum variance (principal axes).

---

### Q10. What is the explained variance ratio?
The fraction of dataset variance captured by each principal component.

---

### Q11. How do you decide the number of components to keep?
- Cumulative explained variance (e.g., 95%)  
- Scree plot (elbow method)  

---

### Q12. What is a scree plot?
Graph of eigenvalues vs. component index to decide number of components.

---

### Q13. What is orthogonality in PCA?
Principal components are uncorrelated and orthogonal to each other.

---

### Q14. How does PCA handle correlated features?
It combines correlated features into fewer uncorrelated components.

---

### Q15. What type of data is suitable for PCA?
Continuous numerical data with correlations among features.

---

### Q16. Is PCA supervised or unsupervised?
Unsupervised — it doesn’t use labels.

---

### Q17. What are advantages of PCA?
- Reduces dimensionality  
- Removes multicollinearity  
- Speeds up algorithms  
- Useful for visualization  

---

### Q18. What are disadvantages of PCA?
- Hard to interpret transformed features  
- Assumes linear relationships  
- Sensitive to scaling and outliers  

---

### Q19. What happens if you don’t standardize data?
Features with larger scales dominate principal components.

---

### Q20. What is the relation between PCA and SVD?
PCA can be performed using Singular Value Decomposition (SVD).

---

### Q21. What is SVD?
A matrix factorization method decomposing data into U, Σ, V^T.

---

### Q22. Why is SVD preferred over eigen decomposition?
SVD is more stable and works even if covariance matrix is not full rank.

---

### Q23. What is whitening in PCA?
Scaling principal components so they have unit variance.

---

### Q24. What is kernel PCA?
A nonlinear extension of PCA using kernel functions for mapping data to higher dimensions.

---

### Q25. What is sparse PCA?
A PCA variant encouraging sparsity in loadings for interpretability.

---

### Q26. What is incremental PCA?
A memory-efficient PCA that processes data in mini-batches.

---

### Q27. What is probabilistic PCA?
A probabilistic model of PCA using Gaussian latent variables.

---

### Q28. How does PCA relate to factor analysis?
Both reduce dimensionality, but factor analysis models latent factors, PCA focuses on variance.

---

### Q29. What is t-SNE vs PCA?
t-SNE is nonlinear and better for visualization; PCA is linear and interpretable.

---

### Q30. What is LDA vs PCA?
- PCA: unsupervised, maximizes variance.  
- LDA: supervised, maximizes class separation.  

---

### Q31. Can PCA be used for feature selection?
Yes, by selecting top k components with highest variance.

---

### Q32. Can PCA handle categorical data?
Not directly, but categorical data can be encoded numerically first.

---

### Q33. How does PCA help with multicollinearity?
It converts correlated variables into orthogonal principal components.

---

### Q34. Does PCA reduce overfitting?
Yes, by removing redundant/noisy features.

---

### Q35. Can PCA be used for image compression?
Yes, by keeping top principal components of pixel data.

---

### Q36. How is PCA used in face recognition?
Eigenfaces approach uses PCA to represent faces as principal components.

---

### Q37. Can PCA be used in finance?
Yes, for risk factor modeling, portfolio analysis, and feature reduction.

---

### Q38. What is cumulative explained variance?
Sum of explained variance ratios up to a certain number of components.

---

### Q39. What if all components are kept in PCA?
No reduction occurs, transformation is equivalent to rotation.

---

### Q40. What is reconstruction error in PCA?
The loss of information when projecting data into reduced components.

---

### Q41. What is the effect of outliers on PCA?
Outliers can distort variance and shift principal components.

---

### Q42. What are loadings in PCA?
The coefficients of original variables in each principal component.

---

### Q43. How do you interpret loadings?
Higher magnitude = stronger influence of a feature on a component.

---

### Q44. What is the difference between scores and loadings?
- Scores: transformed data points.  
- Loadings: weights of features in components.  

---

### Q45. Can PCA be applied to non-Gaussian data?
Yes, but Gaussian-like distributions work better.

---

### Q46. What is robust PCA?
A PCA variant designed to handle outliers effectively.

---

### Q47. What is randomized PCA?
An approximation of PCA using random projections for large datasets.

---

### Q48. Is PCA unique?
Eigenvalues are unique, eigenvectors may differ in sign or rotation.

---

### Q49. Can PCA improve classification performance?
Yes, by reducing noise and redundancy before classification.

---

### Q50. What is dimensionality curse?
High-dimensional spaces make data sparse; PCA mitigates this by reducing dimensions.

---

### Q51. What is feature decorrelation in PCA?
Principal components are uncorrelated linear transformations of original features.

---

### Q52. What is maximum variance criterion?
PCA chooses directions that maximize variance in the data.

---

### Q53. What is Hotelling’s T-squared statistic?
A multivariate measure based on PCA scores to detect anomalies.

---

### Q54. How is PCA used in anomaly detection?
Low-dimensional projections highlight deviations from normal patterns.

---

### Q55. Can PCA be applied on sparse data?
Yes, but preprocessing like scaling may be required.

---

### Q56. What is the computational complexity of PCA?
O(n³) for eigen decomposition; SVD and approximations reduce cost.

---

### Q57. What is dual PCA?
Performing PCA in feature space when number of features > number of samples.

---

### Q58. What is a biplot in PCA?
Visualization of both data points and feature vectors in PCA space.

---

### Q59. What is eigenface method?
Using PCA on facial images to extract key features for recognition.

---

### Q60. What is PCA whitening in deep learning?
Preprocessing step that removes correlation between pixels.

---

### Q61. What is truncated SVD vs PCA?
Truncated SVD is PCA-like but works directly on sparse matrices.

---

### Q62. What is negative eigenvalue in PCA?
It indicates numerical instability or incorrect covariance matrix computation.

---

### Q63. What happens if features are independent?
PCA doesn’t help much since variance is already uncorrelated.

---

### Q64. What is an eigen decomposition requirement?
Covariance matrix must be symmetric and square.

---

### Q65. Can PCA be used for denoising?
Yes, by discarding low-variance components representing noise.

---

### Q66. What is correlation PCA?
PCA performed on correlation matrix instead of covariance matrix.

---

### Q67. What is Kaiser criterion?
Keep components with eigenvalues > 1.

---

### Q68. What is Bartlett’s test of sphericity?
A statistical test to check PCA suitability by testing correlations.

---

### Q69. What is KMO measure?
Kaiser-Meyer-Olkin measure tests adequacy for PCA (should be > 0.6).

---

### Q70. How do you interpret PCA plots?
By checking variance explained and direction of feature contributions.

---

### Q71. What is the relation between PCA and ICA?
- PCA maximizes variance.  
- ICA maximizes statistical independence.  

---

### Q72. What is non-linear PCA?
Extending PCA with kernels or neural networks to capture non-linearities.

---

### Q73. Can PCA fail?
Yes, if data has no correlation, non-linear structure, or strong outliers.

---

### Q74. What is autoencoder vs PCA?
Both reduce dimensions, but autoencoders can learn non-linear transformations.

---

### Q75. What is variance retention in PCA?
How much of the original variance is preserved after reduction.

---

### Q76. Can PCA be applied incrementally?
Yes, using incremental PCA for streaming/large data.

---

### Q77. What is scikit-learn’s PCA implementation?
`sklearn.decomposition.PCA` class with explained variance ratio output.

---

### Q78. What parameters are used in PCA (sklearn)?
`n_components`, `svd_solver`, `whiten`.

---

### Q79. What is randomized SVD in sklearn?
Fast approximation used for large datasets when `svd_solver='randomized'`.

---

### Q80. What happens if `n_components=None` in sklearn PCA?
All components are kept, no reduction.

---

### Q81. What is PCA in ML pipelines?
Used for preprocessing before classification, clustering, regression.

---

### Q82. Can PCA be used before clustering?
Yes, PCA improves clustering by removing noise and redundancy.

---

### Q83. How is PCA used in NLP?
On word embeddings to reduce dimensions.

---

### Q84. Can PCA be applied to time series?
Yes, by treating time points as features.

---

### Q85. What is dynamic PCA?
Time-series PCA that accounts for temporal correlations.

---

### Q86. What is PCA loading plot?
A graph showing feature contributions to each component.

---

### Q87. What is overfitting in PCA?
If too many components are retained, PCA may preserve noise.

---

### Q88. What is underfitting in PCA?
If too few components are kept, significant variance is lost.

---

### Q89. What is PCA rotation?
Orthogonal rotation of components for better interpretability.

---

### Q90. What is varimax rotation?
A method to make PCA loadings more interpretable by maximizing variance.

---

### Q91. What is oblique rotation?
Allows correlated components in rotated PCA.

---

### Q92. What is L1-PCA?
A robust PCA method using L1 norm instead of L2 norm.

---

### Q93. What is graph PCA?
Applying PCA on graph Laplacian for dimensionality reduction.

---

### Q94. What is sparse coding vs PCA?
Sparse coding learns dictionary basis functions, PCA uses orthogonal vectors.

---

### Q95. What is maximum likelihood PCA?
Probabilistic PCA estimation using maximum likelihood.

---

### Q96. What is PCA in feature engineering?
Used to create new, uncorrelated features for ML models.

---

### Q97. What is PCA in anomaly detection?
Outliers appear in low-variance regions of PCA space.

---

### Q98. What is PCA in recommender systems?
Used to reduce user-item interaction matrix dimensionality.

---

### Q99. What is PCA’s relation with noise?
Low-variance components often correspond to noise.

---

### Q100. Summarize PCA in one line.
PCA reduces dimensionality by projecting data onto uncorrelated components that capture maximum variance.

---
