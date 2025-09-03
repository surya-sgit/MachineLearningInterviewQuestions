# t-SNE (t-Distributed Stochastic Neighbor Embedding) - 100 Interview Questions & Answers

---

### Q1. What is t-SNE?
t-SNE (t-distributed Stochastic Neighbor Embedding) is a nonlinear dimensionality reduction technique mainly used for visualizing high-dimensional data in 2D or 3D.

---

### Q2. Who developed t-SNE?
Laurens van der Maaten and Geoffrey Hinton in 2008.

---

### Q3. What is the main use of t-SNE?
To visualize high-dimensional datasets by preserving local neighbor relationships in a lower-dimensional space.

---

### Q4. Is t-SNE supervised or unsupervised?
Unsupervised.

---

### Q5. How does t-SNE work at a high level?
It models pairwise similarities between points in high-dimensional space and tries to preserve those similarities in low-dimensional space using probability distributions.

---

### Q6. What is the role of perplexity in t-SNE?
Perplexity controls the balance between local and global aspects of the data; it’s roughly the number of neighbors considered for each point.

---

### Q7. What’s a typical range for perplexity?
Between 5 and 50.

---

### Q8. What is the cost function in t-SNE?
Kullback-Leibler (KL) divergence between high-dimensional and low-dimensional probability distributions.

---

### Q9. What distance metric does t-SNE use?
Euclidean distance is typically used to compute similarities.

---

### Q10. What distribution does t-SNE use in the low-dimensional space?
A Student’s t-distribution with one degree of freedom (Cauchy distribution).

---

### Q11. Why does t-SNE use t-distribution instead of Gaussian in low dimensions?
To prevent the crowding problem by allowing heavier tails.

---

### Q12. What is the crowding problem in t-SNE?
In low dimensions, it’s hard to preserve distances for both near and far points; Gaussian distributions cause points to crowd together.

---

### Q13. What optimization method does t-SNE use?
Gradient descent with momentum.

---

### Q14. What is early exaggeration in t-SNE?
An initial step where similarities are exaggerated to form clusters before refining.

---

### Q15. What happens if perplexity is too low?
Clusters may break apart, losing global structure.

---

### Q16. What happens if perplexity is too high?
Clusters may merge, losing local detail.

---

### Q17. What are advantages of t-SNE?
- Excellent visualization of high-dimensional data  
- Captures non-linear relationships  
- Creates meaningful clusters  

---

### Q18. What are disadvantages of t-SNE?
- Computationally expensive  
- Non-deterministic (different runs may give different results)  
- Poor at preserving global structure  

---

### Q19. Can t-SNE be used for large datasets?
It’s computationally expensive, but approximations (Barnes-Hut t-SNE, FFT-based t-SNE) make it feasible.

---

### Q20. What is Barnes-Hut t-SNE?
An approximation of t-SNE reducing complexity from O(N²) to O(N log N).

---

### Q21. What is openTSNE?
A fast, scalable Python library for t-SNE.

---

### Q22. Is t-SNE deterministic?
No, results vary due to random initialization.

---

### Q23. How can you make t-SNE results reproducible?
Set a random seed.

---

### Q24. What’s the difference between PCA and t-SNE?
- PCA: linear, preserves variance.  
- t-SNE: nonlinear, preserves local neighborhoods for visualization.  

---

### Q25. Can t-SNE be used for clustering?
No, but clusters in t-SNE plots can hint at natural groupings.

---

### Q26. Is t-SNE good for feature selection?
No, it’s used for visualization, not feature selection.

---

### Q27. Can t-SNE be used for classification?
No, but it can help visualize separability between classes.

---

### Q28. How does t-SNE handle noise?
It can be sensitive to noise and outliers.

---

### Q29. What preprocessing is recommended before t-SNE?
- Scale features  
- Optionally use PCA to reduce dimensions first  

---

### Q30. Why use PCA before t-SNE?
To denoise, speed up t-SNE, and remove redundant features.

---

### Q31. How many dimensions can t-SNE reduce to?
Typically 2 or 3 for visualization.

---

### Q32. Can t-SNE reduce to more than 3 dimensions?
Technically yes, but it’s not useful beyond visualization.

---

### Q33. What is the runtime complexity of naive t-SNE?
O(N²), where N is the number of points.

---

### Q34. How does Barnes-Hut t-SNE speed up computation?
It approximates far-away points’ contributions using quad-trees.

---

### Q35. What is parametric t-SNE?
A neural network learns the mapping to low-dimensional space.

---

### Q36. What is the role of learning rate in t-SNE?
Controls step size in gradient descent; too low or high may distort results.

---

### Q37. What is a good learning rate for t-SNE?
Typically between 100 and 1000.

---

### Q38. What happens if learning rate is too small?
Points get stuck in small clusters.

---

### Q39. What happens if learning rate is too large?
Points scatter randomly without structure.

---

### Q40. How does t-SNE scale with dataset size?
Poorly for large datasets; requires approximations.

---

### Q41. What is the initialization method in t-SNE?
- Random initialization  
- PCA initialization  

---

### Q42. Which initialization is better?
PCA initialization gives more stable results.

---

### Q43. Can t-SNE handle categorical variables?
Not directly; categorical data must be encoded numerically.

---

### Q44. How does t-SNE differ from UMAP?
- t-SNE: better local structure  
- UMAP: better global structure, faster  

---

### Q45. What is the role of momentum in t-SNE optimization?
Helps accelerate convergence and escape local minima.

---

### Q46. What is exaggeration factor in t-SNE?
A multiplier applied to similarities during early exaggeration.

---

### Q47. Can t-SNE handle streaming data?
Not natively; it’s a batch algorithm.

---

### Q48. What are limitations of interpreting t-SNE plots?
Distances and cluster sizes are not always meaningful globally.

---

### Q49. Can t-SNE show false clusters?
Yes, artifacts may appear due to parameter tuning.

---

### Q50. Should t-SNE be used before clustering?
No, clustering should be performed on original/high-dimensional features.

---

### Q51. Can t-SNE be used on text data?
Yes, after embeddings (e.g., word2vec, BERT).

---

### Q52. Can t-SNE be used on images?
Yes, after feature extraction (e.g., CNN embeddings).

---

### Q53. How does t-SNE compare to autoencoders?
- Autoencoders: nonlinear dimensionality reduction with reconstruction  
- t-SNE: visualization only  

---

### Q54. Is t-SNE scalable to millions of samples?
Not efficiently; UMAP or parametric t-SNE are better.

---

### Q55. What is a similarity matrix in t-SNE?
Matrix of pairwise conditional probabilities of points being neighbors.

---

### Q56. How is similarity computed in high dimensions?
Using Gaussian distributions.

---

### Q57. How is similarity computed in low dimensions?
Using t-distribution.

---

### Q58. What is joint probability in t-SNE?
Symmetrized pairwise similarity combining conditional probabilities.

---

### Q59. Why does t-SNE use KL divergence?
To minimize difference between high- and low-dimensional similarities.

---

### Q60. What optimizer is used in scikit-learn’s t-SNE?
Gradient descent with momentum.

---

### Q61. What’s the difference between `fit_transform` and `fit` in sklearn t-SNE?
- `fit_transform`: computes and transforms data.  
- `fit`: only learns structure.  

---

### Q62. Can t-SNE embed new data points after training?
Not directly; retraining is required unless using parametric t-SNE.

---

### Q63. How do you speed up t-SNE?
- Use PCA first  
- Reduce dataset size  
- Use Barnes-Hut or FFT-based approximations  

---

### Q64. How does perplexity relate to dataset size?
Larger datasets allow higher perplexity; small datasets require lower.

---

### Q65. Can t-SNE separate overlapping clusters?
Yes, if local structure is strong enough.

---

### Q66. What is multi-scale t-SNE?
Runs t-SNE with multiple perplexities for better structure preservation.

---

### Q67. Can t-SNE be used in reinforcement learning?
Yes, to visualize policy or state embeddings.

---

### Q68. How does dimensionality affect t-SNE runtime?
Higher dimensions increase pairwise distance computation cost.

---

### Q69. What is the role of normalization before t-SNE?
Ensures all features contribute equally.

---

### Q70. What is an embedding in t-SNE?
The low-dimensional coordinates for each data point.

---

### Q71. What is the shape of t-SNE output?
(N, d) where N = number of samples, d = target dimensions (2 or 3).

---

### Q72. How do you evaluate t-SNE results?
Qualitatively by visualization; no strict quantitative metric.

---

### Q73. Why can’t t-SNE be directly used for dimensionality reduction in ML models?
It distorts distances and doesn’t preserve global structure.

---

### Q74. What is the curse of dimensionality’s role in t-SNE?
It motivates nonlinear methods like t-SNE for visualization.

---

### Q75. How is t-SNE different from MDS?
- MDS: preserves global pairwise distances  
- t-SNE: preserves local neighborhoods  

---

### Q76. What is the KL divergence minimized in t-SNE?
KL(P||Q) where P = high-dimensional similarities, Q = low-dimensional similarities.

---

### Q77. Why doesn’t t-SNE minimize KL(Q||P)?
That would overemphasize global structure instead of local.

---

### Q78. What is the role of heavy tails in t-distribution?
Helps spread points apart in low dimensions.

---

### Q79. What is the computational bottleneck of t-SNE?
Pairwise similarity computation.

---

### Q80. Can GPU accelerate t-SNE?
Yes, libraries like cuML and openTSNE support GPU.

---

### Q81. Can t-SNE handle non-Euclidean distances?
Not directly, but kernelized variants exist.

---

### Q82. How do you tune t-SNE?
Adjust perplexity, learning rate, iterations, initialization.

---

### Q83. What is the role of iteration count in t-SNE?
More iterations refine embeddings; typical range: 500–2000.

---

### Q84. What happens if iteration count is too low?
Embeddings may not converge.

---

### Q85. What is the advantage of UMAP over t-SNE?
UMAP is faster, scalable, and preserves more global structure.

---

### Q86. What is semi-supervised t-SNE?
Guiding t-SNE with labels to improve separation.

---

### Q87. Can t-SNE be used for anomaly detection?
No, but anomalies may appear as isolated points in visualization.

---

### Q88. What is joint perplexity?
Combining multiple perplexities for stability.

---

### Q89. Can t-SNE embed streaming data?
No, it’s not incremental.

---

### Q90. What is t-SNE’s role in NLP?
Used to visualize embeddings like word2vec, GloVe, BERT.

---

### Q91. What is the effect of dimensionality before applying t-SNE?
Too many irrelevant dimensions slow t-SNE; PCA helps.

---

### Q92. What is the effect of scaling before t-SNE?
Without scaling, large-scale features dominate.

---

### Q93. What is the difference between global and local structure?
- Global: distances between clusters  
- Local: relationships within clusters  

---

### Q94. Does t-SNE preserve global structure?
No, it primarily preserves local structure.

---

### Q95. How does t-SNE handle imbalanced datasets?
It may exaggerate small clusters due to local focus.

---

### Q96. What is affinity matrix in t-SNE?
Matrix of conditional probabilities representing affinities.

---

### Q97. How does t-SNE perform dimensionality reduction differently than autoencoders?
- Autoencoders: learn reconstruction  
- t-SNE: only neighbor-preserving embedding  

---

### Q98. What is one disadvantage of t-SNE in reproducibility?
Results vary across runs due to random initialization.

---

### Q99. What is one disadvantage of t-SNE in scalability?
O(N²) cost limits large datasets.

---

### Q100. Summarize t-SNE in one line.
t-SNE is a nonlinear technique for visualizing high-dimensional data by preserving local neighbor relationships in 2D/3D.

---
