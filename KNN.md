# K-Nearest Neighbors (KNN) – 100 Interview Questions & Answers

---

### Q1. What is KNN?  
**A:** K-Nearest Neighbors (KNN) is a non-parametric, instance-based supervised learning algorithm that classifies data points based on the majority class among the k closest neighbors in the feature space.  

---

### Q2. Is KNN supervised or unsupervised?  
**A:** KNN is a **supervised learning algorithm**, because it requires labeled training data for classification or regression.  

---

### Q3. How does KNN work?  
**A:**  
1. Choose the number of neighbors (k).  
2. Calculate the distance between the query point and all training points.  
3. Identify the k nearest neighbors.  
4. Assign the majority label (classification) or average (regression).  

---

### Q4. What is the role of “k” in KNN?  
**A:** The value of k determines the number of neighbors considered. A small k may cause overfitting (too sensitive to noise), while a large k may underfit (oversmooth decision boundaries).  

---

### Q5. How do you select the best value of k?  
**A:** Using cross-validation. A common heuristic is to use the square root of the number of training samples.  

---

### Q6. What happens if k = 1?  
**A:** The model becomes highly sensitive to noise, as classification is based only on the nearest neighbor.  

---

### Q7. What happens if k is very large?  
**A:** The algorithm becomes biased, leading to underfitting because distant points may dominate the classification.  

---

### Q8. Is KNN parametric or non-parametric?  
**A:** KNN is **non-parametric**, because it makes no assumptions about the underlying data distribution.  

---

### Q9. Is KNN a lazy learner or eager learner?  
**A:** KNN is a **lazy learner** because it does not learn a model during training. It stores the dataset and makes predictions only at query time.  

---

### Q10. What is the difference between KNN and k-means clustering?  
**A:**  
- KNN → supervised classification algorithm.  
- K-means → unsupervised clustering algorithm.  

---

### Q11. Can KNN be used for regression?  
**A:** Yes. In regression, KNN predicts the average (or weighted average) of the values of its k nearest neighbors.  

---

### Q12. What distance metrics are used in KNN?  
**A:**  
- Euclidean distance  
- Manhattan distance  
- Minkowski distance  
- Cosine similarity (for text or high-dimensional data)  

---

### Q13. Which distance metric is most common?  
**A:** **Euclidean distance** is the most commonly used for continuous features.  

---

### Q14. How does feature scaling affect KNN?  
**A:** Feature scaling is critical because KNN relies on distance. Features with larger ranges dominate if not scaled.  

---

### Q15. What preprocessing is required for KNN?  
**A:**  
- Normalization/Standardization  
- Handling missing values  
- Encoding categorical variables  

---

### Q16. What are the advantages of KNN?  
**A:**  
- Simple and intuitive  
- No training phase  
- Can adapt to complex decision boundaries  

---

### Q17. What are the disadvantages of KNN?  
**A:**  
- High computational cost at prediction time  
- Sensitive to irrelevant features and scaling  
- Poor performance with high-dimensional data (curse of dimensionality)  

---

### Q18. What is the curse of dimensionality in KNN?  
**A:** As dimensions increase, distances between points become less meaningful, reducing the effectiveness of KNN.  

---

### Q19. How to handle curse of dimensionality in KNN?  
**A:** Use dimensionality reduction techniques such as PCA, feature selection, or autoencoders.  

---

### Q20. What is the time complexity of KNN?  
**A:**  
- Training: O(1) (lazy learner).  
- Prediction: O(n × d), where n = number of training points, d = dimensions.  

---

### Q21. How can we speed up KNN?  
**A:** Use **KD-Trees**, **Ball Trees**, or **Approximate Nearest Neighbors** methods.  

---

### Q22. How does KNN handle categorical data?  
**A:** By using distance measures suitable for categorical features, like Hamming distance.  

---

### Q23. Can KNN work with imbalanced datasets?  
**A:** It struggles with imbalanced data because majority classes dominate. Solutions include resampling, SMOTE, or class-weighted distances.  

---

### Q24. What is weighted KNN?  
**A:** A variation where closer neighbors are given higher weights, typically inversely proportional to their distance.  

---

### Q25. How do you evaluate a KNN model?  
**A:** Using metrics like accuracy, precision, recall, F1-score, ROC-AUC for classification, and RMSE/MAE for regression.  

---

### Q26. What is the effect of noise in KNN?  
**A:** Noise can significantly affect predictions since decisions are based on local neighbors.  

---

### Q27. What are KD-Trees?  
**A:** KD-Trees are data structures that partition space for efficient nearest neighbor searches in lower dimensions.  

---

### Q28. What are Ball Trees?  
**A:** Ball Trees partition data points into hyperspheres, useful for higher-dimensional spaces.  

---

### Q29. Which is better: KD-Tree or Ball Tree?  
**A:** KD-Trees work well for low dimensions (<20), while Ball Trees perform better for higher dimensions.  

---

### Q30. Why is KNN called an instance-based algorithm?  
**A:** Because it stores and uses actual training examples (instances) for predictions rather than learning a parametric function.  

---

### Q31. What is the memory requirement for KNN?  
**A:** High, since it needs to store the entire dataset.  

---

### Q32. How does KNN handle missing values?  
**A:** Typically requires imputation. Alternatively, modified distance metrics can ignore missing attributes.  

---

### Q33. Can KNN be used for anomaly detection?  
**A:** Yes. Points far from neighbors can be flagged as anomalies.  

---

### Q34. What are some real-life applications of KNN?  
**A:**  
- Recommender systems  
- Handwriting recognition  
- Medical diagnosis  
- Credit scoring  

---

### Q35. How is KNN used in recommender systems?  
**A:** By finding users/items similar to the target and recommending based on nearest neighbors.  

---

### Q36. What happens if features are highly correlated in KNN?  
**A:** Correlated features distort distance calculation. Feature selection or PCA can help.  

---

### Q37. How do categorical and continuous variables affect KNN together?  
**A:** They require careful preprocessing. Categorical variables may need one-hot encoding or specific distance measures.  

---

### Q38. Can KNN be parallelized?  
**A:** Yes. Distance computations for neighbors can be distributed across multiple processors.  

---

### Q39. What happens if the dataset is huge in KNN?  
**A:** Prediction becomes very slow. Approximate Nearest Neighbor search or data reduction techniques are used.  

---

### Q40. Can KNN overfit?  
**A:** Yes, when k is too small (e.g., k=1).  

---

### Q41. Can KNN underfit?  
**A:** Yes, when k is too large, leading to oversmoothing.  

---

### Q42. What is the decision boundary in KNN?  
**A:** The regions in feature space assigned to different classes, determined by nearest neighbors.  

---

### Q43. How does KNN handle multi-class classification?  
**A:** It chooses the majority class among k neighbors.  

---

### Q44. How does class overlap affect KNN?  
**A:** Overlap reduces accuracy, as neighbors from different classes are close to each other.  

---

### Q45. What is the effect of irrelevant features in KNN?  
**A:** They distort distance calculations, leading to poor accuracy. Feature selection is important.  

---

### Q46. What is the role of cross-validation in KNN?  
**A:** Used to tune the value of k and distance metric.  

---

### Q47. Can KNN work online (streaming data)?  
**A:** No. It requires storing all data, making it inefficient for online updates.  

---

### Q48. How do you scale KNN for big data?  
**A:** Use ANN (Approximate Nearest Neighbor) methods, hashing techniques, or distributed computation frameworks.  

---

### Q49. Is KNN deterministic?  
**A:** Yes, unless there are ties in nearest neighbors, which may require tie-breaking rules.  

---

### Q50. How do you break ties in KNN?  
**A:** By choosing the smaller class index, weighted distances, or random selection.  

---

### Q51. What are some variants of KNN?  
**A:**  
- Weighted KNN  
- Radius-based neighbors  
- Condensed KNN (reduces training data size)  

---

### Q52. What is radius-based neighbors?  
**A:** Instead of k fixed neighbors, all points within a fixed radius are considered.  

---

### Q53. What is condensed KNN?  
**A:** A method that selects a subset of training data to reduce memory and computation while retaining accuracy.  

---

### Q54. What are approximate nearest neighbors (ANN)?  
**A:** Algorithms that trade exact accuracy for much faster neighbor searches, useful in large datasets.  

---

### Q55. Why does KNN have high variance?  
**A:** Because predictions can change drastically with small changes in training data.  

---

### Q56. How does KNN perform compared to SVM or logistic regression?  
**A:** KNN can be competitive on small, clean datasets but performs worse on high-dimensional or large datasets.  

---

### Q57. Can KNN be used for time-series forecasting?  
**A:** Indirectly, yes. By using sliding windows and treating it as a supervised problem.  

---

### Q58. What is the impact of feature weighting in KNN?  
**A:** Assigning higher weights to important features improves performance by reducing noise.  

---

### Q59. How is cosine similarity used in KNN?  
**A:** Often used for text data, where angles between vectors are more meaningful than Euclidean distances.  

---

### Q60. How do you evaluate KNN on imbalanced datasets?  
**A:** Use metrics like ROC-AUC, precision, recall, and F1-score rather than accuracy.  

---

### Q61. Can KNN be used for probability estimation?  
**A:** Yes. Probability of a class = fraction of neighbors belonging to that class.  

---

### Q62. What is the geometric intuition of KNN?  
**A:** It divides feature space into regions around training points, assigning labels based on local neighborhoods.  

---

### Q63. Can KNN handle missing class labels in training?  
**A:** No. Training requires fully labeled data.  

---

### Q64. What is the role of dimensionality reduction in KNN?  
**A:** Reduces noise, computation, and improves performance in high-dimensional spaces.  

---

### Q65. Can KNN handle noisy labels?  
**A:** Poorly. Small k values are heavily influenced by noisy points.  

---

### Q66. Is KNN affected by outliers?  
**A:** Yes. Outliers can distort classification, especially with small k.  

---

### Q67. How to reduce the effect of outliers in KNN?  
**A:** Use weighted distances, robust distance metrics, or preprocessing with outlier removal.  

---

### Q68. Can KNN work with sparse data?  
**A:** It struggles with sparsity. Cosine similarity works better in such cases.  

---

### Q69. What is the effect of dimensionality on KNN runtime?  
**A:** Runtime increases exponentially due to distance computation in many dimensions.  

---

### Q70. What is prototype selection in KNN?  
**A:** A technique to reduce training data by keeping only representative points.  

---

### Q71. What is prototype generation in KNN?  
**A:** Creating synthetic points to summarize training data for faster predictions.  

---

### Q72. What is edited KNN?  
**A:** Removes noisy or misclassified points from training data to improve accuracy.  

---

### Q73. What is reduced KNN?  
**A:** Combines condensation and editing to reduce dataset size while keeping performance.  

---

### Q74. Can KNN handle mixed data types?  
**A:** Yes, with suitable distance metrics that combine categorical and continuous features.  

---

### Q75. Can KNN be kernelized?  
**A:** Yes, kernel functions can redefine similarity in feature space.  

---

### Q76. How do you visualize KNN decision boundaries?  
**A:** By plotting classification regions in 2D or 3D space, often using matplotlib in Python.  

---

### Q77. What is the bias-variance tradeoff in KNN?  
**A:** Small k → low bias, high variance. Large k → high bias, low variance.  

---

### Q78. How does KNN perform on small datasets?  
**A:** Generally well, as long as data is clean and low-dimensional.  

---

### Q79. How does KNN perform on large datasets?  
**A:** Poorly, due to high computation cost and memory requirements.  

---

### Q80. What is hybrid KNN?  
**A:** Combines KNN with other algorithms like decision trees or SVMs to improve accuracy.  

---

### Q81. Can KNN be used with streaming data?  
**A:** Not efficiently, since it requires storing all past data.  

---

### Q82. What is locally weighted regression in context of KNN?  
**A:** A regression version of KNN where points closer to query have more weight.  

---

### Q83. How do you implement KNN in Python?  
**A:** Using scikit-learn’s `KNeighborsClassifier` or `KNeighborsRegressor`.  

---

### Q84. What is the role of sklearn’s n_neighbors parameter?  
**A:** It sets the value of k (number of neighbors).  

---

### Q85. What is the effect of weights parameter in sklearn KNN?  
**A:** It determines whether all neighbors are weighted equally (`uniform`) or closer ones have more weight (`distance`).  

---

### Q86. How does sklearn choose distance metric?  
**A:** Default is **Minkowski distance (p=2 → Euclidean)**, but can be customized.  

---

### Q87. What is the default algorithm used in sklearn KNN?  
**A:** Auto, which selects **ball tree, kd-tree, or brute force** depending on dataset size and dimension.  

---

### Q88. How do you tune hyperparameters in KNN?  
**A:** Use grid search or randomized search over values of k, distance metrics, and weights.  

---

### Q89. What is the effect of uneven feature scaling in KNN?  
**A:** Features with larger ranges dominate distance calculations, leading to biased predictions.  

---

### Q90. How to normalize data for KNN?  
**A:** Apply min-max normalization or standardization (z-score).  

---

### Q91. What is the prediction complexity of KNN?  
**A:** O(n × d) per prediction, where n = number of points, d = dimensions.  

---

### Q92. What is approximate nearest neighbor search?  
**A:** A faster alternative that finds approximate rather than exact nearest neighbors, reducing computation.  

---

### Q93. What is the drawback of approximate nearest neighbors?  
**A:** Accuracy may drop slightly since neighbors found are approximate.  

---

### Q94. What’s the role of PCA in KNN?  
**A:** PCA reduces dimensionality, improving runtime and reducing overfitting.  

---

### Q95. What is the main limitation of KNN?  
**A:** High computational and storage cost, especially with large datasets.  

---

### Q96. How does KNN compare to Naive Bayes?  
**A:** KNN is instance-based, requires no training; Naive Bayes assumes conditional independence and builds a probabilistic model.  

---

### Q97. How does KNN compare to Decision Trees?  
**A:** Decision Trees partition feature space explicitly, while KNN uses local distances. Trees train faster; KNN predicts slower.  

---

### Q98. How does KNN compare to SVM?  
**A:** SVM finds global decision boundaries, while KNN makes local decisions. SVM handles high dimensions better.  

---

### Q99. How do you handle ties in KNN regression?  
**A:** Average all tied values.  

---

### Q100. Summarize when to use KNN.  
**A:** Use KNN when:  
- Dataset is small to medium-sized.  
- Features are low-dimensional and well-scaled.  
- Decision boundaries are irregular.  
- Interpretability and simplicity are important.  

---
