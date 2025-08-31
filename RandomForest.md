# Random Forest – 100 Interview Questions & Answers

---

### Q1. What is a Random Forest?
**A:** Random Forest is an ensemble learning algorithm that builds multiple decision trees and combines their predictions using majority voting (classification) or averaging (regression).

---

### Q2. Why is it called “Random” Forest?
**A:** Randomness comes from (1) bootstrapping samples for each tree (bagging) and (2) selecting a random subset of features at each split.

---

### Q3. How does Random Forest improve over a single Decision Tree?
**A:** It reduces overfitting, improves generalization, and is more robust by averaging predictions from multiple de-correlated trees.

---

### Q4. What type of algorithm is Random Forest?
**A:** It is an ensemble method based on bagging (Bootstrap Aggregating) of decision trees.

---

### Q5. Can Random Forest be used for both classification and regression?
**A:** Yes. For classification, it uses majority voting; for regression, it averages predictions.

---

### Q6. What is bootstrap sampling in Random Forest?
**A:** Sampling with replacement from the training data to create different subsets for building each tree.

---

### Q7. What is feature randomness in Random Forest?
**A:** At each split, only a random subset of features is considered, ensuring trees are de-correlated.

---

### Q8. How does Random Forest handle overfitting?
**A:** By averaging predictions from multiple trees, variance is reduced, preventing overfitting.

---

### Q9. What is bagging?
**A:** Bagging is training multiple models on different bootstrap samples and aggregating their predictions.

---

### Q10. Why does Random Forest work well with high-dimensional data?
**A:** Because feature randomness ensures not all trees rely on the same dominant features.

---

### Q11. How are predictions made in classification Random Forest?
**A:** By majority voting among all the trees’ predicted classes.

---

### Q12. How are predictions made in regression Random Forest?
**A:** By averaging the outputs of all trees.

---

### Q13. What are hyperparameters of Random Forest?
**A:** Common ones include: number of trees (`n_estimators`), max depth, min samples per split, max features, bootstrap options.

---

### Q14. What is `n_estimators` in Random Forest?
**A:** The number of trees to build. More trees generally improve performance but increase computation.

---

### Q15. What is `max_features` in Random Forest?
**A:** The number of features randomly considered at each split. Defaults: √n_features (classification), n_features (regression).

---

### Q16. What is `max_depth`?
**A:** Maximum allowed depth of a tree. Limiting it prevents overfitting.

---

### Q17. What is `min_samples_split`?
**A:** The minimum number of samples required to split a node.

---

### Q18. What is `min_samples_leaf`?
**A:** The minimum number of samples required to be at a leaf node.

---

### Q19. What is Out-of-Bag (OOB) error in Random Forest?
**A:** OOB error is the validation error computed using samples not included in the bootstrap sample of a tree.

---

### Q20. Why is OOB error useful?
**A:** It provides an unbiased estimate of model performance without requiring a separate validation set.

---

### Q21. How does Random Forest reduce variance?
**A:** By averaging predictions of many trees, variance decreases significantly.

---

### Q22. How does Random Forest reduce bias?
**A:** By combining multiple trees, it usually lowers bias compared to a single shallow tree, though deep trees may dominate.

---

### Q23. What is feature importance in Random Forest?
**A:** A score indicating how useful a feature is in reducing impurity across the forest.

---

### Q24. How is feature importance measured?
**A:** Commonly by Mean Decrease in Impurity (MDI) or Mean Decrease in Accuracy (MDA).

---

### Q25. What is Mean Decrease in Impurity (MDI)?
**A:** Measures how much each feature reduces impurity, averaged across trees.

---

### Q26. What is Mean Decrease in Accuracy (MDA)?
**A:** Measures drop in accuracy when a feature’s values are randomly permuted.

---

### Q27. What is the main advantage of Random Forest over Decision Trees?
**A:** Random Forest is more accurate and less prone to overfitting.

---

### Q28. What is the main disadvantage of Random Forest?
**A:** Less interpretable compared to a single decision tree.

---

### Q29. Can Random Forest handle missing values?
**A:** Yes, Random Forest can handle missing values relatively well, either through surrogate splits or imputation.

---

### Q30. How does Random Forest handle categorical features?
**A:** They are typically one-hot encoded, although some implementations can handle them directly.

---

### Q31. Is Random Forest a parametric or non-parametric algorithm?
**A:** Non-parametric, as it makes no assumptions about data distribution.

---

### Q32. What is the time complexity of training Random Forest?
**A:** O(n_estimators * n_samples * log(n_samples)) approximately.

---

### Q33. What is the time complexity of predicting with Random Forest?
**A:** O(n_estimators * tree_depth).

---

### Q34. How does Random Forest handle class imbalance?
**A:** By using class weights, resampling techniques, or balanced bootstrap samples.

---

### Q35. What is the difference between Bagging and Random Forest?
**A:** Random Forest adds feature randomness at each split in addition to bagging.

---

### Q36. What is the difference between Random Forest and Boosting?
**A:** Random Forest builds trees independently (parallel), Boosting builds trees sequentially with weights on errors.

---

### Q37. Can Random Forest overfit?
**A:** Yes, if the trees are too deep and `n_estimators` is too small, though it’s less likely than single trees.

---

### Q38. What happens if we increase the number of trees in Random Forest?
**A:** Variance decreases, model stabilizes, but computation time increases.

---

### Q39. What is the effect of increasing `max_depth`?
**A:** Trees become more complex, reducing bias but increasing overfitting risk.

---

### Q40. How does Random Forest handle noisy data?
**A:** It is robust to noise because multiple trees dilute the effect of noisy samples.

---

### Q41. What is the curse of dimensionality in Random Forest?
**A:** It can still suffer if dimensions are extremely high, but random feature selection helps mitigate it.

---

### Q42. How do you tune Random Forest?
**A:** By adjusting hyperparameters: `n_estimators`, `max_depth`, `min_samples_split`, `max_features`.

---

### Q43. What evaluation metrics are used for classification?
**A:** Accuracy, Precision, Recall, F1-score, AUC-ROC.

---

### Q44. What evaluation metrics are used for regression?
**A:** RMSE, MAE, R² score.

---

### Q45. Why is Random Forest resistant to overfitting?
**A:** Because averaging across trees reduces variance and smooths predictions.

---

### Q46. How does Random Forest handle correlated features?
**A:** Highly correlated features may be split across trees; importance might be diluted.

---

### Q47. Can Random Forest detect interactions between features?
**A:** Yes, because decision trees naturally capture feature interactions.

---

### Q48. What happens if you set `max_features` too high?
**A:** Trees become more correlated, reducing the benefit of ensemble diversity.

---

### Q49. What happens if you set `max_features` too low?
**A:** Trees may be weak, increasing bias.

---

### Q50. Is Random Forest suitable for online learning?
**A:** No, it is designed for batch learning.

---

### Q51. How does Random Forest compare to Gradient Boosting?
**A:** RF is parallel and faster to train, Boosting is sequential but often more accurate.

---

### Q52. Can Random Forest be parallelized?
**A:** Yes, each tree can be trained independently, making it highly parallelizable.

---

### Q53. How is Random Forest different from Extra Trees (Extremely Randomized Trees)?
**A:** Extra Trees choose split points randomly instead of optimizing impurity reduction.

---

### Q54. How do you handle overfitting in Random Forest?
**A:** Limit tree depth, increase `min_samples_split`, use fewer features per split.

---

### Q55. What is the difference between OOB error and test error?
**A:** OOB error is computed during training without a separate dataset, test error requires a validation set.

---

### Q56. How to interpret feature importance in Random Forest?
**A:** Higher scores mean the feature contributed more to reducing impurity or accuracy loss.

---

### Q57. Can Random Forest be used for feature selection?
**A:** Yes, by ranking features using importance scores.

---

### Q58. Is Random Forest affected by scaling?
**A:** No, scaling is not required.

---

### Q59. Can Random Forest handle outliers?
**A:** Yes, it is relatively robust to outliers.

---

### Q60. Does Random Forest require normalization?
**A:** No, because decision trees are scale-invariant.

---

### Q61. What is the main intuition behind Random Forest?
**A:** Combining many weak learners (trees) reduces variance and improves accuracy.

---

### Q62. Why is Random Forest less interpretable?
**A:** Because it aggregates hundreds of trees, making it difficult to visualize decision boundaries.

---

### Q63. Can Random Forest handle non-linear relationships?
**A:** Yes, decision trees can model non-linear patterns, and RF combines many of them.

---

### Q64. Is Random Forest sensitive to irrelevant features?
**A:** Less so than single trees, but performance may degrade if too many irrelevant features exist.

---

### Q65. What is the role of randomness in Random Forest?
**A:** Ensures diversity among trees, improving generalization.

---

### Q66. How does Random Forest behave with small datasets?
**A:** May overfit if too many trees are built; simpler models may perform better.

---

### Q67. How does Random Forest scale with large datasets?
**A:** It scales reasonably well but training can be computationally expensive.

---

### Q68. What is class weighting in Random Forest?
**A:** Assigning higher weights to minority classes to handle class imbalance.

---

### Q69. Can Random Forest be used for anomaly detection?
**A:** Yes, by using techniques like Isolation Forest (variant of RF).

---

### Q70. Can Random Forest be used for ranking problems?
**A:** Not directly, but can be adapted using feature importance or regression.

---

### Q71. What is the difference between Random Forest and BaggingClassifier?
**A:** Random Forest introduces extra feature randomness, BaggingClassifier does not.

---

### Q72. What is bootstrap aggregation’s benefit?
**A:** Reduces variance and improves model stability.

---

### Q73. What happens if bootstrap is set to False?
**A:** Each tree is trained on the full dataset instead of bootstrap samples.

---

### Q74. Can Random Forest be applied to time-series data?
**A:** Yes, but needs feature engineering as it doesn’t handle temporal order inherently.

---

### Q75. What is the bias-variance trade-off in Random Forest?
**A:** RF reduces variance significantly but may still have moderate bias.

---

### Q76. Why does increasing the number of trees not cause overfitting?
**A:** More trees stabilize predictions and reduce variance, not increase complexity.

---

### Q77. Can Random Forest handle sparse data?
**A:** Yes, but performance may degrade compared to boosting methods.

---

### Q78. How is Random Forest used in practice?
**A:** Classification, regression, feature selection, imputation, anomaly detection.

---

### Q79. What is proximity in Random Forest?
**A:** Measure of how often two samples land in the same leaf across trees.

---

### Q80. What is Random Forest Regression?
**A:** RF applied to predict continuous outcomes by averaging outputs of multiple regression trees.

---

### Q81. What is Random Forest Classification?
**A:** RF applied to categorical outputs using majority voting across classification trees.

---

### Q82. What’s the relationship between number of trees and variance?
**A:** More trees reduce variance but returns diminish after a point.

---

### Q83. How do you validate a Random Forest model?
**A:** Using cross-validation, OOB error, or a hold-out validation set.

---

### Q84. What if two features are highly correlated?
**A:** Feature importance may be split between them, making interpretation tricky.

---

### Q85. How do Random Forests handle skewed distributions?
**A:** They can handle them reasonably but class balancing may still be needed.

---

### Q86. How do you tune the number of trees?
**A:** Increase until OOB error/test error stabilizes.

---

### Q87. Why is Random Forest slower in prediction than Decision Trees?
**A:** Because predictions must traverse multiple trees instead of one.

---

### Q88. What happens if Random Forest is underfitting?
**A:** Increase number of trees, max depth, or number of features considered.

---

### Q89. What happens if Random Forest is overfitting?
**A:** Decrease tree depth, use fewer features per split, increase min samples per leaf.

---

### Q90. What is warm start in Random Forest?
**A:** Allows incremental addition of trees without retraining from scratch.

---

### Q91. What is the memory requirement of Random Forest?
**A:** High, as it stores multiple deep trees.

---

### Q92. How do you interpret Random Forest decision boundaries?
**A:** By aggregating multiple tree partitions; often results in smoother boundaries.

---

### Q93. Why is Random Forest popular?
**A:** Easy to use, robust, good accuracy without much hyperparameter tuning.

---

### Q94. What is the main weakness of Random Forest?
**A:** Lack of interpretability and slower inference for large forests.

---

### Q95. Can Random Forest be used for multi-label classification?
**A:** Yes, with adaptations or wrappers around the standard implementation.

---

### Q96. Can Random Forest extrapolate beyond training range in regression?
**A:** No, it predicts within the range of training values.

---

### Q97. What is the main assumption of Random Forest?
**A:** There is no strict assumption about distribution; it assumes diverse trees improve generalization.

---

### Q98. Why do Random Forests need randomness in both data and features?
**A:** To decorrelate trees, ensuring diversity and reducing variance.

---

### Q99. How do you visualize feature importance in Random Forest?
**A:** Using bar plots of importance scores (MDI or MDA).

---

### Q100. When should you not use Random Forest?
**A:** When interpretability, real-time predictions, or very high-dimensional sparse data are primary requirements.

---
