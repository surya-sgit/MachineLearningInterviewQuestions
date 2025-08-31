# Decision Trees – 100 Interview Questions & Answers

---

## Fundamentals

### Q1. What is a Decision Tree?  
A Decision Tree is a supervised machine learning algorithm used for classification and regression. It splits data into subsets based on feature values, creating a tree-like model of decisions. Nodes represent questions on features, branches represent answers, and leaves represent predictions.

---

### Q2. How does a Decision Tree make predictions?  
The model starts from the root node, applies feature-based conditions, follows branches according to input values, and ends at a leaf node which outputs either a class label (classification) or a numeric value (regression).

---

### Q3. What are the main components of a Decision Tree?  
- **Root Node** – represents the entire dataset.  
- **Decision Nodes** – internal nodes where splitting occurs.  
- **Leaf Nodes** – terminal nodes with outcomes.  
- **Branches** – edges connecting nodes.  

---

### Q4. Difference between Classification and Regression Trees?  
- **Classification Trees** predict categorical outcomes (e.g., spam vs. not spam).  
- **Regression Trees** predict continuous outcomes (e.g., house price).  

---

### Q5. What metrics are used for splitting in classification trees?  
- **Gini Impurity**  
- **Entropy / Information Gain**  
- **Misclassification Error**  

---

### Q6. What metrics are used in regression trees?  
- Mean Squared Error (MSE)  
- Mean Absolute Error (MAE)  
- Variance reduction  

---

### Q7. What is Gini Impurity?  
Probability of incorrect classification if randomly assigned:  
\[
Gini = 1 - \sum p_i^2
\]  

---

### Q8. What is Entropy?  
A measure of impurity:  
\[
Entropy = - \sum p_i \log_2(p_i)
\]  

---

### Q9. What is Information Gain?  
Reduction in entropy after splitting:  
\[
IG = Entropy(Parent) - \sum \frac{|child|}{|parent|} \cdot Entropy(child)
\]  

---

### Q10. What are common stopping criteria?  
- All samples in a node belong to one class.  
- No features left.  
- Max depth reached.  
- Minimum sample per split/leaf threshold.  

---

## Tree Growth & Complexity

### Q11. What is tree depth?  
Maximum number of splits from root to leaf. Deeper trees capture more detail but risk overfitting.

---

### Q12. What is pruning?  
Process of removing less useful branches to prevent overfitting.  

---

### Q13. Types of pruning?  
- **Pre-pruning (early stopping)** – stop tree growth early.  
- **Post-pruning** – grow full tree, then cut back.  

---

### Q14. Pre vs Post-pruning?  
- Pre-pruning is computationally efficient but risks underfitting.  
- Post-pruning yields better generalization but is more expensive.  

---

### Q15. Advantages of Decision Trees?  
- Simple and interpretable.  
- No need for scaling/normalization.  
- Handles categorical + numerical data.  
- Captures nonlinearities.  

---

### Q16. Disadvantages?  
- Prone to overfitting.  
- High variance (sensitive to data changes).  
- Biased towards features with more categories.  
- Lower accuracy vs ensembles.  

---

### Q17. Handling categorical features?  
Split categories into subsets (e.g., {Red} vs {Blue, Green}).  

---

### Q18. Handling continuous features?  
Use threshold splits (e.g., Age ≤ 30 vs Age > 30).  

---

### Q19. What is a greedy algorithm in Decision Trees?  
At each step, the best local split is chosen without considering future splits.  

---

### Q20. What is CART?  
CART (Classification and Regression Trees) uses Gini for classification and variance for regression, always produces binary splits.  

---

### Q21. Methods to prevent overfitting?  
- Limit max depth.  
- Min samples per leaf.  
- Min impurity decrease.  
- Pruning.  
- Use ensembles (Random Forests, Boosting).  

---

### Q22. Gini vs Entropy?  
- Gini is faster, favors dominant classes.  
- Entropy is more computationally expensive but more informative.  

---

### Q23. What is feature importance in Decision Trees?  
Calculated by total impurity reduction contributed by a feature across the tree.  

---

### Q24. How are missing values handled?  
- Imputation.  
- Dropping samples.  
- Surrogate splits.  

---

### Q25. What are surrogate splits?  
Alternative splits used when data for main feature is missing.  

---

### Q26. How is the best split selected?  
By choosing the split with maximum impurity reduction.  

---

### Q27. Computational complexity of building a Decision Tree?  
Approximately \(O(n \cdot m \cdot \log n)\), where n = samples, m = features.  

---

### Q28. What is ID3?  
Early tree algorithm using Information Gain. Supports categorical features only.  

---

### Q29. What is C4.5?  
Improved version of ID3, handles continuous features, uses Gain Ratio, supports pruning.  

---

### Q30. What is Gain Ratio?  
\[
GainRatio = \frac{InformationGain}{SplitInformation}
\]  
Reduces bias towards features with many categories.  

---

## Advanced Concepts

### Q31. What is CHAID?  
Chi-square Automatic Interaction Detection. Uses chi-square tests for splitting categorical data.  

---

### Q32. What is cost-complexity pruning?  
Balances complexity vs training error using parameter α:  
\[
R_\alpha(T) = R(T) + \alpha |T|
\]  

---

### Q33. What is a decision boundary in Decision Trees?  
A piecewise-constant boundary dividing feature space into regions with uniform predictions.  

---

### Q34. How does a Decision Tree handle outliers?  
Trees are robust since splits are based on thresholds, but extreme values may influence regression trees.  

---

### Q35. Can Decision Trees overfit?  
Yes, deep trees can memorize training data.  

---

### Q36. Can they underfit?  
Yes, shallow trees may miss patterns.  

---

### Q37. How do Decision Trees handle high-dimensional data?  
They may overfit or become computationally heavy. Feature selection or ensembles are preferred.  

---

### Q38. What is variance in Decision Trees?  
High variance arises when small data changes produce very different trees.  

---

### Q39. What is bias in Decision Trees?  
Bias occurs when trees are too shallow, failing to capture complexity.  

---

### Q40. How do ensembles help?  
Random Forests reduce variance; Boosting reduces bias.  

---

### Q41. What is a Random Forest?  
An ensemble of Decision Trees using bagging + feature randomness.  

---

### Q42. What is Gradient Boosted Trees?  
Sequential trees trained to fix errors of previous trees, using gradient descent.  

---

### Q43. How does AdaBoost work with trees?  
Assigns higher weights to misclassified samples, builds weak learners (stumps).  

---

### Q44. What are Decision Stumps?  
Single-split trees used in boosting.  

---

### Q45. What is the difference between bagging and boosting?  
- Bagging: parallel trees, reduces variance.  
- Boosting: sequential trees, reduces bias.  

---

### Q46. What is XGBoost?  
An efficient boosting library that uses gradient boosting with regularization.  

---

### Q47. What is LightGBM?  
Boosting algorithm optimized for large datasets, uses histogram-based splitting.  

---

### Q48. What is CatBoost?  
Boosting library optimized for categorical features.  

---

### Q49. What is interpretability in Decision Trees?  
Decision Trees are highly interpretable since decision paths can be visualized.  

---

### Q50. How to visualize a Decision Tree?  
- sklearn’s `plot_tree` or `export_graphviz`.  
- Tools like Graphviz.  

---

## Practical Applications

### Q51. Use cases of Decision Trees?  
- Credit scoring.  
- Fraud detection.  
- Medical diagnosis.  
- Recommendation systems.  
- Risk assessment.  

---

### Q52. Are Decision Trees suitable for image classification?  
Not directly, but can be used with feature-engineered datasets.  

---

### Q53. How do Decision Trees handle imbalanced classes?  
- Class weighting.  
- Balanced splitting criteria.  
- Sampling methods.  

---

### Q54. What is class weighting?  
Assigning higher weight to minority class in split criterion.  

---

### Q55. What is the effect of correlated features?  
Trees may repeatedly split on correlated features, reducing interpretability.  

---

### Q56. How do trees perform feature selection?  
They naturally perform selection by choosing only informative features for splitting.  

---

### Q57. How are categorical variables with many levels treated?  
Risk of overfitting, as many splits can be formed. Gain Ratio or one-hot encoding is often used.  

---

### Q58. How do missing values affect prediction?  
If surrogate splits are unavailable, missing values must be imputed.  

---

### Q59. Can Decision Trees be used for time series?  
Not directly, but lagged features and engineered variables allow their use.  

---

### Q60. What is model interpretability vs accuracy tradeoff?  
Decision Trees are interpretable but less accurate; ensembles sacrifice interpretability for accuracy.  

---

## Technical Depth

### Q61. What is impurity decrease?  
Amount by which impurity (Gini, Entropy) is reduced due to a split.  

---

### Q62. Why are Decision Trees biased toward features with many categories?  
More categories increase chance of perfect splits, artificially boosting information gain.  

---

### Q63. How to mitigate high-cardinality bias?  
Use **Gain Ratio** or regularization constraints.  

---

### Q64. What are monotonic constraints in trees?  
Constraints that enforce monotonic relationships between features and output (used in LightGBM, XGBoost).  

---

### Q65. What is multi-output Decision Tree?  
A tree that predicts multiple outputs simultaneously.  

---

### Q66. How are Decision Trees evaluated?  
- Accuracy, Precision, Recall, F1 (classification).  
- RMSE, MAE, R² (regression).  

---

### Q67. What is CART pruning using cross-validation?  
Select pruning level that minimizes error on validation data.  

---

### Q68. How does sample size affect Decision Trees?  
Small datasets → overfitting risk. Large datasets → more reliable splits.  

---

### Q69. What is recursive binary splitting?  
Process of splitting nodes recursively into two child nodes.  

---

### Q70. What is greedy splitting’s drawback?  
It may miss globally optimal solutions.  

---

### Q71. What is Minimum Description Length (MDL)?  
Criterion for model selection balancing tree complexity with accuracy.  

---

### Q72. What are hyperparameters in Decision Trees?  
- Max depth.  
- Min samples split/leaf.  
- Max features.  
- Criterion (gini, entropy, mse).  

---

### Q73. Difference between sklearn `DecisionTreeClassifier` and `DecisionTreeRegressor`?  
Classifier predicts discrete labels; Regressor predicts continuous values.  

---

### Q74. What is sklearn’s default split criterion?  
- Gini for classification.  
- MSE for regression.  

---

### Q75. How does `max_features` help?  
Limits number of features considered per split, reduces overfitting.  

---

### Q76. How does regularization help?  
Constraints like max depth, min samples prevent overly complex trees.  

---

### Q77. Why are trees unstable?  
Small data variations lead to very different splits (high variance).  

---

### Q78. What is the curse of dimensionality for trees?  
In high dimensions, splits become sparse, reducing reliability.  

---

### Q79. What is quantization in trees?  
Discretization of continuous features to speed up training (used in LightGBM).  

---

### Q80. How does binning help?  
Reduces computational complexity by grouping continuous values into bins.  

---

### Q81. What is oblique Decision Tree?  
Splits using linear combinations of features, not single features.  

---

### Q82. Difference between axis-aligned vs oblique trees?  
- Axis-aligned → simple threshold splits.  
- Oblique → hyperplane splits, more flexible.  

---

### Q83. What is random feature selection in Random Forests?  
Each split considers a random subset of features, adding diversity.  

---

### Q84. What is bootstrapping in Decision Trees?  
Sampling with replacement to build multiple trees in ensembles.  

---

### Q85. What is out-of-bag (OOB) error?  
Validation error estimated using samples not included in bootstrap sample.  

---

### Q86. What is ExtraTrees?  
Extremely Randomized Trees use random thresholds for splits to increase diversity.  

---

### Q87. Why are Decision Trees piecewise-constant models?  
Because predictions are constant within each region (leaf).  

---

### Q88. Can Decision Trees extrapolate?  
No, predictions are limited to values observed in training data.  

---

### Q89. How does feature scaling affect trees?  
It doesn’t matter, since splits are based on thresholds.  

---

### Q90. Do trees need normalization?  
No, unlike linear models or SVM.  

---

## Expert Level

### Q91. How does class imbalance affect impurity measures?  
Splits may favor majority class; weighted criteria can fix it.  

---

### Q92. What is the difference between Decision Trees and Rule-based classifiers?  
Trees are hierarchical, rule-based models are flat collections of if-else rules.  

---

### Q93. How do Decision Trees relate to Bayesian models?  
Both can represent conditional dependencies, but trees are deterministic, Bayesian models probabilistic.  

---

### Q94. What is model stability index (MSI) in trees?  
Metric used in credit scoring to measure stability of predictive variables.  

---

### Q95. Can Decision Trees model XOR problems?  
Yes, by splitting multiple times, unlike linear models.  

---

### Q96. What is interpretability tradeoff in ensembles?  
Single trees are interpretable, ensembles improve accuracy but reduce transparency.  

---

### Q97. Why are Decision Trees prone to fragmentation?  
As splits progress, data fragments into small subsets, increasing variance.  

---

### Q98. What is the relationship between Decision Trees and Random Forests?  
Random Forest = ensemble of many Decision Trees trained on bootstrapped data and random features.  

---

### Q99. What are practical limitations of Decision Trees?  
- Poor generalization alone.  
- Sensitive to noise.  
- Large memory requirements in deep trees.  

---

### Q100. When would you not use Decision Trees?  
- When data is very high-dimensional.  
- When smooth continuous predictions are needed.  
- When interpretability is not required but accuracy is critical (ensembles/NNs may be better).  

---

✅ End of **100 Detailed Decision Tree Interview Questions & Answers**
