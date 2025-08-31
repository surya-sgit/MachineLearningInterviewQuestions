# XGBoost – 100 Interview Questions & Answers

---

## Basics (Q1–Q20)

### Q1. What is XGBoost?  
XGBoost (Extreme Gradient Boosting) is a scalable and efficient gradient boosting framework used for supervised learning. It builds an ensemble of decision trees sequentially, where each new tree corrects errors from previous trees.

### Q2. Is XGBoost supervised or unsupervised?  
XGBoost is a supervised learning algorithm for regression, classification, and ranking tasks.

### Q3. How does XGBoost differ from regular Gradient Boosting?  
- Optimized for speed and memory.  
- Supports parallel tree building.  
- Includes regularization (L1, L2).  
- Handles missing values internally.  
- Pruning via `max_depth` and `min_child_weight`.

### Q4. What are the key components of XGBoost?  
- Decision trees (CART)  
- Gradient boosting framework  
- Regularization  
- Objective function  
- Learning rate (`eta`)

### Q5. What is the objective function in XGBoost?  
It combines training loss (e.g., log loss, RMSE) and regularization (L1/L2 on tree weights) to prevent overfitting.

### Q6. What is gradient boosting?  
An ensemble method where models are trained sequentially, each new model predicts the residual errors of prior models.

### Q7. Difference between XGBoost and AdaBoost?  
- AdaBoost adjusts weights of misclassified points.  
- XGBoost uses gradients of the loss function, supports regularization, and parallel computation.

### Q8. What is learning rate (`eta`) in XGBoost?  
Controls the contribution of each tree. Lower `eta` → slower learning but better generalization.

### Q9. What is `max_depth`?  
Maximum depth of a tree; larger depth → more expressive, may overfit.

### Q10. What is `min_child_weight`?  
Minimum sum of instance weights required in a child node. Prevents overfitting small partitions.

### Q11. What is `subsample` in XGBoost?  
Fraction of rows used per tree. Helps reduce overfitting.

### Q12. What is `colsample_bytree` / `colsample_bylevel`?  
Fraction of columns/features sampled per tree or per split. Reduces correlation and overfitting.

### Q13. What is `gamma` in XGBoost?  
Minimum loss reduction required to make a split. Higher gamma → more conservative trees.

### Q14. How does XGBoost handle missing values?  
Automatically learns the optimal path for missing values during training (sparsity-aware split finding).

### Q15. What is regularization in XGBoost?  
- L1 (`alpha`) and L2 (`lambda`) regularization on leaf weights.  
- Helps prevent overfitting.

### Q16. Difference between XGBoost and Random Forest?  
- XGBoost: sequential boosting, reduces bias.  
- Random Forest: parallel bagging, reduces variance.  
- XGBoost often more accurate but sensitive to hyperparameters.

### Q17. What are booster types in XGBoost?  
- `gbtree`: tree-based boosting.  
- `gblinear`: linear booster.  
- `dart`: dropout trees.

### Q18. What is DART in XGBoost?  
Dropout Additive Regression Trees; randomly drops trees during training to reduce overfitting.

### Q19. What is early stopping?  
Stops training if the validation metric does not improve for a specified number of rounds.

### Q20. What metrics are used with XGBoost?  
- Classification: logloss, error, AUC.  
- Regression: RMSE, MAE.  
- Ranking: NDCG, MAP.

---

## Advanced Concepts (Q21–Q40)

### Q21. Role of gradient and hessian in XGBoost?  
- Gradient: first derivative of loss → direction to reduce error.  
- Hessian: second derivative → curvature information for optimal split.

### Q22. How does XGBoost perform tree pruning?  
Uses `max_depth`, `min_child_weight`, and post-pruning to reduce branches with negative gain.

### Q23. What is objective function customization?  
Define a custom loss function by providing gradients and Hessians.

### Q24. How does XGBoost handle multi-class classification?  
Uses `softmax` objective; one tree per class internally.

### Q25. Difference between parallelization in XGBoost vs traditional GBM?  
XGBoost builds histograms in parallel, unlike sequential traditional GBM.

### Q26. What is `tree_method`?  
Specifies the algorithm for building trees: `auto`, `exact`, `approx`, `hist`, `gpu_hist`.

### Q27. How to avoid overfitting in XGBoost?  
- Reduce `max_depth`  
- Increase `min_child_weight`  
- Use subsample/colsample  
- Apply regularization  
- Early stopping

### Q28. How to tune hyperparameters?  
Grid search, random search, Bayesian optimization; focus on `max_depth`, `eta`, `subsample`, `colsample_bytree`, `n_estimators`.

### Q29. What is `n_estimators`?  
Number of boosting rounds/trees.

### Q30. Role of `scale_pos_weight`?  
Balances positive and negative classes in imbalanced datasets.

### Q31. Difference between `reg:squarederror` and `reg:linear`?  
`reg:squarederror` → standard regression (MSE). `reg:linear` is deprecated.

### Q32. What is `binary:logistic` objective?  
Binary classification, outputs probabilities (0–1).

### Q33. Difference between `gbtree` and `gblinear`?  
- `gbtree`: non-linear tree boosting  
- `gblinear`: linear boosting, faster, less expressive

### Q34. How does XGBoost calculate feature importance?  
- Gain: total loss reduction contribution.  
- Cover: number of samples affected.  
- Weight: frequency feature used in splits.

### Q35. How does XGBoost handle categorical variables?  
Requires one-hot or label encoding; newer versions support categorical splits internally.

### Q36. What is monotone constraint?  
Ensures predictions increase/decrease with specific features for interpretability.

### Q37. Depth-first vs breadth-first tree building?  
- Depth-first: grow one branch to leaf before next  
- Breadth-first: grow all nodes level by level (histogram method)

### Q38. Can XGBoost be used for ranking?  
Yes, with objectives like `rank:pairwise`, `rank:ndcg`, `rank:map`.

### Q39. Handling missing values during prediction?  
Uses the learned default direction at each split.

### Q40. XGBoost vs LightGBM?  
- LightGBM: leaf-wise growth, faster for large datasets  
- XGBoost: depth-wise growth, more robust for small/medium datasets

---

## Practical Applications (Q41–Q60)

### Q41. Libraries?  
Python: `xgboost`  
R: `xgboost`  
GPU support: `gpu_hist`

### Q42. Training example:  
```python
import xgboost as xgb
model = xgb.XGBClassifier()
model.fit(X_train, y_train)
