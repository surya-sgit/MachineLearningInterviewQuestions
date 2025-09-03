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

---

### Q43. How do you predict probabilities with XGBoost?

Use the scikit-learn wrapper `predict_proba()` or the native `predict()` on a `DMatrix` (with default `output_margin=False`) to get probabilistic outputs for classification.

```python
# sklearn wrapper
probs = model.predict_proba(X_test)

# native API
dtest = xgb.DMatrix(X_test)
probs = booster.predict(dtest)
````

---

### Q44. How do you save and load a trained XGBoost model?

Use `save_model()`/`load_model()` for the booster or sklearn wrapper; you can also use pickle for the sklearn wrapper.

```python
# Booster
booster.save_model("model.json")
booster = xgb.Booster()
booster.load_model("model.json")

# sklearn wrapper
model.save_model("xgb_sklearn.json")
model.load_model("xgb_sklearn.json")
```

---

### Q45. How does XGBoost handle imbalanced datasets?

Common strategies:

* Set `scale_pos_weight = (num_negative / num_positive)`.
* Use stratified CV and opt for metrics like AUC/F1.
* Resample (SMOTE/undersampling) or adjust thresholds.
* Use appropriate `eval_metric` and early stopping.

---

### Q46. How to perform cross-validation with XGBoost?

Use `xgb.cv()` (native) or `sklearn.model_selection.cross_val_score`. `xgb.cv()` supports `early_stopping_rounds`, custom metrics, and returns per-round metrics.

```python
cv_results = xgb.cv(params, dtrain, num_boost_round=500, nfold=5,
                    metrics='auc', early_stopping_rounds=50, as_pandas=True)
```

---

### Q47. Does XGBoost support sparse data?

Yes — XGBoost efficiently handles sparse matrices (CSR/CSC) and learns default directions for missing values, making it suitable for high-dimensional sparse features (e.g., text TF-IDF).

---

### Q48. What are common finance applications of XGBoost?

Credit scoring, fraud detection, default prediction, customer risk segmentation, algorithmic feature modeling, and churn/retention modeling.

---

### Q49. What are common healthcare applications of XGBoost?

Disease risk prediction, patient readmission/churn prediction, survival analysis (with engineered features), treatment outcome modeling, and medical imaging feature modeling (with tabular metadata).

---

### Q50. What are common marketing applications of XGBoost?

Customer churn prediction, campaign response scoring, propensity modeling, segmentation, CLTV estimation (with features), and A/B uplift modeling (with careful experiment features).

---

### Q51. How to prevent overfitting with XGBoost on small datasets?

* Reduce `max_depth`.
* Lower `n_estimators` or use early stopping.
* Increase `lambda`/`alpha` regularization.
* Use `subsample` and `colsample_bytree`.
* Use simpler model (`gblinear`) or strong cross-validation.

---

### Q52. What’s the difference between Logloss and Error metrics?

* **Logloss**: measures probability quality (penalizes confident mistakes). Good for probabilistic models.
* **Error** (or classification error): fraction of incorrect predictions (0/1). Less informative for probability calibration.

---

### Q53. Does XGBoost support multi-output regression?

Not natively. Typical approaches:

* Train one XGBoost model per target.
* Use `sklearn.multioutput.MultiOutputRegressor` wrapping XGBoost.
* For highly correlated outputs, consider other multi-output methods.

---

### Q54. How can you visualize XGBoost trees?

Use `xgb.plot_tree()` or export to Graphviz. For many trees, visualize feature importance or SHAP summaries instead.

```python
xgb.plot_tree(booster, num_trees=0)
xgb.to_graphviz(booster, num_trees=0)
```

---

### Q55. How should you interpret feature importance from XGBoost?

Importance types: `gain` (quality of splits), `cover` (samples affected), `weight` (frequency). Prefer SHAP for local/global consistent explanations; raw importance can be misleading (correlated features, different scales).

---

### Q56. Which hyperparameter tuning techniques work well for XGBoost?

Grid search, random search, Bayesian optimization (e.g., Optuna), and sequential searches (tune tree depth → sampling → learning rate → regularization). Use cross-validation with early stopping to evaluate candidates.

---

### Q57. How to handle categorical variables with XGBoost?

Options:

* One-hot encoding or ordinal encoding (classic).
* Use target encoding with CV-based smoothing.
* Use `enable_categorical=True` and integer-encoded categories in modern XGBoost versions (check your installed version).

---

### Q58. What are SHAP values and how are they used with XGBoost?

SHAP (SHapley Additive exPlanations) quantify each feature’s contribution to a single prediction. Use `shap.TreeExplainer(booster)` to compute per-instance explanations and aggregate importances and interactions for interpretability.

---

### Q59. How to enable GPU acceleration in XGBoost?

Set `tree_method='gpu_hist'` (and optionally `predictor='gpu_predictor'`) and ensure compatible CUDA drivers. GPU is helpful for large datasets/hyperparameter searches.

---

### Q60. What does `early_stopping_rounds` do?

Stops boosting when evaluation metric on validation set does not improve for the given number of rounds, preventing overfitting and speeding up training (also gives `best_iteration`).

---

### Q61. How to perform multi-class classification with XGBoost?

Set `objective='multi:softprob'` (probabilities) or `multi:softmax` (labels) and `num_class=<K>`. Use `predict_proba()` for softprob outputs.

---

### Q62. How do you combine XGBoost with cross-validation for parameter selection?

Run `xgb.cv()` with parameter sets, early stopping, and chosen metric; compare CV mean/variance across parameter sets; optionally wrap with sklearn CV utilities for pipeline integration.

---

### Q63. What’s the difference between `gain`, `cover`, and `weight` in feature importance?

* **Gain**: average improvement in loss from splits using the feature (most informative).
* **Cover**: average number of samples affected by splits with the feature.
* **Weight**: how many times the feature appears in splits.

---

### Q64. Which regularization parameters exist and when to use them?

* `lambda` (L2): smooths leaf weights; use to reduce variance.
* `alpha` (L1): drives sparse solutions; use for feature selection. Tune both with CV.

---

### Q65. How does XGBoost mitigate overfitting overall?

Via shrinkage (`eta`), tree constraints (`max_depth`, `min_child_weight`), subsampling (`subsample`, `colsample_*`), regularization (`alpha`, `lambda`), and early stopping.

---

### Q66. What is `min_child_weight` and when increase it?

Minimum sum of instance weights needed in a child. Increase to make algorithm more conservative (useful with noisy data or to prevent overfitting).

---

### Q67. What does `max_delta_step` do?

Constrains the maximum delta step for leaf weight updates; useful to stabilize training in logistic regression or extremely imbalanced scenarios.

---

### Q68. How does `scale_pos_weight` work and when to use it?

Set `scale_pos_weight = neg / pos` to help with imbalance in binary classification. It scales gradient/hessian for positive examples, biasing updates to account for class imbalance.

---

### Q69. What’s the difference between `binary:logistic` and regression objectives?

`binary:logistic` outputs probabilities for binary classification; regression objectives (e.g., `reg:squarederror`) predict continuous targets and optimize MSE or chosen loss.

---

### Q70. `n_estimators` vs `num_boost_round` — what’s the difference?

* `n_estimators`: parameter name used in sklearn wrapper (number of trees).
* `num_boost_round`: native API parameter controlling boosting rounds. They represent the same concept in different APIs.

---

### Q71. What is column sampling (`colsample_bytree` / `colsample_bylevel`) and why use it?

Randomly samples features per tree (`colsample_bytree`) or per level (`colsample_bylevel`) to reduce feature correlation, improve generalization, and speed training on high-dimensional data.

---

### Q72. How to implement a custom loss/objective in XGBoost?

Provide a Python function returning gradient and hessian arrays for predictions vs labels and pass it as `obj` in the training API. Ensure numerical stability and correct shapes.

---

### Q73. How do you diagnose underfitting vs overfitting with XGBoost?

Compare training and validation metrics:

* **Underfitting**: both errors high and similar → increase model capacity (`max_depth`, `n_estimators`) or decrease regularization.
* **Overfitting**: training error low, validation error high → increase regularization, reduce depth, use subsampling, or apply early stopping.

---

### Q74. What are the pros of using XGBoost?

Fast, scalable, strong predictive performance, native handling of missing/sparse data, flexible objectives, and built-in regularization.

---

### Q75. What are the cons or limitations of XGBoost?

Requires careful hyperparameter tuning, can overfit if misconfigured, less interpretable than simple models, and raw sequential data (time series, text) often requires feature engineering before use.

---

### Q76. How do you monitor training progress in XGBoost?

Provide `eval_set` with `(X_val, y_val)` in sklearn API or `watchlist` in native API. Metrics are printed per iteration, and can be plotted.

---

### Q77. What is the difference between `objective` and `eval_metric`?

* `objective`: defines the loss function to optimize (training target).
* `eval_metric`: defines the metric(s) to report during training and evaluate model performance.

---

### Q78. Can XGBoost perform ranking tasks?

Yes, with objectives like `rank:pairwise`, `rank:ndcg`, or `rank:map`. Useful in recommendation systems and search ranking.

---

### Q79. What is DMatrix in XGBoost?

An optimized internal data structure for training, supporting sparse input, missing values, weights, and efficient memory usage.

---

### Q80. What’s the advantage of using DMatrix over numpy arrays?

Better memory efficiency, faster computation, and additional features like per-instance weights and missing value handling.

---

### Q81. Can XGBoost handle missing values automatically?

Yes. During training, XGBoost learns the optimal direction to handle missing values for each split.

---

### Q82. How to tune `max_depth`?

Start with small values (3–6). Higher values increase model complexity and risk of overfitting. Tune with CV.

---

### Q83. What is the difference between `subsample` and `colsample_bytree`?

* `subsample`: row sampling (instances).
* `colsample_bytree`: column sampling (features).
  Both add randomness and prevent overfitting.

---

### Q84. How does learning rate (`eta`) affect the model?

Smaller `eta` → slower learning, requires more trees but better generalization. Larger `eta` → faster learning, higher risk of overfitting.

---

### Q85. What is `gamma` in XGBoost?

Minimum loss reduction required to make a split. Higher values make the algorithm more conservative.

---

### Q86. How does XGBoost differ from LightGBM?

* XGBoost: depth-wise growth, more stable but slower.
* LightGBM: leaf-wise growth, faster but may overfit.
* LightGBM handles categorical features natively better.

---

### Q87. How do you ensemble XGBoost with other models?

Methods: stacking (meta-learners), blending, bagging with different seeds, or combining with linear/NN models.

---

### Q88. Can XGBoost be used for time-series forecasting?

Yes, but it requires feature engineering (lags, rolling means, date parts). XGBoost doesn’t model temporal order directly.

---

### Q89. How to interpret `best_iteration` in XGBoost?

The boosting round at which validation metric achieved its best score (with early stopping). Use it to set the final number of trees.

---

### Q90. What’s the advantage of histogram-based tree building (`hist`)?

Faster training, less memory, supports larger datasets, and GPU acceleration.

---

### Q91. How does XGBoost compare to Random Forest?

* XGBoost: boosting (sequential, reduces bias).
* Random Forest: bagging (parallel, reduces variance).
  XGBoost usually outperforms but is more complex.

---

### Q92. What is the role of `base_score`?

Initial prediction score for all instances (before any trees). Default is 0.5 for classification.

---

### Q93. What is feature interaction constraint in XGBoost?

Allows specifying which features are allowed to interact, improving interpretability and enforcing domain constraints.

---

### Q94. How do monotonic constraints work in XGBoost?

Force the relationship between a feature and prediction to be monotonic (increasing or decreasing), useful for regulated domains like finance.

---

### Q95. How to use XGBoost in a scikit-learn pipeline?

```python
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

pipe = Pipeline([
    ("scaler", StandardScaler()),
    ("xgb", XGBClassifier())
])
```

---

### Q96. What is the difference between booster types?

* `gbtree`: tree-based boosting (default).
* `gblinear`: linear boosting (like regularized linear models).
* `dart`: dropout meets boosting (adds dropout to gbtree).

---

### Q97. How does `dart` booster differ?

`dart` randomly drops trees during boosting, improving generalization but increasing training variance.

---

### Q98. How to save feature importance from XGBoost?

```python
importance = model.get_booster().get_score(importance_type='gain')
```

Export as dict, JSON, or plot with `xgb.plot_importance`.

---

### Q99. What’s the benefit of using external memory in XGBoost?

Allows training on datasets larger than memory by using disk-based caching. Slower, but makes training possible.

---

### Q100. When should you avoid using XGBoost?

* Small datasets where linear/logistic regression is sufficient.
* Problems requiring sequential modeling (time series without engineered features).
* When interpretability is more critical than accuracy.

---

