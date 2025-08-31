# Logistic Regression – 100 Interview Questions & Answers

---

### Q1. What is logistic regression?  
**A:** Logistic regression is a classification algorithm used to predict the probability of a binary outcome (0/1). It uses the logistic (sigmoid) function to map predictions into the range [0,1].  

---

### Q2. How is logistic regression different from linear regression?  
**A:** Linear regression predicts continuous values, while logistic regression predicts probabilities for categorical outcomes. Logistic regression constrains output between 0 and 1 using a sigmoid function.  

---

### Q3. What are the types of logistic regression?  
**A:**  
- Binary logistic regression → for two classes (yes/no).  
- Multinomial logistic regression → for more than two unordered classes.  
- Ordinal logistic regression → for ordered categorical outcomes.  

---

### Q4. What is the mathematical form of logistic regression?  
**A:**  
\[
p = \frac{1}{1 + e^{-(β_0 + β_1x_1 + … + β_nx_n)}}
\]  
Where *p* is the probability of the outcome being 1.  

---

### Q5. Why do we use the sigmoid function in logistic regression?  
**A:** Because the sigmoid maps any real value to the (0,1) range, which is ideal for representing probabilities.  

---

### Q6. What does the output of logistic regression represent?  
**A:** It represents the probability that the input belongs to the positive class (usually labeled as 1).  

---

### Q7. How do you interpret logistic regression coefficients?  
**A:** Each coefficient represents the change in the log-odds of the outcome for a one-unit increase in the predictor, holding other predictors constant.  

---

### Q8. What is the log-odds in logistic regression?  
**A:** Log-odds (logit) = log(p / (1 – p)). Logistic regression models this as a linear function of predictors.  

---

### Q9. Why can’t logistic regression be solved with OLS?  
**A:** Because the relationship is non-linear, residuals are not normally distributed, and OLS would give values outside [0,1]. Instead, Maximum Likelihood Estimation (MLE) is used.  

---

### Q10. What is the cost function used in logistic regression?  
**A:** Logistic regression uses **log loss (binary cross-entropy)** as its cost function.  

---

### Q11. Why use log loss instead of MSE?  
**A:** MSE leads to non-convex optimization in logistic regression, making convergence harder. Log loss is convex and easier to optimize.  

---

### Q12. What is the decision boundary in logistic regression?  
**A:** The decision boundary is the threshold probability (commonly 0.5) used to classify an observation into one of the classes.  

---

### Q13. How do you choose the threshold value?  
**A:** By default, 0.5 is used. But it can be tuned using precision-recall trade-offs, ROC curves, or business-specific requirements.  

---

### Q14. What happens if you change the threshold?  
**A:** Lower threshold → higher recall but lower precision.  
Higher threshold → higher precision but lower recall.  

---

### Q15. What is the odds ratio in logistic regression?  
**A:** Odds ratio = e^(coefficient). It shows how odds of the outcome change with a one-unit increase in the predictor.  

---

### Q16. How do you interpret the odds ratio?  
**A:** If odds ratio = 2, odds of outcome double for a one-unit increase. If odds ratio < 1, odds decrease.  

---

### Q17. Can logistic regression handle non-linear relationships?  
**A:** Not directly. But you can add polynomial terms, interaction terms, or transformations to model non-linear effects.  

---

### Q18. How does logistic regression handle categorical variables?  
**A:** By encoding them (e.g., one-hot encoding). Each dummy variable acts as a binary predictor.  

---

### Q19. What is one-vs-rest (OvR) in logistic regression?  
**A:** For multiclass problems, OvR builds one logistic regression model per class, treating that class as positive and the rest as negative.  

---

### Q20. What is multinomial logistic regression?  
**A:** A logistic regression variant that directly models multiple classes without breaking them into binary subproblems.  

---

### Q21. How are logistic regression parameters estimated?  
**A:** Using **Maximum Likelihood Estimation (MLE)**, which finds coefficients that maximize the probability of observing the data.  

---

### Q22. Why is gradient descent used?  
**A:** Because logistic regression has no closed-form solution. Gradient descent optimizes the log-likelihood function iteratively.  

---

### Q23. What is maximum likelihood estimation (MLE)?  
**A:** MLE selects parameter values that maximize the likelihood of observing the given data.  

---

### Q24. How does MLE work in logistic regression?  
**A:** It calculates likelihoods across all data points, takes the log (log-likelihood), and then optimizes coefficients via iterative methods like gradient descent.  

---

### Q25. What is the log-likelihood function?  
**A:**  
\[
LL(β) = \sum [y_i \log(p_i) + (1-y_i)\log(1-p_i)]
\]  
It’s maximized to find the best-fitting coefficients.  

---

### Q26. What optimization algorithms are used in logistic regression?  
**A:** Gradient Descent, Stochastic Gradient Descent, Newton-Raphson, Iteratively Reweighted Least Squares (IRLS), and quasi-Newton methods like BFGS.  

---

### Q27. What is Newton-Raphson in logistic regression?  
**A:** An optimization method using second-order derivatives (Hessian matrix) to converge faster than gradient descent.  

---

### Q28. What is IRLS (Iteratively Reweighted Least Squares)?  
**A:** An algorithm used to estimate logistic regression parameters by repeatedly solving weighted least squares problems.  

---

### Q29. What assumptions does logistic regression make?  
**A:**  
- Linear relationship between predictors and log-odds.  
- Independent observations.  
- No or little multicollinearity.  
- Large sample size.  

---

### Q30. Does logistic regression require normality?  
**A:** No, predictors don’t need to be normally distributed.  

---

### Q31. Does logistic regression require homoscedasticity?  
**A:** No, unlike linear regression, equal variance of errors is not required.  

---

### Q32. How do you check the goodness of fit in logistic regression?  
**A:** Using measures like deviance, AIC, Hosmer-Lemeshow test, ROC-AUC, precision, recall, and F1-score.  

---

### Q33. What is deviance in logistic regression?  
**A:** Deviance = -2 × log-likelihood. Lower deviance means better model fit.  

---

### Q34. What is the Hosmer-Lemeshow test?  
**A:** A statistical test that checks how well predicted probabilities match actual outcomes.  

---

### Q35. What is pseudo-R² in logistic regression?  
**A:** Metrics like McFadden’s R² that provide a measure of model fit similar to R² in linear regression.  

---

### Q36. What is AIC in logistic regression?  
**A:** Akaike Information Criterion evaluates model quality by balancing goodness of fit and complexity. Lower is better.  

---

### Q37. What is BIC in logistic regression?  
**A:** Bayesian Information Criterion, similar to AIC but with stronger penalties for model complexity.  

---

### Q38. What evaluation metrics are used for logistic regression?  
**A:** Accuracy, Precision, Recall, F1-score, ROC-AUC, Log Loss, and Confusion Matrix.  

---

### Q39. Why not use accuracy alone?  
**A:** Accuracy can be misleading with imbalanced datasets. Precision, recall, and AUC are better.  

---

### Q40. What is ROC curve?  
**A:** Receiver Operating Characteristic curve plots True Positive Rate vs. False Positive Rate at different thresholds.  

---

### Q41. What is AUC?  
**A:** Area Under ROC Curve. AUC = 0.5 means random, AUC = 1 means perfect classifier.  

---

### Q42. What is Precision-Recall tradeoff?  
**A:** Increasing precision often decreases recall and vice versa. Threshold tuning balances them.  

---

### Q43. What is F1-score?  
**A:** Harmonic mean of precision and recall. Useful for imbalanced datasets.  

---

### Q44. How do you handle imbalanced datasets in logistic regression?  
**A:** Techniques include class weighting, oversampling (SMOTE), undersampling, and threshold adjustment.  

---

### Q45. Can logistic regression overfit?  
**A:** Yes, especially with many predictors. Regularization helps prevent it.  

---

### Q46. What regularization methods are used?  
**A:** L1 (Lasso), L2 (Ridge), and Elastic Net.  

---

### Q47. How does L1 regularization work?  
**A:** Adds absolute penalty (λΣ|β|), encourages sparsity, performs feature selection.  

---

### Q48. How does L2 regularization work?  
**A:** Adds squared penalty (λΣβ²), shrinks coefficients but doesn’t force them to zero.  

---

### Q49. What is Elastic Net regularization?  
**A:** Combination of L1 and L2 penalties, balancing feature selection and coefficient shrinkage.  

---

### Q50. What happens if λ is too high in regularization?  
**A:** Coefficients shrink too much, leading to underfitting.  

---

### Q51. How do you select λ (regularization parameter)?  
**A:** Using cross-validation to balance bias-variance tradeoff.  

---

### Q52. What is multicollinearity in logistic regression?  
**A:** High correlation between predictors, which inflates standard errors and destabilizes coefficients.  

---

### Q53. How do you detect multicollinearity?  
**A:** Using Variance Inflation Factor (VIF), correlation matrix, or condition number.  

---

### Q54. How do you handle multicollinearity?  
**A:** Drop variables, combine correlated features, or use regularization.  

---

### Q55. What happens if multicollinearity is ignored?  
**A:** Coefficient estimates become unreliable, making interpretation difficult.  

---

### Q56. Can logistic regression handle missing values?  
**A:** No, missing values must be handled before training (imputation or deletion).  

---

### Q57. How do you deal with missing values?  
**A:** Methods include mean/median imputation, kNN imputation, or modeling-based imputation.  

---

### Q58. What is feature scaling? Is it needed?  
**A:** Scaling puts predictors on similar ranges. Logistic regression benefits, especially when regularization is used.  

---

### Q59. Does logistic regression assume linearity of predictors?  
**A:** It assumes linearity with log-odds, not with probabilities.  

---

### Q60. How do you test linearity assumption?  
**A:** Use Box-Tidwell test or include interaction/polynomial terms to check relationships.  

---

### Q61. How do you check model performance visually?  
**A:** Using ROC curve, calibration curve, lift curve, or residual plots.  

---

### Q62. What are residuals in logistic regression?  
**A:** Difference between observed and predicted probabilities (not actual class labels).  

---

### Q63. What are deviance residuals?  
**A:** Measure how well the model fits each observation, based on deviance contribution.  

---

### Q64. What is leverage in logistic regression?  
**A:** Indicates the influence of a predictor on the fitted value. High leverage → outlier influence.  

---

### Q65. What are influential observations?  
**A:** Observations with high leverage and high residuals that significantly affect model estimates.  

---

### Q66. How do you detect influential observations?  
**A:** Using Cook’s Distance, leverage plots, and DFBetas.  

---

### Q67. What is separation in logistic regression?  
**A:** When a predictor perfectly classifies outcomes. It causes infinite coefficients.  

---

### Q68. How to handle separation?  
**A:** Use regularization, Bayesian priors, or penalized likelihood methods (Firth’s logistic regression).  

---

### Q69. What is multilevel logistic regression?  
**A:** Logistic regression that accounts for data hierarchy (e.g., patients within hospitals).  

---

### Q70. What is conditional logistic regression?  
**A:** Used for matched case-control studies, controlling for confounding variables.  

---

### Q71. What is penalized logistic regression?  
**A:** Logistic regression with penalty terms (L1, L2, Elastic Net) to prevent overfitting.  

---

### Q72. What is Bayesian logistic regression?  
**A:** Logistic regression using Bayesian priors and posterior inference instead of MLE.  

---

### Q73. What is ordinal logistic regression?  
**A:** Models ordered categorical outcomes (e.g., rating: poor, fair, good).  

---

### Q74. What is multinomial logistic regression?  
**A:** Extends logistic regression for more than two unordered outcome categories.  

---

### Q75. What are interaction terms?  
**A:** Terms that capture combined effects of two or more predictors on the outcome.  

---

### Q76. Why include interaction terms?  
**A:** To model cases where effect of one predictor depends on another.  

---

### Q77. What is feature selection in logistic regression?  
**A:** Choosing relevant predictors using methods like stepwise selection, LASSO, or information criteria (AIC/BIC).  

---

### Q78. What is forward selection?  
**A:** Start with no predictors, add variables one by one based on significance.  

---

### Q79. What is backward elimination?  
**A:** Start with all predictors, remove least significant ones step by step.  

---

### Q80. What is stepwise selection?  
**A:** Combination of forward and backward selection.  

---

### Q81. What are confounding variables?  
**A:** Variables that influence both predictor and outcome, causing biased estimates.  

---

### Q82. How to handle confounding?  
**A:** Include confounders in the model, stratification, or matching techniques.  

---

### Q83. What is overdispersion in logistic regression?  
**A:** When variance of data exceeds model assumptions. Common in count data, less relevant in binary outcomes.  

---

### Q84. What is quasi-binomial logistic regression?  
**A:** An extension that accounts for overdispersion by adjusting variance estimates.  

---

### Q85. Can logistic regression be used for time series?  
**A:** Yes, with lagged predictors, but specialized models (like survival analysis) are better.  

---

### Q86. Can logistic regression be used for survival data?  
**A:** Not directly. Survival analysis (Cox regression) is more appropriate.  

---

### Q87. How do you extend logistic regression to handle repeated measures?  
**A:** Use Generalized Estimating Equations (GEE) or Mixed-Effects Logistic Regression.  

---

### Q88. What is a calibration curve?  
**A:** Plot of predicted probabilities vs. observed frequencies. Checks probability calibration.  

---

### Q89. What is model calibration?  
**A:** Adjusting predicted probabilities so they match real-world frequencies.  

---

### Q90. How do you check calibration?  
**A:** Using Brier score or calibration plots.  

---

### Q91. What is the Brier score?  
**A:** Mean squared difference between predicted probability and actual outcome. Lower is better.  

---

### Q92. What are common pitfalls in logistic regression?  
**A:** Multicollinearity, separation, ignoring non-linear effects, overfitting, and poor threshold selection.  

---

### Q93. What are advantages of logistic regression?  
**A:** Simple, interpretable, requires fewer computations, works well with linearly separable data.  

---

### Q94. What are disadvantages of logistic regression?  
**A:** Assumes linear log-odds, sensitive to outliers, struggles with high-dimensional and non-linear data.  

---

### Q95. When should you not use logistic regression?  
**A:** When dataset is small, highly non-linear, or requires complex decision boundaries.  

---

### Q96. What are alternatives to logistic regression?  
**A:** Decision trees, Random Forests, SVMs, Gradient Boosting, and Neural Networks.  

---

### Q97. Why is logistic regression still widely used?  
**A:** High interpretability, efficiency, and strong baseline performance in classification tasks.  

---

### Q98. How do you explain logistic regression to a non-technical person?  
**A:** It’s like calculating the probability of something happening (e.g., passing an exam) based on input factors (study hours, attendance).  

---

### Q99. How do you implement logistic regression in Python?  
**A:** Using libraries like scikit-learn (`LogisticRegression`), statsmodels (`Logit`), or TensorFlow/Keras for custom models.  

---

### Q100. Give real-world applications of logistic regression.  
**A:**  
- Medical diagnosis (disease vs. no disease).  
- Credit scoring (default vs. non-default).  
- Marketing (customer response prediction).  
- Spam detection (spam vs. not spam).  

---
