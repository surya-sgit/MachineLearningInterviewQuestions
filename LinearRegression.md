# Linear Regression – 100 Interview Questions & Answers

---

### Q1. What is linear regression?  
**A:** Linear regression is a statistical technique used to model the relationship between a dependent variable and one or more independent variables using a linear equation.  

---

### Q2. What are the types of linear regression?  
**A:**  
- Simple Linear Regression (one independent variable)  
- Multiple Linear Regression (multiple independent variables)  

---

### Q3. What is the equation for simple linear regression?  
**A:**  
\[
Y = β_0 + β_1X + ε
\]  
Where Y is dependent variable, X is independent variable, β₀ is intercept, β₁ is slope, and ε is error term.  

---

### Q4. What does the slope coefficient represent?  
**A:** The slope represents the change in the dependent variable for a one-unit change in the independent variable.  

---

### Q5. What does the intercept represent?  
**A:** The intercept is the expected value of the dependent variable when all independent variables are zero.  

---

### Q6. What are the assumptions of linear regression?  
**A:**  
1. Linearity  
2. Independence of errors  
3. Homoscedasticity (constant variance of errors)  
4. Normal distribution of errors  
5. No multicollinearity  

---

### Q7. What is multicollinearity?  
**A:** Multicollinearity occurs when independent variables are highly correlated with each other, making it difficult to estimate coefficients reliably.  

---

### Q8. How do you detect multicollinearity?  
**A:** Using Variance Inflation Factor (VIF), correlation matrix, or condition index.  

---

### Q9. What is homoscedasticity?  
**A:** Homoscedasticity means that the variance of the errors is constant across all levels of the independent variables.  

---

### Q10. What happens if homoscedasticity is violated?  
**A:** Standard errors become unreliable, leading to incorrect p-values and confidence intervals.  

---

### Q11. How do you check linearity in regression?  
**A:** By plotting residuals vs fitted values, or using scatterplots of variables.  

---

### Q12. What is heteroscedasticity?  
**A:** When the variance of residuals is not constant across values of predictors.  

---

### Q13. How can heteroscedasticity be corrected?  
**A:** By transforming variables (log, square root), using weighted least squares, or robust standard errors.  

---

### Q14. What is the error term in regression?  
**A:** The error term represents the difference between observed values and predicted values.  

---

### Q15. What is R-squared?  
**A:** R² measures the proportion of variance in the dependent variable explained by the independent variables.  

---

### Q16. What is adjusted R-squared?  
**A:** Adjusted R² accounts for the number of predictors and penalizes adding irrelevant variables.  

---

### Q17. What is the difference between R-squared and adjusted R-squared?  
**A:** R² increases with more predictors, even if they are irrelevant. Adjusted R² only increases if predictors improve the model significantly.  

---

### Q18. What is the range of R-squared?  
**A:** Between 0 and 1. Higher values indicate better fit.  

---

### Q19. What is the significance of p-values in regression?  
**A:** P-values test whether a predictor variable significantly contributes to explaining the dependent variable.  

---

### Q20. What is the null hypothesis in regression coefficient testing?  
**A:** H₀: The coefficient = 0 (no effect).  

---

### Q21. How do you interpret a coefficient with p-value < 0.05?  
**A:** It means the predictor has a statistically significant relationship with the dependent variable.  

---

### Q22. What is Ordinary Least Squares (OLS)?  
**A:** OLS is the method used to estimate regression coefficients by minimizing the sum of squared residuals.  

---

### Q23. Why do we use OLS?  
**A:** OLS provides unbiased, efficient, and consistent estimates under regression assumptions.  

---

### Q24. What is residual in regression?  
**A:** Residual = observed value – predicted value.  

---

### Q25. Why do we square residuals in OLS?  
**A:** To avoid cancellation of positive and negative errors and to penalize larger errors more.  

---

### Q26. What are influential points in regression?  
**A:** Data points that disproportionately affect the regression model’s parameters.  

---

### Q27. How do you detect influential points?  
**A:** Using Cook’s distance, leverage, or DFBETAs.  

---

### Q28. What is leverage in regression?  
**A:** Leverage measures how far an observation’s predictor values are from the mean. High leverage points can affect regression line.  

---

### Q29. What is Cook’s distance?  
**A:** A metric that measures the influence of an observation on regression coefficients.  

---

### Q30. What is autocorrelation?  
**A:** When residuals are correlated with each other, common in time series data.  

---

### Q31. How do you detect autocorrelation?  
**A:** Using Durbin-Watson test or residual plots.  

---

### Q32. Why is autocorrelation a problem?  
**A:** It violates independence assumption and leads to biased standard errors.  

---

### Q33. How can autocorrelation be corrected?  
**A:** By using autoregressive models, adding lag variables, or generalized least squares.  

---

### Q34. What is endogeneity in regression?  
**A:** Endogeneity occurs when predictors are correlated with the error term, leading to biased estimates.  

---

### Q35. What causes endogeneity?  
**A:** Omitted variables, measurement error, or simultaneity.  

---

### Q36. How can endogeneity be addressed?  
**A:** Using instrumental variables, fixed effects models, or two-stage least squares.  

---

### Q37. What is the difference between correlation and regression?  
**A:** Correlation measures association; regression establishes predictive relationships and quantifies effect size.  

---

### Q38. What is the Gauss-Markov theorem?  
**A:** It states that OLS estimators are the Best Linear Unbiased Estimators (BLUE) under standard assumptions.  

---

### Q39. What does BLUE stand for?  
**A:** Best Linear Unbiased Estimator.  

---

### Q40. Why are OLS estimators unbiased?  
**A:** Because their expected value equals the true population parameter.  

---

### Q41. What is the F-test in regression?  
**A:** It tests whether the model as a whole is statistically significant.  

---

### Q42. When do you use the t-test in regression?  
**A:** To test the significance of individual regression coefficients.  

---

### Q43. What is overfitting in regression?  
**A:** When the model captures noise in the training data and performs poorly on unseen data.  

---

### Q44. How can overfitting be reduced?  
**A:** By removing irrelevant variables, using regularization (Lasso, Ridge), or cross-validation.  

---

### Q45. What is underfitting?  
**A:** When the model is too simple to capture the underlying relationship in the data.  

---

### Q46. How do you handle categorical variables in regression?  
**A:** By encoding them using dummy variables or one-hot encoding.  

---

### Q47. Why is dummy variable trap a problem?  
**A:** Because it introduces multicollinearity by including redundant variables.  

---

### Q48. How do you avoid dummy variable trap?  
**A:** By dropping one dummy variable from each category.  

---

### Q49. What is stepwise regression?  
**A:** A variable selection method that adds or removes predictors based on statistical significance.  

---

### Q50. What is regularization in regression?  
**A:** A technique to prevent overfitting by penalizing large coefficients (Lasso, Ridge, Elastic Net).  

---

### Q51. What is Lasso regression?  
**A:** Lasso adds L1 penalty to the cost function, shrinking some coefficients to zero, thus performing variable selection.  

---

### Q52. What is Ridge regression?  
**A:** Ridge adds L2 penalty to the cost function, shrinking coefficients but not forcing them to zero.  

---

### Q53. What is Elastic Net regression?  
**A:** Elastic Net combines L1 and L2 penalties, balancing variable selection and coefficient shrinkage.  

---

### Q54. How do you decide between Lasso and Ridge?  
**A:** Use Lasso when feature selection is important, Ridge when all predictors are expected to contribute.  

---

### Q55. What is cross-validation in regression?  
**A:** A technique to evaluate model performance by splitting data into training and validation sets multiple times.  

---

### Q56. What is K-fold cross-validation?  
**A:** Data is divided into K folds, model is trained on K-1 folds, validated on 1 fold, repeated K times.  

---

### Q57. What is the difference between training error and test error?  
**A:** Training error is calculated on training data; test error on unseen data. Test error reflects generalization ability.  

---

### Q58. Why is test error more important than training error?  
**A:** Because it indicates how well the model generalizes to unseen data.  

---

### Q59. What is variance in regression models?  
**A:** Variance measures how much model predictions change with different training datasets.  

---

### Q60. What is bias in regression models?  
**A:** Bias is the error introduced by approximating a real-world problem with a simplified model.  

---

### Q61. What is the bias-variance tradeoff?  
**A:** A balance between bias (underfitting) and variance (overfitting) to achieve optimal model performance.  

---

### Q62. What is adjusted R-squared used for in multiple regression?  
**A:** To evaluate model fit while accounting for the number of predictors.  

---

### Q63. Why is adjusted R-squared better than R-squared in multiple regression?  
**A:** Because it penalizes irrelevant predictors, avoiding misleading improvements in R².  

---

### Q64. What is the difference between simple and multiple regression?  
**A:** Simple regression uses one predictor, multiple regression uses more than one predictor.  

---

### Q65. Can linear regression be used for classification?  
**A:** No, but logistic regression (a variant) is used for classification tasks.  

---

### Q66. What is polynomial regression?  
**A:** A form of regression where independent variables are raised to powers to capture non-linear relationships.  

---

### Q67. Is polynomial regression still linear?  
**A:** Yes, because coefficients are estimated linearly. The term “linear” refers to parameters, not variables.  

---

### Q68. What is interaction term in regression?  
**A:** A term representing the combined effect of two predictors on the dependent variable.  

---

### Q69. How do you check normality of residuals?  
**A:** Using Q-Q plots, histograms, or statistical tests like Shapiro-Wilk.  

---

### Q70. Why is normality of residuals important?  
**A:** For valid hypothesis testing and confidence intervals in regression.  

---

### Q71. What if residuals are not normally distributed?  
**A:** Use transformations, bootstrapping, or robust regression techniques.  

---

### Q72. What is robust regression?  
**A:** A regression method that reduces the influence of outliers.  

---

### Q73. What is quantile regression?  
**A:** A regression technique that estimates conditional quantiles (like median) instead of mean.  

---

### Q74. When would you use quantile regression?  
**A:** When the relationship differs across different points of the distribution.  

---

### Q75. What is weighted least squares regression?  
**A:** A method that gives different weights to observations, useful when heteroscedasticity is present.  

---

### Q76. How do you detect outliers in regression?  
**A:** Using studentized residuals, leverage values, or Cook’s distance.  

---

### Q77. What is the impact of outliers on regression?  
**A:** They can disproportionately affect coefficients and reduce model accuracy.  

---

### Q78. How can outliers be handled?  
**A:** By removing, transforming, or using robust regression techniques.  

---

### Q79. What is the difference between parametric and non-parametric regression?  
**A:** Parametric assumes a fixed functional form (e.g., linear regression), non-parametric makes fewer assumptions (e.g., regression trees).  

---

### Q80. Why is linear regression called a parametric model?  
**A:** Because it estimates a finite set of parameters (coefficients).  

---

### Q81. What is mean squared error (MSE)?  
**A:** Average of squared differences between observed and predicted values.  

---

### Q82. What is root mean squared error (RMSE)?  
**A:** Square root of MSE, providing error in the same units as the dependent variable.  

---

### Q83. What is mean absolute error (MAE)?  
**A:** Average of absolute differences between observed and predicted values.  

---

### Q84. What is the difference between RMSE and MAE?  
**A:** RMSE penalizes large errors more than MAE.  

---

### Q85. Which metric is better: RMSE or MAE?  
**A:** Depends on context. RMSE if large errors are critical, MAE if all errors are equally important.  

---

### Q86. What is PRESS statistic?  
**A:** Prediction sum of squares, a measure of how well the model predicts new data.  

---

### Q87. What is AIC in regression?  
**A:** Akaike Information Criterion, a measure used to compare models, penalizing complexity.  

---

### Q88. What is BIC in regression?  
**A:** Bayesian Information Criterion, similar to AIC but penalizes complexity more strongly.  

---

### Q89. Which is stricter: AIC or BIC?  
**A:** BIC is stricter as it imposes a larger penalty for model complexity.  

---

### Q90. How do you compare regression models?  
**A:** Using R², Adjusted R², AIC, BIC, cross-validation, or error metrics (MSE, RMSE).  

---

### Q91. What is stepwise selection?  
**A:** A regression method that automatically adds or removes predictors based on statistical significance.  

---

### Q92. What are backward elimination and forward selection?  
**A:**  
- Forward selection starts with no variables and adds predictors.  
- Backward elimination starts with all predictors and removes them iteratively.  

---

### Q93. What is multivariate regression?  
**A:** Regression with multiple dependent variables.  

---

### Q94. What is covariance in regression?  
**A:** Covariance measures how two variables change together, used in estimating regression slopes.  

---

### Q95. What is the difference between covariance and correlation?  
**A:** Covariance shows direction of relationship, correlation standardizes it between -1 and 1.  

---

### Q96. What is partial regression plot?  
**A:** A plot that shows relationship between dependent variable and one predictor, controlling for others.  

---

### Q97. What is standardized coefficient?  
**A:** A regression coefficient obtained after normalizing variables, allowing comparison of effect sizes.  

---

### Q98. What is multicollinearity tolerance?  
**A:** Tolerance = 1 – R² of regression of one predictor on others. Low tolerance indicates high multicollinearity.  

---

### Q99. What is the condition number in regression?  
**A:** A measure of multicollinearity based on eigenvalues of predictor matrix.  

---

### Q100. When should linear regression not be used?  
**A:** When assumptions are violated (non-linearity, multicollinearity, heteroscedasticity, non-normal errors) or for categorical dependent variables.  

---
