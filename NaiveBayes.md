# Naive Bayes – 100 Interview Questions and Answers

## Basics (Q1–Q20)

### Q1. What is Naive Bayes?
Naive Bayes is a probabilistic classification algorithm based on Bayes’ Theorem with the assumption that features are conditionally independent given the class. Despite the “naive” assumption, it performs well in many real-world problems like text classification and spam detection.

### Q2. What is Bayes’ theorem?
Bayes’ theorem describes the probability of a hypothesis based on prior knowledge and observed evidence.  
\[
P(H|E) = \frac{P(E|H) \cdot P(H)}{P(E)}
\]  
Here, \(H\) is the hypothesis (class), and \(E\) is the observed evidence (features).

### Q3. Why is it called “naive”?  
It is called “naive” because it assumes that all features are independent of each other given the class label. This assumption is rarely true in real-world data, but the model still works surprisingly well.

### Q4. What are the main applications of Naive Bayes?  
Common applications include spam filtering, sentiment analysis, text classification, medical diagnosis, and recommendation systems. It is particularly useful in natural language processing due to high-dimensional data handling.

### Q5. What types of Naive Bayes classifiers exist?  
- **Gaussian Naive Bayes** – for continuous data assumed to follow normal distribution.  
- **Multinomial Naive Bayes** – for discrete counts (e.g., word counts in text).  
- **Bernoulli Naive Bayes** – for binary/boolean features (word present or absent).  
- **Complement Naive Bayes** – a variant designed for imbalanced text classification.

### Q6. Why is Naive Bayes popular for text classification?  
Text data is high-dimensional and sparse. Naive Bayes handles high dimensions efficiently, is computationally fast, and often performs surprisingly well compared to more complex algorithms.

### Q7. What is prior probability?  
Prior probability represents initial belief about the probability of a class before seeing any data. It is usually estimated as the relative frequency of classes in the training set.

### Q8. What is likelihood in Naive Bayes?  
Likelihood is the probability of observing a feature given a class, i.e., \(P(feature|class)\). It is estimated from training data and combined with priors to compute posterior probabilities.

### Q9. What is posterior probability?  
Posterior probability is the probability of a class given the observed features. It is computed using Bayes’ theorem and used to decide class labels.

### Q10. How does Naive Bayes make predictions?  
Naive Bayes computes posterior probabilities for all classes given the input features and predicts the class with the maximum posterior probability (MAP estimate).

### Q11. Why is Naive Bayes fast?  
It only requires counting feature occurrences for each class and applying simple probability multiplications. Training and prediction are both linear in time with respect to the number of features and samples.

### Q12. What is the assumption of feature independence?  
It assumes that all features contribute independently to the probability of the class. In other words, one feature’s presence or absence doesn’t affect another.

### Q13. Can Naive Bayes be used for regression?  
Naive Bayes is primarily a classification algorithm, not regression. However, probabilistic regression variants exist but are rarely used.

### Q14. What kind of data is suitable for Naive Bayes?  
Categorical, text, and high-dimensional data are most suitable. Continuous features can be handled with Gaussian Naive Bayes if they follow normal distribution.

### Q15. Is Naive Bayes parametric or non-parametric?  
It is a parametric algorithm since it makes assumptions about the probability distribution of features (e.g., Gaussian, multinomial).

### Q16. Why is Naive Bayes considered a generative model?  
Because it models the joint probability distribution \(P(X, Y)\), where \(X\) are features and \(Y\) is the class label, unlike discriminative models (like logistic regression) which model \(P(Y|X)\).

### Q17. What is conditional independence?  
Conditional independence means that two variables are independent given knowledge of a third. Naive Bayes assumes features are conditionally independent given the class label.

### Q18. Why is Naive Bayes not always accurate despite strong performance in text?  
In many real-world datasets, features are correlated, which violates the independence assumption. This can lead to overconfident predictions, though classification accuracy may still be high.

### Q19. What is the difference between generative and discriminative classifiers?  
Generative models (Naive Bayes) model joint probability \(P(X,Y)\). Discriminative models (Logistic Regression, SVM) model conditional probability \(P(Y|X)\). Generative models can generate new data points, discriminative models focus on decision boundaries.

### Q20. How do you implement Naive Bayes in practice?  
Most libraries like scikit-learn provide implementations (`GaussianNB`, `MultinomialNB`, `BernoulliNB`). Implementation involves fitting the model on training data and using `.predict()` for classification.

---

## Types of Naive Bayes (Q21–Q40)

### Q21. What is Gaussian Naive Bayes?  
It assumes continuous features follow a normal distribution. Likelihood is estimated using the Gaussian probability density function.

### Q22. What is Multinomial Naive Bayes?  
Used for discrete count data (e.g., word frequencies in documents). Likelihood is based on multinomial distribution.

### Q23. What is Bernoulli Naive Bayes?  
Assumes binary features (present/absent). Useful for text classification when only word presence matters.

### Q24. What is Complement Naive Bayes?  
A modification of Multinomial NB that reduces bias for imbalanced classes. It is designed for text classification.

### Q25. When to use Gaussian NB?  
Use when features are continuous and approximately normally distributed, such as medical data with measurements.

### Q26. When to use Multinomial NB?  
Use when data consists of count-based features like word counts or term frequencies.

### Q27. When to use Bernoulli NB?  
Use when features are binary indicators (e.g., word present or absent).

### Q28. What are the limitations of Gaussian NB?  
If features are not normally distributed, the Gaussian assumption may be inaccurate, reducing performance.

### Q29. Why is Complement NB better for imbalanced datasets?  
It estimates probabilities using complement classes, which gives more balanced estimates for minority classes.

### Q30. How do you decide which type of NB to use?  
It depends on the feature type: Gaussian for continuous, Multinomial for counts, Bernoulli for binary, Complement for imbalanced text.

---

## Probability & Math (Q41–Q60)

### Q41. Why use log probabilities?  
Multiplying many small probabilities leads to underflow. Logs convert multiplication into addition, ensuring numerical stability.

### Q42. How is MAP (Maximum A Posteriori) used in NB?  
The class with the highest posterior probability (MAP) is chosen as the prediction.

### Q43. What is MLE in Naive Bayes?  
Maximum Likelihood Estimation is used to estimate parameters like class priors and feature likelihoods.

### Q44. What is Laplace smoothing?  
A technique to handle zero probabilities by adding a constant (usually 1) to all counts.

### Q45. Why is smoothing important?  
Without it, unseen words would make class probabilities zero, leading to misclassification.

### Q46. What is add-k smoothing?  
Generalized Laplace smoothing where any constant k is added, not just 1.

### Q47. How does NB handle missing data?  
Features with missing values can be skipped since independence assumption allows ignoring them.

### Q48. What is conditional probability in NB?  
It is the probability of a feature given the class, estimated from training data.

### Q49. How do you calculate posterior probability?  
Posterior = (Likelihood × Prior) / Evidence. The evidence normalizes probabilities across classes.

### Q50. What is zero-frequency problem?  
When a feature never appears with a class, probability becomes zero. Smoothing fixes this.

### Q51. What is the curse of dimensionality for NB?  
With many correlated features, independence assumption breaks down, reducing accuracy.

### Q52. Why does NB often work well despite independence assumption being false?  
Because decision boundaries are often correct even if probability estimates are off.

### Q53. What is Naive Bayes’ decision rule?  
Choose the class with the maximum posterior probability (MAP rule).

### Q54. What is the difference between MAP and MLE in NB?  
MLE estimates parameters directly from data, MAP incorporates priors into estimation.

### Q55. Why does NB assume feature independence?  
For simplicity and computational efficiency. It allows easy probability calculation.

### Q56. What is the evidence term in Bayes theorem?  
It is the marginal probability of observed data, ensuring probabilities sum to 1.

### Q57. Why is the denominator often ignored in NB?  
Since it is constant across classes, only numerator matters for classification.

### Q58. What happens if priors are incorrect?  
If priors are wrong, predictions may be biased, especially with small data.

### Q59. Can priors be uniform?  
Yes, using uniform priors assumes all classes are equally likely.

### Q60. How do you estimate priors from data?  
By calculating class frequencies in the training dataset.

---

## Applications & Practical Aspects (Q61–Q80)

### Q61. Why is NB used in spam filtering?  
Emails can be represented as word counts, making NB efficient and effective for spam detection.

### Q62. Why is NB used in sentiment analysis?  
Sentiment classification relies on word presence/absence, which NB handles well.

### Q63. What is text classification with NB?  
Documents are represented as bag-of-words and classified using NB probabilities.

### Q64. Why is NB fast in training?  
It only requires counting frequencies, no iterative optimization.

### Q65. Can NB handle online learning?  
Yes, probabilities can be updated incrementally as new data arrives.

### Q66. Can NB handle continuous features?  
Yes, with Gaussian NB assuming normal distribution.

### Q67. What is a limitation of NB in image classification?  
Pixel correlations violate independence assumption, leading to poor performance.

### Q68. How does NB handle imbalanced data?  
Poorly, unless modified (Complement NB) or with class weighting.

### Q69. Is NB interpretable?  
Yes, because probabilities for each feature and class are transparent.

### Q70. Why is NB robust to irrelevant features?  
Irrelevant features contribute equally to all classes, reducing impact.

### Q71. What are common preprocessing steps for NB?  
Tokenization, stopword removal, and feature scaling (for Gaussian NB).

### Q72. Does feature scaling matter for NB?  
Only for Gaussian NB, since it assumes normal distribution.

### Q73. Can NB handle outliers?  
Poorly, since extreme values distort probability estimates.

### Q74. What is the runtime complexity of NB?  
Training: O(n × d), Prediction: O(d), where n=instances, d=features.

### Q75. Why is NB memory efficient?  
It only stores class counts and conditional probabilities.

### Q76. Can NB be used for anomaly detection?  
Yes, by modeling likelihood of normal events and flagging unlikely ones.

### Q77. Can NB be used for multiclass problems?  
Yes, NB naturally handles multiple classes.

### Q78. How does NB compare to Logistic Regression?  
NB is faster, works with less data, but is less flexible than logistic regression.

### Q79. How does NB compare to SVM?  
SVM often achieves higher accuracy but is computationally heavier.

### Q80. How does NB compare to Decision Trees?  
NB is probabilistic and simpler; Decision Trees are non-parametric and more flexible.

---

## Advanced & Real-World (Q81–Q100)

### Q81. What are the strengths of NB?  
Fast, simple, scalable, works well with high-dimensional data, interpretable.

### Q82. What are the weaknesses of NB?  
Strong independence assumption, poor with correlated features, bad with continuous non-Gaussian data.

### Q83. Can NB handle non-linear decision boundaries?  
No, NB creates linear decision boundaries in log space.

### Q84. How does NB deal with class imbalance?  
Performance suffers; solutions include resampling or using Complement NB.

### Q85. What evaluation metrics are best for NB?  
Accuracy, Precision, Recall, F1-score, ROC-AUC, depending on class balance.

### Q86. Why does NB sometimes outperform complex models?  
Because high-dimensional sparse data favors simple probability models.

### Q87. Can NB be used in ensemble methods?  
Yes, NB can be a base learner in bagging, boosting, or stacking.

### Q88. How does NB perform with small datasets?  
Performs reasonably well since it needs few parameters, but may be biased.

### Q89. Why does NB not overfit easily?  
Because of its simple structure and few parameters.

### Q90. How do you interpret NB results?  
By analyzing conditional probabilities of features given classes.

### Q91. Can NB probabilities be calibrated?  
Yes, methods like Platt scaling or isotonic regression can calibrate them.

### Q92. Why do NB probabilities tend to be extreme?  
Independence assumption causes over-counting of correlated evidence.

### Q93. Can NB handle mixed feature types?  
Yes, but features must be handled separately (Gaussian for continuous, Multinomial/Bernoulli for categorical).

### Q94. Why is NB a good baseline?  
It’s simple, fast, and often provides competitive results to compare with.

### Q95. What industries use NB?  
Email spam filtering, medical diagnosis, sentiment analysis, document classification.

### Q96. What is a real-world limitation of NB?  
Poor handling of correlated features like pixels in images.

### Q97. How can NB be improved?  
Through feature selection, dimensionality reduction, or using hybrid models.

### Q98. Can NB be used for feature selection?  
Yes, by ranking features based on information gain or likelihood.

### Q99. What is the role of smoothing in real-world NB?  
Smoothing ensures unseen words don’t nullify probabilities, critical in text applications.

### Q100. What is the bottom-line advantage of NB?  
It is simple, interpretable, efficient, and works surprisingly well on many practical problems despite its naive assumption.
