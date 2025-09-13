# Hyperparameter Tuning and Fine-Tuning – 100 Interview Questions & Answers

---

### Q1. What is hyperparameter tuning?
**A1.** Hyperparameter tuning is the process of finding the best set of hyperparameters (configuration values not learned during training) that optimize model performance.

---

### Q2. How do hyperparameters differ from parameters?
**A2.** Parameters are learned during training (e.g., weights, biases), while hyperparameters are set before training (e.g., learning rate, batch size, number of layers).

---

### Q3. Why is hyperparameter tuning important?
**A3.** Proper tuning improves model accuracy, generalization, and stability while avoiding underfitting or overfitting.

---

### Q4. Give examples of common hyperparameters in neural networks.
**A4.** Learning rate, batch size, number of epochs, dropout rate, number of hidden units, optimizer type.

---

### Q5. What is grid search?
**A5.** Grid search systematically tries all possible combinations of hyperparameter values from a predefined grid.

---

### Q6. What is random search?
**A6.** Random search samples random hyperparameter combinations instead of trying all, often finding good configurations faster.

---

### Q7. Compare grid search vs random search.
**A7.** Grid search is exhaustive but computationally expensive; random search is less exhaustive but often more efficient in high-dimensional spaces.

---

### Q8. What is Bayesian optimization?
**A8.** Bayesian optimization builds a probabilistic model of the objective function and selects hyperparameters using acquisition functions to balance exploration and exploitation.

---

### Q9. What is an acquisition function in Bayesian optimization?
**A9.** It determines the next set of hyperparameters to evaluate, examples include Expected Improvement (EI) and Upper Confidence Bound (UCB).

---

### Q10. Explain Hyperband.
**A10.** Hyperband uses adaptive resource allocation and early-stopping strategies to quickly discard poor configurations and allocate more resources to promising ones.

---

### Q11. What is Population-Based Training (PBT)?
**A11.** PBT evolves a population of models by periodically mutating hyperparameters and replacing underperforming models with better ones.

---

### Q12. What is learning rate scheduling?
**A12.** Adjusting the learning rate during training (e.g., step decay, exponential decay, cosine annealing) to improve convergence.

---

### Q13. Why is learning rate one of the most critical hyperparameters?
**A13.** Too high leads to divergence; too low causes slow convergence and possible local minima.

---

### Q14. What is early stopping?
**A14.** A regularization technique that stops training when validation performance stops improving, preventing overfitting.

---

### Q15. What is fine-tuning?
**A15.** Fine-tuning is adapting a pre-trained model to a new dataset or task, usually by training the last few layers or the entire network with lower learning rates.

---

### Q16. What are the main approaches to fine-tuning?
**A16.**
1. Feature extraction (freeze base, train classifier).  
2. Fine-tuning entire model with small learning rate.  
3. Layer-wise unfreezing.

---

### Q17. When should you fine-tune a model?
**A17.** When you have limited data but a pre-trained model exists on a related task/domain.

---

### Q18. What is transfer learning?
**A18.** Transfer learning uses knowledge from one task (pre-trained model) to improve performance on another task.

---

### Q19. Difference between transfer learning and fine-tuning?
**A19.** Transfer learning uses pre-trained features as is (frozen layers), fine-tuning adjusts weights for the new task.

---

### Q20. Why do we use smaller learning rates in fine-tuning?
**A20.** To avoid catastrophic forgetting and preserve useful pre-trained weights.

---

### Q21. What is overfitting in hyperparameter tuning?
**A21.** Overfitting occurs when tuned hyperparameters overly optimize validation data but fail to generalize to unseen data.

---

### Q22. How do you prevent overfitting during tuning?
**A22.** Use cross-validation, early stopping, regularization, dropout, and monitor test set performance.

---

### Q23. What is cross-validation in tuning?
**A23.** A resampling technique (e.g., k-fold) that splits data into multiple folds to ensure robust hyperparameter evaluation.

---

### Q24. What is nested cross-validation?
**A24.** Two-layer cross-validation: inner loop for tuning, outer loop for unbiased performance estimation.

---

### Q25. What is Optuna?
**A25.** Optuna is an open-source hyperparameter optimization library using Bayesian optimization and pruning techniques.

---

### Q26. What is Ray Tune?
**A26.** A distributed hyperparameter tuning library supporting grid search, random search, Hyperband, and PBT.

---

### Q27. What is Keras Tuner?
**A27.** A library for hyperparameter tuning in TensorFlow/Keras supporting random search, Hyperband, and Bayesian optimization.

---

### Q28. What is pruning in tuning?
**A28.** Early stopping of unpromising trials to save computation resources.

---

### Q29. What is a learning rate finder?
**A29.** A method that runs a short training with exponentially increasing learning rates to identify optimal ranges.

---

### Q30. What are warm restarts in tuning?
**A30.** Restarting training periodically with high learning rates (cosine annealing restarts) to escape local minima.

---

### Q31. What is fine-tuning in NLP?
**A31.** Adapting pre-trained language models (like BERT, GPT) to downstream tasks such as classification or QA.

---

### Q32. Why is fine-tuning critical in NLP?
**A32.** Pre-trained models capture general language patterns, fine-tuning adapts them to specific tasks/domains.

---

### Q33. What is full fine-tuning in NLP?
**A33.** Training all model parameters on the downstream task.

---

### Q34. What is parameter-efficient fine-tuning (PEFT)?
**A34.** Fine-tuning only small additional parameters (adapters, LoRA) while keeping the main model frozen.

---

### Q35. What is LoRA (Low-Rank Adaptation)?
**A35.** A parameter-efficient fine-tuning method where low-rank matrices are inserted into pre-trained layers, reducing memory usage.

---

### Q36. What is prompt tuning?
**A36.** Fine-tuning only task-specific prompt embeddings while keeping the model frozen.

---

### Q37. What is prefix tuning?
**A37.** Similar to prompt tuning but trains continuous prefix vectors added to each layer’s input.

---

### Q38. Difference between LoRA and adapter tuning?
**A38.** LoRA uses low-rank decompositions; adapter tuning adds small trainable layers; both reduce memory and compute costs.

---

### Q39. What is catastrophic forgetting?
**A39.** When fine-tuning causes a model to lose knowledge from its pre-training.

---

### Q40. How to reduce catastrophic forgetting?
**A40.** Use small learning rates, gradual unfreezing, and regularization techniques like Elastic Weight Consolidation.

---

### Q41. What is Elastic Weight Consolidation (EWC)?
**A41.** A regularization method that prevents important weights from changing too much during fine-tuning.

---

### Q42. What is continual learning?
**A42.** Training models on sequential tasks while retaining performance on old tasks, often related to fine-tuning challenges.

---

### Q43. What is hyperparameter space?
**A43.** The set of all possible hyperparameter combinations for tuning.

---

### Q44. Why is defining search space critical?
**A44.** Too wide wastes resources, too narrow may miss optimal solutions.

---

### Q45. What is a learning rate warmup?
**A45.** Gradually increasing learning rate at the start of training to stabilize convergence.

---

### Q46. What is cosine annealing?
**A46.** A learning rate schedule where LR decreases following a cosine curve, often with restarts.

---

### Q47. What is fine-tuning in computer vision?
**A47.** Using pre-trained CNNs (e.g., ResNet, VGG) and adapting them to new datasets like medical images or object detection.

---

### Q48. Why use ImageNet pre-trained weights?
**A48.** They capture general image features (edges, textures) useful for transfer to other vision tasks.

---

### Q49. What is model checkpointing?
**A49.** Saving model states during training to resume or restore best-performing models.

---

### Q50. What is hyperparameter sensitivity?
**A50.** The extent to which model performance depends on specific hyperparameter values.

---

### Q51. Which hyperparameters are most sensitive in deep learning?
**A51.** Learning rate, batch size, optimizer choice, and weight initialization.

---

### Q52. What is gradient clipping?
**A52.** Restricting gradient values during backpropagation to prevent exploding gradients.

---

### Q53. How does batch size affect training?
**A53.** Larger batch sizes improve stability but need more memory; smaller batches add noise but may generalize better.

---

### Q54. What is mixed precision training?
**A54.** Training with both FP16 and FP32 precision to reduce memory usage and speed up training.

---

### Q55. What is model fine-tuning vs feature extraction?
**A55.** Fine-tuning updates weights; feature extraction uses frozen pre-trained features and only trains the classifier.

---

### Q56. When should you freeze layers in fine-tuning?
**A56.** When the dataset is small or the new task is similar to the pre-trained one.

---

### Q57. When should you unfreeze layers?
**A57.** When the dataset is large or the new task is significantly different from pre-training.

---

### Q58. What is differential learning rate in fine-tuning?
**A58.** Using different learning rates for different layers (lower for earlier layers, higher for later).

---

### Q59. What is model distillation?
**A59.** Transferring knowledge from a large teacher model to a smaller student model, often combined with fine-tuning.

---

### Q60. What is zero-shot transfer?
**A60.** Using a pre-trained model directly on a new task without any fine-tuning.

---

### Q61. What is few-shot fine-tuning?
**A61.** Adapting models to tasks using very small labeled datasets.

---

### Q62. What is domain adaptation in fine-tuning?
**A62.** Adjusting pre-trained models to work in new domains (e.g., from news text to medical text).

---

### Q63. What is hyperparameter optimization objective function?
**A63.** The function (e.g., validation accuracy, loss) we aim to optimize during tuning.

---

### Q64. What is black-box optimization?
**A64.** Optimization without explicit knowledge of the objective function’s form, common in hyperparameter tuning.

---

### Q65. What is surrogate modeling?
**A65.** Building an approximate model of the objective function to guide hyperparameter selection (used in Bayesian optimization).

---

### Q66. What is multi-fidelity optimization?
**A66.** Evaluating models with reduced resources (smaller dataset, fewer epochs) to approximate performance cheaply.

---

### Q67. What is grid resolution in grid search?
**A67.** The number of discrete values chosen for each hyperparameter.

---

### Q68. Why does random search often outperform grid search?
**A68.** Random search explores hyperparameter space more efficiently in high dimensions.

---

### Q69. What is curse of dimensionality in hyperparameter tuning?
**A69.** The exponential growth of possible combinations as the number of hyperparameters increases.

---

### Q70. What is model ensembling after fine-tuning?
**A70.** Combining multiple fine-tuned models to improve robustness and accuracy.

---

### Q71. What is checkpoint averaging?
**A71.** Averaging weights from multiple checkpoints of a fine-tuned model to improve stability.

---

### Q72. What is gradient accumulation?
**A72.** Accumulating gradients over multiple mini-batches to simulate larger batch training with less memory.

---

### Q73. What is search budget in tuning?
**A73.** The computational resources (time, trials, GPUs) allocated for tuning.

---

### Q74. What is manual hyperparameter tuning?
**A74.** Adjusting hyperparameters manually based on intuition and trial/error.

---

### Q75. Why is manual tuning not scalable?
**A75.** It’s time-consuming, subjective, and inefficient for complex models.

---

### Q76. What are default hyperparameters?
**A76.** Pre-set values in libraries (like scikit-learn) that work reasonably well for general tasks.

---

### Q77. Should you rely on default hyperparameters?
**A77.** They are good starting points but often suboptimal for specific tasks.

---

### Q78. What is meta-learning in tuning?
**A78.** Learning from previous tuning tasks to accelerate tuning on new tasks.

---

### Q79. What is transfer tuning?
**A79.** Using tuning results from similar datasets/models as priors for new tasks.

---

### Q80. What is adaptive learning rate optimizers?
**A80.** Optimizers like Adam, RMSprop, and Adagrad that adjust learning rates per parameter.

---

### Q81. Why is Adam popular?
**A81.** Adam combines momentum and adaptive learning rates, making it efficient and robust.

---

### Q82. When might SGD outperform Adam?
**A82.** On very large datasets and when better generalization is required.

---

### Q83. What is weight decay?
**A83.** A regularization technique that penalizes large weights, preventing overfitting.

---

### Q84. How is weight decay different from L2 regularization?
**A84.** Functionally similar, but weight decay is applied directly in optimizers.

---

### Q85. What is fine-tuning with data augmentation?
**A85.** Applying transformations (flip, rotate, noise) during fine-tuning to improve generalization.

---

### Q86. What is learning rate annealing?
**A86.** Gradually decreasing learning rate as training progresses.

---

### Q87. What is one-cycle learning rate policy?
**A87.** Learning rate increases then decreases in a single cycle, often improving convergence.

---

### Q88. What is model calibration after fine-tuning?
**A88.** Adjusting predicted probabilities to reflect true likelihoods.

---

### Q89. What is hyperparameter sweep?
**A89.** Running multiple experiments with different hyperparameters to explore performance.

---

### Q90. What is early pruning in tuning?
**A90.** Stopping underperforming trials before completion to save resources.

---

### Q91. What is trial in tuning?
**A91.** One evaluation run of a specific hyperparameter configuration.

---

### Q92. What is experiment tracking?
**A92.** Logging hyperparameters, metrics, and results (tools: MLflow, Weights & Biases).

---

### Q93. Why is reproducibility important in tuning?
**A93.** To ensure consistent results and fair comparisons.

---

### Q94. How to ensure reproducibility?
**A94.** Fix random seeds, log configurations, and use version-controlled datasets.

---

### Q95. What is hyperparameter transferability?
**A95.** Whether hyperparameters tuned on one dataset generalize to another.

---

### Q96. What is fine-tuning efficiency challenge?
**A96.** High computational cost when adapting very large pre-trained models.

---

### Q97. What is distillation + fine-tuning?
**A97.** Distilling knowledge into a smaller model and fine-tuning it for efficiency.

---

### Q98. What is large-scale hyperparameter tuning?
**A98.** Tuning models using distributed clusters, often in industrial-scale ML.

---

### Q99. What is AutoML in hyperparameter tuning?
**A99.** Automated Machine Learning frameworks that search for best models and hyperparameters automatically.

---

### Q100. Summarize hyperparameter tuning vs fine-tuning.
**A100.** Hyperparameter tuning finds best training settings; fine-tuning adapts pre-trained models to new tasks. Both aim to improve performance, stability, and generalization.

---
