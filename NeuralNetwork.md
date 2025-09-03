# Neural Networks Interview Questions & Answers

---

### Q1. What is a neural network?
A neural network is a computational model inspired by the human brain, consisting of interconnected nodes (neurons) organized into layers that process input data to produce outputs.

---

### Q2. What are the main components of a neural network?
- **Input layer**: receives raw data  
- **Hidden layers**: process and transform features  
- **Output layer**: produces predictions  
- **Weights**: parameters adjusted during training  
- **Activation functions**: introduce non-linearity  

---

### Q3. What is an artificial neuron?
It is a mathematical function that applies a weighted sum of inputs, adds a bias, and passes it through an activation function.

---

### Q4. What is an activation function?
A function that introduces non-linearity into a neural network, enabling it to learn complex patterns.

---

### Q5. What are common activation functions?
- Sigmoid  
- Tanh  
- ReLU  
- Leaky ReLU  
- Softmax  

---

### Q6. Why is non-linearity important in neural networks?
Non-linearity allows networks to approximate complex, non-linear functions beyond simple linear relationships.

---

### Q7. What is a feedforward neural network?
A type of NN where connections flow forward from inputs to outputs without cycles.

---

### Q8. What is backpropagation?
An algorithm used to train neural networks by propagating errors backward to update weights.

---

### Q9. What is the loss function?
A function that measures the difference between predicted and actual outputs.

---

### Q10. What are common loss functions?
- Mean Squared Error (MSE)  
- Cross-Entropy Loss  
- Hinge Loss  

---

### Q11. What is gradient descent?
An optimization algorithm that updates weights by moving in the direction of the negative gradient of the loss function.

---

### Q12. What are variants of gradient descent?
- Batch Gradient Descent  
- Stochastic Gradient Descent (SGD)  
- Mini-Batch Gradient Descent  

---

### Q13. What is the learning rate?
A hyperparameter controlling the step size in weight updates during training.

---

### Q14. What happens if the learning rate is too high?
The model may diverge and fail to converge.

---

### Q15. What happens if the learning rate is too low?
The training will be very slow and may get stuck in local minima.

---

### Q16. What is overfitting in neural networks?
When a model learns training data too well, including noise, and fails to generalize to unseen data.

---

### Q17. How can overfitting be prevented?
- Regularization (L1/L2)  
- Dropout  
- Early stopping  
- Data augmentation  

---

### Q18. What is underfitting?
When a model is too simple to capture underlying patterns in the data.

---

### Q19. What are hyperparameters in neural networks?
Parameters set before training, such as learning rate, number of layers, number of neurons, batch size, and dropout rate.

---

### Q20. What is weight initialization?
The process of setting initial values for network weights before training.

---

### Q21. What are common weight initialization methods?
- Random initialization  
- Xavier initialization  
- He initialization  

---

### Q22. What is vanishing gradient problem?
When gradients become too small during backpropagation, preventing weights from updating effectively.

---

### Q23. What is exploding gradient problem?
When gradients become too large, causing unstable weight updates.

---

### Q24. How can vanishing/exploding gradients be addressed?
- Proper weight initialization  
- ReLU activations  
- Gradient clipping  
- Batch normalization  

---

### Q25. What is dropout?
A regularization technique that randomly deactivates neurons during training to prevent overfitting.

---

### Q26. What is batch normalization?
A technique to normalize inputs of each layer, improving stability and speeding up training.

---

### Q27. What is the difference between shallow and deep networks?
- Shallow: 1–2 hidden layers  
- Deep: many hidden layers enabling hierarchical feature learning  

---

### Q28. What is deep learning?
A subset of machine learning using deep neural networks with multiple layers.

---

### Q29. What are convolutional neural networks (CNNs)?
NNs specialized for grid-like data such as images, using convolutional layers.

---

### Q30. What are recurrent neural networks (RNNs)?
NNs with recurrent connections, suitable for sequential data like time series and text.

---

### Q31. What is transfer learning?
Using a pre-trained model on one task as a starting point for another task.

---

### Q32. What is fine-tuning?
Adjusting the weights of a pre-trained model slightly for a new task.

---

### Q33. What is feature extraction?
Using pre-trained model layers as fixed feature generators for a new task.

---

### Q34. What is reinforcement learning in relation to NNs?
Using NNs as function approximators in reinforcement learning agents.

---

### Q35. What is an epoch?
One full pass through the entire training dataset.

---

### Q36. What is a batch in training?
A subset of training data processed before updating weights.

---

### Q37. What is mini-batch training?
Splitting data into small batches for weight updates, balancing speed and stability.

---

### Q38. What is early stopping?
Halting training when validation performance stops improving.

---

### Q39. What is the difference between supervised and unsupervised NNs?
- Supervised: trained with labeled data  
- Unsupervised: learns patterns without labels (e.g., autoencoders)  

---

### Q40. What is a perceptron?
The simplest form of NN unit, performing a weighted sum and threshold activation.

---

### Q41. What is a multi-layer perceptron (MLP)?
A feedforward NN with one or more hidden layers.

---

### Q42. What is the universal approximation theorem?
A neural network with at least one hidden layer can approximate any continuous function, given enough neurons.

---

### Q43. What is pruning in NNs?
Removing unnecessary neurons or connections to reduce complexity.

---

### Q44. What are skip connections?
Links that bypass layers, used in architectures like ResNet to mitigate vanishing gradients.

---

### Q45. What are residual networks (ResNets)?
Deep NNs using skip connections to allow training of very deep models.

---

### Q46. What is a GAN?
A Generative Adversarial Network, consisting of a generator and discriminator trained adversarially.

---

### Q47. What is an autoencoder?
An NN trained to compress and reconstruct input data, useful for dimensionality reduction.

---

### Q48. What is word embedding?
Representing words as dense vectors in a continuous space for NLP tasks.

---

### Q49. What are attention mechanisms?
Techniques to focus on relevant parts of input sequences, improving performance in sequence models.

---

### Q50. What is a transformer model?
A deep learning architecture based on self-attention, widely used in NLP.

---

### Q51. What are hyperparameter optimization techniques?
- Grid search  
- Random search  
- Bayesian optimization  
- Hyperband  

---

### Q52. What is regularization in neural networks?
Techniques to prevent overfitting, e.g., L1, L2, dropout.

---

### Q53. What is L1 regularization?
Encourages sparsity by penalizing absolute values of weights.

---

### Q54. What is L2 regularization?
Penalizes squared values of weights, preventing large weights.

---

### Q55. What is elastic net regularization?
Combination of L1 and L2 regularization.

---

### Q56. What is gradient clipping?
Restricting gradient values to prevent exploding gradients.

---

### Q57. What is weight decay?
A form of L2 regularization applied during optimization.

---

### Q58. What is a learning rate scheduler?
A technique to adjust learning rate during training for better convergence.

---

### Q59. What is a momentum term in optimization?
Helps accelerate SGD by smoothing updates using past gradients.

---

### Q60. What is Adam optimizer?
An adaptive optimizer that combines momentum and adaptive learning rates.

---

### Q61. What is RMSProp?
An optimizer that scales learning rates by a moving average of squared gradients.

---

### Q62. What is Adagrad?
An optimizer that adapts learning rates for each parameter individually.

---

### Q63. What is the main difference between Adam and SGD?
Adam adapts learning rates dynamically, while SGD uses a fixed rate.

---

### Q64. What is gradient checking?
A method to verify correctness of backpropagation by comparing analytical and numerical gradients.

---

### Q65. What is the curse of dimensionality in NNs?
High-dimensional data makes learning harder and requires more data.

---

### Q66. What is dimensionality reduction in NNs?
Reducing input feature space using autoencoders or PCA.

---

### Q67. What are embeddings?
Low-dimensional dense vector representations of high-dimensional inputs.

---

### Q68. What is a softmax function?
Converts raw outputs into probabilities summing to 1.

---

### Q69. What is cross-entropy loss?
Measures difference between predicted and true probability distributions.

---

### Q70. What is a confusion matrix?
A table showing true vs predicted classifications.

---

### Q71. What are precision, recall, and F1 score?
Metrics evaluating classification performance.  
- Precision: TP / (TP+FP)  
- Recall: TP / (TP+FN)  
- F1: harmonic mean of precision and recall  

---

### Q72. What is ROC-AUC?
A performance metric measuring tradeoff between true positive and false positive rates.

---

### Q73. What is data augmentation?
Expanding dataset size by transformations (flips, rotations, noise).

---

### Q74. What is one-hot encoding?
Representing categorical variables as binary vectors.

---

### Q75. What is label smoothing?
Assigning softened labels (e.g., 0.9 instead of 1.0) to improve generalization.

---

### Q76. What is ensemble learning in NNs?
Combining multiple models to improve performance.

---

### Q77. What is model distillation?
Training a smaller network (student) to mimic a larger one (teacher).

---

### Q78. What is adversarial training?
Training with adversarial examples to improve robustness.

---

### Q79. What are adversarial examples?
Inputs slightly perturbed to fool a neural network.

---

### Q80. What is gradient vanishing in RNNs?
Gradients shrink exponentially over time steps, limiting learning of long-term dependencies.

---

### Q81. What is a recurrent connection?
A loop connection in RNNs that feeds previous outputs as inputs.

---

### Q82. What is an embedding layer?
A layer mapping categorical variables to dense vectors.

---

### Q83. What is fine-tuning in neural networks?
Adjusting pre-trained weights slightly for a specific task.

---

### Q84. What is parameter sharing in NNs?
Reusing the same weights across multiple parts of the model (common in CNNs).

---

### Q85. What is an attention layer?
A mechanism to weigh importance of different inputs in sequence models.

---

### Q86. What is sequence-to-sequence learning?
Mapping input sequences to output sequences (e.g., translation).

---

### Q87. What is gradient explosion?
When gradients grow uncontrollably during backpropagation.

---

### Q88. What is normalization in NNs?
Scaling inputs or activations to improve stability (e.g., batch norm, layer norm).

---

### Q89. What is a highway network?
A type of NN with gating mechanisms to control information flow.

---

### Q90. What is parameter initialization?
Setting initial values of weights and biases before training.

---

### Q91. What is a bias term?
A constant added to neuron input to shift activation function.

---

### Q92. What is forward propagation?
The process of passing input data through the network to get predictions.

---

### Q93. What is stochasticity in training?
Randomness introduced through initialization, data shuffling, or dropout.

---

### Q94. What is the difference between training and inference?
- Training: weights updated using backpropagation.  
- Inference: forward pass only for predictions.  

---

### Q95. What is a receptive field in NNs?
The input region a neuron is sensitive to (important in CNNs).

---

### Q96. What is multi-task learning?
Training a model to perform multiple related tasks simultaneously.

---

### Q97. What is a loss landscape?
The surface defined by loss values over parameter space.

---

### Q98. What are skip-gram and CBOW?
Neural architectures for word embeddings.  
- Skip-gram: predicts context from word.  
- CBOW: predicts word from context.  

---

### Q99. What is a Siamese network?
A NN architecture with twin subnetworks used for similarity learning.

---

### Q100. Summarize neural networks in one line.
Neural networks are layered computational models that learn hierarchical feature representations to approximate complex functions.

---
