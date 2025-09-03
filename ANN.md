# Artificial Neural Networks (ANN) – 100 Interview Questions & Answers

---

### Q1. What is an Artificial Neural Network (ANN)?
An ANN is a computational model inspired by the human brain that consists of interconnected nodes (neurons) arranged in layers to process data, recognize patterns, and learn representations.

---

### Q2. What are the main components of a neuron in ANN?
* **Inputs**  
* **Weights**  
* **Summation function**  
* **Activation function**  
* **Output**

---

### Q3. What are activation functions in ANN?
Mathematical functions that introduce non-linearity into the model, enabling ANNs to approximate complex relationships.

Common ones: Sigmoid, Tanh, ReLU, Leaky ReLU, Softmax.

---

### Q4. Why is non-linearity important in ANNs?
Without non-linear activations, multiple layers collapse into a single linear transformation, limiting the network’s ability to model complex patterns.

---

### Q5. What is the difference between shallow and deep neural networks?
* **Shallow NN**: 1–2 hidden layers  
* **Deep NN**: Multiple hidden layers (typically >3) enabling hierarchical feature extraction.

---

### Q6. What is forward propagation in ANN?
The process of passing inputs through the network layers, applying weights, biases, and activation functions to produce predictions.

---

### Q7. What is backpropagation?
An algorithm for training ANNs by computing gradients of loss with respect to weights using the chain rule, then updating weights via gradient descent.

---

### Q8. What is the loss function in ANN?
A mathematical function measuring the difference between predicted and actual values.  
Examples: MSE, Cross-Entropy, Hinge Loss.

---

### Q9. What is gradient descent in ANN training?
An optimization algorithm to minimize loss by updating weights in the opposite direction of the gradient of the loss function.

---

### Q10. What are variants of gradient descent?
* Batch Gradient Descent  
* Stochastic Gradient Descent (SGD)  
* Mini-Batch Gradient Descent

---

### Q11. What is the vanishing gradient problem?
In deep networks, gradients can shrink during backpropagation, making learning slow or impossible, especially with sigmoid/tanh activations.

---

### Q12. What is the exploding gradient problem?
When gradients grow uncontrollably during training, causing unstable weight updates.

---

### Q13. How to mitigate vanishing/exploding gradients?
* Use ReLU or variants  
* Gradient clipping  
* Batch normalization  
* Proper weight initialization

---

### Q14. What is weight initialization?
Setting starting values of weights before training.  
Techniques: Random initialization, Xavier, He initialization.

---

### Q15. What is bias in ANN?
A learnable parameter added to weighted inputs, helping shift the activation function and improve flexibility of learning.

---

### Q16. What are hyperparameters in ANN?
Parameters not learned during training but set beforehand: learning rate, batch size, number of layers, number of neurons, activation functions.

---

### Q17. What is overfitting in ANN?
When a model learns training data too well, including noise, leading to poor generalization on unseen data.

---

### Q18. How to prevent overfitting in ANN?
* Dropout  
* L1/L2 regularization  
* Early stopping  
* Data augmentation  
* Cross-validation

---

### Q19. What is dropout in ANN?
A regularization technique where neurons are randomly "dropped" during training to reduce overfitting and improve generalization.

---

### Q20. What is early stopping?
A regularization method where training stops once validation loss stops improving, preventing overfitting.

---

### Q21. What is batch normalization?
A technique that normalizes inputs to each layer, stabilizing training, speeding convergence, and reducing internal covariate shift.

---

### Q22. What is an epoch?
One complete pass through the entire training dataset.

---

### Q23. What is a batch in ANN training?
A subset of the dataset processed before updating model parameters.

---

### Q24. What is batch size?
The number of samples in one batch. Smaller batches → noisier updates but faster; larger batches → smoother convergence but more memory usage.

---

### Q25. Difference between training, validation, and test sets?
* **Training set**: Used to fit the model  
* **Validation set**: Used to tune hyperparameters  
* **Test set**: Used to evaluate final performance

---

### Q26. What is a multilayer perceptron (MLP)?
A type of ANN with multiple fully connected layers using non-linear activation functions.

---

### Q27. What is a perceptron?
The simplest form of ANN neuron — linear combination of inputs followed by an activation function (originally step function).

---

### Q28. What are universal approximation capabilities of ANNs?
A neural network with one hidden layer and enough neurons can approximate any continuous function.

---

### Q29. What is Softmax used for?
To convert logits into probabilities for multi-class classification tasks.

---

### Q30. What is the difference between regression and classification in ANN?
* **Regression**: Predict continuous outputs  
* **Classification**: Predict discrete class labels

---

### Q31. What is a confusion matrix?
A table showing true vs predicted classes, used to evaluate classification performance.

---

### Q32. What evaluation metrics are used in ANN classification?
Accuracy, Precision, Recall, F1-score, AUC-ROC.

---

### Q33. What is cross-entropy loss?
A loss function measuring difference between predicted probabilities and actual class labels, widely used in classification.

---

### Q34. What is Mean Squared Error (MSE)?
A regression loss function measuring the average squared difference between predicted and actual values.

---

### Q35. What is L1 vs L2 regularization?
* **L1 (Lasso)**: Adds absolute value of weights to loss (sparse solutions).  
* **L2 (Ridge)**: Adds squared weights to loss (shrinks weights uniformly).

---

### Q36. What is the role of the learning rate?
Controls the size of weight updates during training. Too high → unstable, too low → slow learning.

---

### Q37. What is learning rate decay?
Reducing learning rate over epochs to improve convergence and avoid overshooting minima.

---

### Q38. What are optimizers in ANN?
Algorithms for updating weights: SGD, Momentum, AdaGrad, RMSProp, Adam.

---

### Q39. Why is Adam optimizer popular?
Combines momentum and adaptive learning rates, works well with sparse gradients, and requires little tuning.

---

### Q40. What is the role of momentum in optimization?
Accelerates gradient descent by adding a fraction of the previous update to the current one, helping escape local minima.

---

### Q41. What is the cost of ANN in terms of computation?
ANNs require high computation and memory, especially deep networks. GPUs/TPUs are often used for acceleration.

---

### Q42. What is transfer learning?
Reusing a pre-trained neural network on a new but related task, often with fine-tuning.

---

### Q43. What is fine-tuning in ANN?
Adjusting weights of a pre-trained model on new data, typically with a lower learning rate.

---

### Q44. What are convolutional neural networks (CNNs)?
Specialized ANNs for image and spatial data, using convolution and pooling layers.

---

### Q45. What are recurrent neural networks (RNNs)?
Neural networks designed for sequential data, with feedback loops to maintain memory of past inputs.

---

### Q46. What is the vanishing gradient problem in RNNs?
Gradients diminish across time steps, preventing long-term dependencies from being learned.

---

### Q47. How do LSTMs solve vanishing gradients?
LSTMs introduce gates (input, forget, output) to regulate information flow and maintain long-term dependencies.

---

### Q48. What is a GRU?
Gated Recurrent Unit — a simplified version of LSTM with fewer gates but similar performance.

---

### Q49. What is an autoencoder?
A neural network trained to reconstruct input data, often used for dimensionality reduction or feature learning.

---

### Q50. What are restricted Boltzmann machines (RBMs)?
Stochastic ANN models with visible and hidden units, used in collaborative filtering and feature learning.

---

### Q51. What is the difference between generative and discriminative models?
* **Generative**: Model data distribution (e.g., autoencoders, GANs)  
* **Discriminative**: Model decision boundary (e.g., classifiers)

---

### Q52. What is a GAN?
Generative Adversarial Network — consists of a generator and discriminator trained adversarially to produce realistic data.

---

### Q53. What are embeddings in ANN?
Low-dimensional vector representations of categorical variables or entities (e.g., word embeddings in NLP).

---

### Q54. What is one-hot encoding?
Representing categorical variables as binary vectors, with one position "hot" (1) and others zero.

---

### Q55. What is feature scaling and why is it important?
Normalizing/standardizing input features helps stabilize training and improve convergence in ANNs.

---

### Q56. What is weight sharing in CNNs?
Using the same kernel weights across different positions in the input, reducing parameters and enabling translation invariance.

---

### Q57. What is pooling in CNNs?
Downsampling operation (e.g., max pooling, average pooling) that reduces spatial dimensions.

---

### Q58. What is padding in CNNs?
Adding zeros around the input to preserve spatial dimensions after convolution.

---

### Q59. What is stride in CNNs?
The step size of the convolution filter. Larger stride reduces output dimensions.

---

### Q60. What is dilated convolution?
Convolution with gaps between kernel elements, allowing larger receptive fields without increasing parameters.

---

### Q61. What is residual connection?
A shortcut connection in ResNets that skips layers, enabling training of very deep networks by mitigating vanishing gradients.

---

### Q62. What is attention mechanism?
Technique allowing networks to focus on relevant parts of the input (e.g., in NLP or vision).

---

### Q63. What are transformers in ANN?
Deep learning architectures relying entirely on attention mechanisms, dominating modern NLP and vision tasks.

---

### Q64. What is self-attention?
Mechanism that computes attention weights between all pairs of input tokens, capturing dependencies regardless of distance.

---

### Q65. What is the difference between CNNs and RNNs?
* **CNNs**: Best for spatial data (images)  
* **RNNs**: Best for sequential data (time series, language)

---

### Q66. What is a hybrid neural network?
Combines multiple architectures (e.g., CNN+RNN) to leverage strengths of each.

---

### Q67. What are vanishing/exploding activations?
When neuron outputs grow too small or too large, leading to unstable training.

---

### Q68. What is layer normalization?
A normalization technique applied across features within each data sample, often used in transformers.

---

### Q69. What is gradient clipping?
Limiting the magnitude of gradients to prevent exploding gradients.

---

### Q70. What is a learning curve?
A plot of model performance (loss or accuracy) vs training epochs, used to diagnose under/overfitting.

---

### Q71. What is pruning in ANN?
Removing unnecessary weights or neurons to compress models and improve efficiency.

---

### Q72. What is quantization in ANN?
Reducing precision of weights/activations (e.g., 32-bit → 8-bit) to accelerate inference on edge devices.

---

### Q73. What are spiking neural networks (SNNs)?
ANNs inspired by biological neurons that transmit spikes (events) instead of continuous signals.

---

### Q74. What is Hebbian learning?
Unsupervised rule: "Neurons that fire together, wire together," strengthening connections based on co-activation.

---

### Q75. What is competitive learning?
Unsupervised learning where neurons compete, and only the "winning" neuron updates weights (used in SOMs).

---

### Q76. What is a Self-Organizing Map (SOM)?
Unsupervised ANN for dimensionality reduction and visualization, mapping high-dimensional data to lower dimensions.

---

### Q77. What is reinforcement learning in context of ANNs?
Using ANNs as function approximators (policy or value function) in RL tasks.

---

### Q78. What is Deep Q-Network (DQN)?
A reinforcement learning algorithm combining Q-learning with deep neural networks.

---

### Q79. What is policy gradient in ANN-based RL?
An approach where the ANN directly outputs action probabilities and optimizes expected reward.

---

### Q80. What is imitation learning?
Training ANNs to mimic expert behavior from demonstrations.

---

### Q81. What are attention heads in transformers?
Multiple attention mechanisms run in parallel, allowing the model to focus on different parts of the input simultaneously.

---

### Q82. What is positional encoding in transformers?
Encoding sequence order information since self-attention is permutation-invariant.

---

### Q83. What is masked language modeling?
Task where some tokens are masked and the ANN predicts them (e.g., BERT training).

---

### Q84. What is autoregressive language modeling?
Predicting the next token based on previous ones (e.g., GPT training).

---

### Q85. What is backpropagation through time (BPTT)?
Extension of backpropagation to unroll RNNs across time steps for gradient computation.

---

### Q86. What are vanishing gradients in RNNs?
As sequence length increases, gradients diminish, preventing learning of long-term dependencies.

---

### Q87. What is teacher forcing in RNNs?
Training technique where true previous outputs are fed instead of predicted outputs to speed up convergence.

---

### Q88. What is exposure bias?
Issue in sequence models trained with teacher forcing, as they see true inputs during training but predicted ones during inference.

---

### Q89. What is scheduled sampling?
Gradually replacing true inputs with predicted ones during training to reduce exposure bias.

---

### Q90. What is a capsule network?
ANN architecture with capsules (groups of neurons) that capture spatial hierarchies and relationships.

---

### Q91. What is knowledge distillation?
Compressing a large model (teacher) into a smaller model (student) by training on soft predictions.

---

### Q92. What are adversarial examples?
Slightly perturbed inputs that fool ANNs into making wrong predictions.

---

### Q93. How to defend against adversarial attacks?
Adversarial training, defensive distillation, gradient masking, robust optimization.

---

### Q94. What is explainability in ANNs?
Techniques (e.g., SHAP, LIME, saliency maps) used to interpret ANN predictions.

---

### Q95. What is interpretability vs explainability?
* **Interpretability**: Understanding internal mechanisms (weights, activations)  
* **Explainability**: Understanding why specific predictions were made

---

### Q96. What is catastrophic forgetting?
When ANNs forget old tasks while learning new ones in sequential training.

---

### Q97. How to mitigate catastrophic forgetting?
Techniques like Elastic Weight Consolidation (EWC), rehearsal, or modular networks.

---

### Q98. What is federated learning with ANN?
Distributed learning where models train locally on edge devices and aggregate updates centrally without sharing raw data.

---

### Q99. What are ethical concerns in ANN applications?
Bias, fairness, accountability, explainability, and misuse of AI (e.g., deepfakes).

---

### Q100. What are future trends in ANN research?
* Larger models (LLMs, multimodal)  
* Efficient architectures (pruning, quantization)  
* Neuroscience-inspired models (SNNs, attention variants)  
* Explainable AI  
* Edge deployment of ANN
