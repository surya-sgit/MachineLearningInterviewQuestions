# Deep Learning (DL) - 100 Interview Questions & Answers

---

### Q1. What is Deep Learning?
A subset of machine learning that uses neural networks with multiple layers to learn hierarchical feature representations.

---

### Q2. How is deep learning different from machine learning?
- ML often relies on manual feature engineering.  
- DL automatically learns features from raw data using neural networks.  

---

### Q3. What are common applications of deep learning?
Image recognition, NLP, speech recognition, recommendation systems, autonomous driving, healthcare diagnostics.

---

### Q4. What are artificial neurons?
Mathematical functions inspired by biological neurons that compute weighted sums of inputs and apply activation functions.

---

### Q5. What is an activation function?
A function introducing non-linearity into the network (e.g., ReLU, Sigmoid, Tanh).

---

### Q6. What is the role of ReLU?
Rectified Linear Unit speeds up convergence and avoids vanishing gradient by outputting max(0, x).

---

### Q7. What is the vanishing gradient problem?
When gradients become very small in deep networks, preventing effective learning.

---

### Q8. How to mitigate vanishing gradients?
Use ReLU/Leaky ReLU, batch normalization, proper initialization, skip connections (ResNets).

---

### Q9. What is the exploding gradient problem?
When gradients become excessively large, causing unstable training.

---

### Q10. How to prevent exploding gradients?
Gradient clipping, better initialization, smaller learning rate.

---

### Q11. What is a feedforward neural network?
A neural network where connections do not form cycles (information flows forward).

---

### Q12. What is backpropagation?
Algorithm for training neural networks by propagating errors backward to update weights.

---

### Q13. What is stochastic gradient descent (SGD)?
An optimization algorithm that updates weights using one or a few samples at a time.

---

### Q14. What are variants of SGD?
Momentum, Nesterov Accelerated Gradient, RMSProp, Adam, Adagrad.

---

### Q15. What is the Adam optimizer?
Adaptive Moment Estimation — combines momentum and adaptive learning rates.

---

### Q16. What is batch size?
The number of training samples used before updating weights.

---

### Q17. What is epoch?
One full pass of the training dataset through the model.

---

### Q18. What is mini-batch gradient descent?
Gradient descent using subsets of data per iteration.

---

### Q19. What is overfitting in deep learning?
When the model performs well on training data but poorly on unseen data.

---

### Q20. How to prevent overfitting?
Regularization, dropout, data augmentation, early stopping.

---

### Q21. What is dropout?
Randomly dropping neurons during training to prevent overfitting.

---

### Q22. What is batch normalization?
Technique that normalizes activations in each layer to stabilize training.

---

### Q23. What is layer normalization?
Normalizing inputs across features for each data point.

---

### Q24. What is weight initialization?
Choosing starting values for weights (e.g., Xavier, He initialization).

---

### Q25. What is transfer learning?
Using pre-trained models on new tasks with fine-tuning.

---

### Q26. What is fine-tuning?
Adjusting pre-trained model weights for a specific task.

---

### Q27. What is feature extraction in DL?
Using a pre-trained model to extract features and training a classifier on top.

---

### Q28. What is data augmentation?
Artificially increasing dataset size via transformations (e.g., flipping, rotation).

---

### Q29. What is early stopping?
Stopping training when validation loss stops improving.

---

### Q30. What is cross-entropy loss?
A loss function commonly used in classification tasks.

---

### Q31. What is mean squared error (MSE)?
A loss function for regression: average squared difference between predictions and targets.

---

### Q32. What is categorical cross-entropy?
Loss function for multi-class classification.

---

### Q33. What is binary cross-entropy?
Loss function for binary classification.

---

### Q34. What is the difference between training, validation, and test sets?
- Training: fit the model.  
- Validation: tune hyperparameters.  
- Test: evaluate final performance.  

---

### Q35. What is k-fold cross-validation?
Splitting data into k parts and training/testing k times for robustness.

---

### Q36. What is a convolutional neural network (CNN)?
A DL model specialized for spatial data like images.

---

### Q37. What is a recurrent neural network (RNN)?
A DL model specialized for sequential data.

---

### Q38. What is a transformer?
A model using self-attention for parallel sequence processing.

---

### Q39. What are embeddings?
Vector representations of words, sentences, or items in a lower-dimensional space.

---

### Q40. What is Word2Vec?
A neural embedding method that learns word representations via CBOW or Skip-Gram.

---

### Q41. What is GloVe?
Global Vectors for Word Representation — embedding based on co-occurrence statistics.

---

### Q42. What is BERT?
Bidirectional Encoder Representations from Transformers — a pre-trained NLP model.

---

### Q43. What is GPT?
Generative Pre-trained Transformer — autoregressive language model.

---

### Q44. What is LSTM?
Long Short-Term Memory — an RNN variant handling long-term dependencies.

---

### Q45. What is GRU?
Gated Recurrent Unit — a simplified LSTM with fewer parameters.

---

### Q46. What is an autoencoder?
A neural network for unsupervised learning via encoding and decoding.

---

### Q47. What is a variational autoencoder (VAE)?
A generative model learning latent distributions.

---

### Q48. What is a GAN?
Generative Adversarial Network with generator and discriminator.

---

### Q49. What is reinforcement learning (RL)?
A paradigm where agents learn from rewards via interactions.

---

### Q50. What is supervised learning vs unsupervised learning?
- Supervised: learns from labeled data.  
- Unsupervised: learns from unlabeled data.  

---

### Q51. What is semi-supervised learning?
Uses a small amount of labeled data with lots of unlabeled data.

---

### Q52. What is self-supervised learning?
Generating labels automatically from data itself.

---

### Q53. What is multi-task learning?
Training one model to perform multiple tasks simultaneously.

---

### Q54. What is transfer vs multi-task learning?
- Transfer: sequential tasks.  
- Multi-task: simultaneous tasks.  

---

### Q55. What is one-shot learning?
Learning from just one example.

---

### Q56. What is few-shot learning?
Learning from a small number of examples.

---

### Q57. What is zero-shot learning?
Generalizing to unseen classes without training examples.

---

### Q58. What is attention in DL?
Mechanism to focus on important parts of input sequence.

---

### Q59. What is self-attention?
Attention applied within a sequence to capture dependencies.

---

### Q60. What is multi-head attention?
Running self-attention multiple times in parallel for richer representations.

---

### Q61. What is a residual connection?
Adding input of a layer to its output (ResNet).

---

### Q62. What is a skip connection?
Shortcut connection allowing gradients to flow easily.

---

### Q63. What is gradient clipping?
Limiting gradient magnitude to prevent exploding gradients.

---

### Q64. What is L1 regularization?
Adds |weights| penalty to encourage sparsity.

---

### Q65. What is L2 regularization?
Adds squared weight penalty to reduce overfitting.

---

### Q66. What is elastic net regularization?
Combines L1 and L2 penalties.

---

### Q67. What is hyperparameter tuning?
Optimizing parameters like learning rate, batch size, etc.

---

### Q68. What is grid search?
Exhaustively testing parameter combinations.

---

### Q69. What is random search?
Randomly sampling parameter combinations.

---

### Q70. What is Bayesian optimization?
Probabilistic model to optimize hyperparameters efficiently.

---

### Q71. What is Neural Architecture Search (NAS)?
Automatic search for optimal neural network architectures.

---

### Q72. What is pruning in DL?
Removing unnecessary weights/neurons to compress networks.

---

### Q73. What is quantization?
Reducing precision of weights/activations to lower memory usage.

---

### Q74. What is knowledge distillation?
Training a smaller model (student) using a larger model (teacher).

---

### Q75. What is federated learning?
Training models across decentralized devices without sharing raw data.

---

### Q76. What is differential privacy?
Ensuring individual data privacy during model training.

---

### Q77. What is adversarial example?
Slightly perturbed input that fools deep networks.

---

### Q78. What is adversarial training?
Training with adversarial examples for robustness.

---

### Q79. What is explainable AI (XAI)?
Techniques to interpret deep learning models.

---

### Q80. What is SHAP in DL?
SHapley Additive exPlanations — explaining feature importance.

---

### Q81. What is LIME in DL?
Local Interpretable Model-agnostic Explanations.

---

### Q82. What is saliency map?
Visualization showing input regions influencing predictions.

---

### Q83. What is Grad-CAM?
Gradient-weighted Class Activation Mapping for CNN explanations.

---

### Q84. What is t-SNE?
Dimensionality reduction technique for visualization.

---

### Q85. What is PCA?
Principal Component Analysis — reduces dimensionality linearly.

---

### Q86. What is contrastive learning?
Self-supervised learning by contrasting similar and dissimilar pairs.

---

### Q87. What is triplet loss?
Loss encouraging anchor-positive pairs to be closer than anchor-negative.

---

### Q88. What is Siamese network?
Twin networks sharing weights for similarity learning.

---

### Q89. What is capsule network?
Network using capsules to capture part-whole relationships.

---

### Q90. What is graph neural network (GNN)?
Neural network designed to process graph-structured data.

---

### Q91. What is node2vec?
Graph embedding technique for nodes.

---

### Q92. What is attention in GNNs?
Graph attention mechanism for neighbor importance weighting.

---

### Q93. What is temporal deep learning?
DL applied to time series and sequential data.

---

### Q94. What is CNN-LSTM hybrid?
Combining CNN for feature extraction and LSTM for sequence modeling.

---

### Q95. What is multimodal deep learning?
Learning from multiple data types (image + text + audio).

---

### Q96. What is neural style transfer?
Using DL to apply artistic styles to images.

---

### Q97. What is DeepDream?
Visualization technique amplifying network patterns.

---

### Q98. What is reinforcement learning with DL?
Combining deep learning with RL (Deep RL).

---

### Q99. What is deep Q-learning?
Using deep networks to approximate Q-values.

---

### Q100. What is the future of deep learning?
Trends: foundation models, energy-efficient DL, neuro-symbolic AI, and explainable, trustworthy AI.

---
