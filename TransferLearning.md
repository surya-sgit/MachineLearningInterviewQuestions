# Transfer Learning Interview Questions & Answers

---

### Q1. What is transfer learning?
Transfer learning is a machine learning technique where a model developed for one task is reused as the starting point for another related task.

---

### Q2. Why is transfer learning important?
It reduces training time, lowers data requirements, and improves performance, especially in domains with limited labeled data.

---

### Q3. Give an example of transfer learning in practice.
Using ImageNet-pretrained CNNs like ResNet for medical image classification.

---

### Q4. What are common applications of transfer learning?
- Computer vision (image classification, detection)  
- NLP (text classification, translation)  
- Speech recognition  
- Healthcare predictions  

---

### Q5. What are the benefits of transfer learning?
- Faster convergence  
- Requires less labeled data  
- Improves accuracy  
- Leverages existing knowledge  

---

### Q6. What are the challenges of transfer learning?
- Negative transfer  
- Domain mismatch  
- Large pre-trained models can be expensive  
- Need for fine-tuning strategies  

---

### Q7. What is negative transfer?
When knowledge from the source task hurts performance on the target task.

---

### Q8. How to avoid negative transfer?
- Choose related source and target domains  
- Use domain adaptation  
- Perform feature selection  

---

### Q9. What are the types of transfer learning?
- Inductive  
- Transductive  
- Unsupervised  

---

### Q10. Explain inductive transfer learning.
The target task is different but related to the source task, with labeled target data (e.g., fine-tuning BERT on sentiment analysis).

---

### Q11. Explain transductive transfer learning.
The source and target tasks are the same, but the domains differ (e.g., applying English-trained model to Spanish text).

---

### Q12. Explain unsupervised transfer learning.
No labeled data in source and target; knowledge is transferred via unsupervised methods like embeddings.

---

### Q13. What is feature extraction in transfer learning?
Using the pre-trained model as a fixed feature extractor, training only a new classifier on top.

---

### Q14. What is fine-tuning in transfer learning?
Unfreezing some or all layers of the pre-trained model and retraining them on target data.

---

### Q15. When to use feature extraction vs fine-tuning?
- Feature extraction: small target dataset  
- Fine-tuning: large target dataset and related domain  

---

### Q16. What are domain adaptation techniques in transfer learning?
- Domain adversarial training  
- Instance reweighting  
- Feature alignment  

---

### Q17. What is cross-domain transfer learning?
Applying models trained on one domain to another (e.g., natural images → medical images).

---

### Q18. What is multi-task learning vs transfer learning?
- Multi-task: train multiple tasks simultaneously.  
- Transfer learning: sequentially transfer knowledge from one task to another.  

---

### Q19. What is zero-shot transfer learning?
Model solves a new task without explicit training examples (e.g., GPT answering questions without labeled QA data).

---

### Q20. What is few-shot transfer learning?
Model adapts to a task with only a few labeled examples.

---

### Q21. How does BERT use transfer learning?
Pretrained on large text corpora (masked LM, NSP), then fine-tuned for specific NLP tasks.

---

### Q22. How does GPT use transfer learning?
Pretrained autoregressively on large text data, then adapted for generation or instruction tasks.

---

### Q23. What is pretraining?
Training a model on a large dataset to capture general features.

---

### Q24. What is fine-tuning?
Adapting a pretrained model to a specific target task with labeled data.

---

### Q25. Why does transfer learning work in deep learning?
Because deep models learn general features in earlier layers that are useful across tasks.

---

### Q26. What are frozen layers in transfer learning?
Layers whose weights are kept fixed during training on the target task.

---

### Q27. Why freeze early layers in CNN transfer learning?
Early layers capture general features (edges, textures) that transfer across domains.

---

### Q28. What is catastrophic forgetting?
When fine-tuning causes the model to lose useful knowledge from the source task.

---

### Q29. How to prevent catastrophic forgetting?
- Use smaller learning rates  
- Freeze layers  
- Use regularization techniques  

---

### Q30. What is domain generalization?
Training a model that generalizes well to unseen domains without adaptation.

---

### Q31. What is transfer learning in reinforcement learning?
Using knowledge from one environment to improve learning in another.

---

### Q32. What is model distillation in transfer learning?
Transferring knowledge from a large model (teacher) to a smaller one (student).

---

### Q33. How is transfer learning used in speech processing?
Pretraining acoustic models on large speech corpora and fine-tuning for speaker/language-specific tasks.

---

### Q34. How is transfer learning used in text summarization?
Using pretrained models like BART or T5 fine-tuned on summarization datasets.

---

### Q35. What is task-specific fine-tuning?
Adapting a pretrained model to one specific target task.

---

### Q36. What is multi-task fine-tuning?
Fine-tuning on multiple target tasks jointly to improve generalization.

---

### Q37. What is progressive neural network transfer learning?
Using lateral connections to retain prior knowledge while learning new tasks.

---

### Q38. What is sequential transfer learning?
Learning tasks in sequence, transferring knowledge step by step.

---

### Q39. What is cross-lingual transfer learning?
Adapting models trained in one language to work in others.

---

### Q40. Give an example of cross-lingual transfer.
mBERT pretrained on 100 languages applied to zero-shot tasks in low-resource languages.

---

### Q41. What is meta-learning?
Learning how to learn; improves transfer to new tasks with few samples.

---

### Q42. How is meta-learning related to transfer learning?
Meta-learning facilitates faster adaptation in transfer learning settings.

---

### Q43. What is fine-tuning with discriminative learning rates?
Using different learning rates for different layers to control how much they adapt.

---

### Q44. What is gradual unfreezing?
Unfreezing pretrained layers progressively during fine-tuning.

---

### Q45. What is feature alignment?
Aligning source and target feature distributions for transfer.

---

### Q46. What is adversarial domain adaptation?
Using adversarial training to minimize domain differences.

---

### Q47. What is few-shot classification in transfer learning?
Adapting to classify unseen classes with very few examples.

---

### Q48. What is one-shot learning?
Learning from only a single example per class.

---

### Q49. What is zero-shot classification?
Classifying without seeing any labeled data from the target classes.

---

### Q50. What is universal representation in transfer learning?
Learning embeddings/features applicable across multiple domains and tasks.

---

### Q51. What are pretrained CNN models commonly used for transfer learning?
- VGG  
- ResNet  
- Inception  
- EfficientNet  

---

### Q52. What are pretrained NLP models used for transfer learning?
- BERT  
- GPT  
- RoBERTa  
- XLNet  
- T5  

---

### Q53. What are checkpoints in transfer learning?
Saved weights from pretrained models used for fine-tuning.

---

### Q54. What is sequential domain adaptation?
Adapting across multiple intermediate domains before final target.

---

### Q55. What is feature reuse in transfer learning?
Reusing learned features without retraining them.

---

### Q56. What is model fine-tuning in CV vs NLP?
- CV: usually unfreeze last layers.  
- NLP: often unfreeze most layers for downstream tasks.  

---

### Q57. What is domain shift?
Distribution difference between source and target datasets.

---

### Q58. How to detect domain shift?
By measuring divergence in feature distributions or performance drop.

---

### Q59. What is cross-domain sentiment analysis?
Transferring sentiment models trained on one domain (e.g., product reviews) to another (e.g., movie reviews).

---

### Q60. What is few-shot transfer with prompts?
Using prompt engineering to adapt LLMs without task-specific fine-tuning.

---

### Q61. What is zero-shot transfer with prompts?
Using natural language instructions as prompts to perform unseen tasks.

---

### Q62. What is instruction tuning?
Fine-tuning models with task instructions to improve transfer.

---

### Q63. What is parameter-efficient transfer learning (PETL)?
Adapting large models with fewer trainable parameters (e.g., LoRA, adapters).

---

### Q64. What is LoRA?
Low-Rank Adaptation, a PETL method for large models.

---

### Q65. What are adapters in transfer learning?
Small bottleneck layers added to pretrained models for task-specific adaptation.

---

### Q66. What is prompt tuning?
Learning soft prompts for guiding pretrained models without full fine-tuning.

---

### Q67. What is prefix tuning?
Learning task-specific prefixes for transformers instead of modifying all parameters.

---

### Q68. What is feature-based transfer vs parameter-based transfer?
- Feature-based: reuse learned embeddings.  
- Parameter-based: reuse model weights.  

---

### Q69. What is fine-tuning vs domain adaptation?
- Fine-tuning: labeled target task.  
- Domain adaptation: unlabeled or mismatched domains.  

---

### Q70. What is inductive bias in transfer learning?
Assumptions transferred from source task to guide target learning.

---

### Q71. What is cross-modal transfer learning?
Transferring knowledge across modalities (e.g., vision ↔ text).

---

### Q72. Give an example of cross-modal transfer.
CLIP (image-text joint embeddings).

---

### Q73. What is multimodal transfer learning?
Combining multiple modalities (text, vision, audio) for knowledge transfer.

---

### Q74. What is continual transfer learning?
Adapting continuously to new tasks/domains without forgetting past ones.

---

### Q75. What is life-long learning in transfer learning?
Models accumulate knowledge across many tasks for future transfer.

---

### Q76. What is domain-invariant representation?
Features that are consistent across different domains.

---

### Q77. What is task similarity in transfer learning?
How related source and target tasks are; influences success.

---

### Q78. What is unsupervised domain adaptation?
Adapting to target domain without target labels.

---

### Q79. What is semi-supervised domain adaptation?
Using a few labeled target samples along with unlabeled data.

---

### Q80. What is active transfer learning?
Selecting most informative target samples for labeling.

---

### Q81. What is curriculum transfer learning?
Gradually transferring knowledge from easier to harder tasks.

---

### Q82. What is transfer learning in recommendation systems?
Using pretrained embeddings from one domain to improve recommendations in another.

---

### Q83. What is pretext task in transfer learning?
An auxiliary self-supervised task to learn general features.

---

### Q84. Give an example of a pretext task.
Predicting missing words (masked LM) or predicting rotation in images.

---

### Q85. What is fine-tuning with early stopping?
Stopping adaptation early to avoid overfitting target data.

---

### Q86. What is selective fine-tuning?
Only fine-tuning a subset of layers for efficiency.

---

### Q87. What is head tuning?
Fine-tuning only the classification head on top of frozen features.

---

### Q88. What is hybrid transfer learning?
Combining feature extraction and partial fine-tuning.

---

### Q89. What is progressive resizing in transfer learning?
Training with small images first, then larger ones to improve efficiency.

---

### Q90. What is data augmentation’s role in transfer learning?
Expands target dataset to reduce overfitting.

---

### Q91. What is transfer learning in low-resource languages?
Using multilingual pretrained models to adapt for languages with little data.

---

### Q92. What is universal transfer learning?
Models designed to generalize across multiple tasks and domains.

---

### Q93. What is representation learning in transfer learning?
Learning reusable feature representations.

---

### Q94. What is few-shot prompting in transfer learning?
Guiding LLMs using a few examples in the prompt.

---

### Q95. What is chain-of-thought prompting in transfer learning?
Providing reasoning steps in prompts for better transfer.

---

### Q96. What is in-context learning?
LLMs adapting to tasks based on examples provided in prompts without parameter updates.

---

### Q97. What is self-supervised pretraining in transfer learning?
Using unlabeled data to create auxiliary tasks for pretraining.

---

### Q98. What is multi-source transfer learning?
Using multiple source domains for improved target performance.

---

### Q99. What are risks of large-scale transfer learning?
Bias amplification, data privacy issues, compute cost.

---

### Q100. Summarize transfer learning in one line.
Transfer learning reuses knowledge from one task/domain to accelerate and improve performance on new tasks.

---
