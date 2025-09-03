# Transformers / Attention Models - 100 Interview Questions & Answers

---

### Q1. What is the Transformer architecture?
The Transformer is a deep learning architecture introduced in *"Attention is All You Need"* (2017). It relies entirely on **self-attention** mechanisms, removing recurrence and convolutions, making it highly parallelizable.

---

### Q2. What are the main components of the Transformer?
- Encoder
- Decoder
- Multi-Head Self-Attention
- Position-wise Feed-Forward Layers
- Positional Encoding
- Layer Normalization & Residual Connections

---

### Q3. Why are Transformers faster than RNNs?
Transformers process sequences **in parallel** using attention, while RNNs process sequentially, limiting parallelism.

---

### Q4. What is Self-Attention?
Self-Attention allows each token to attend to all other tokens in a sequence, learning contextual dependencies dynamically.

---

### Q5. What are Queries, Keys, and Values in Attention?
- **Query (Q):** Representation of the token seeking context  
- **Key (K):** Representation of tokens being compared  
- **Value (V):** Information retrieved when Query matches Key  

---

### Q6. How is the Attention score calculated?
\[
\text{Attention}(Q,K,V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V
\]

---

### Q7. Why use the scaling factor \(\sqrt{d_k}\)?
To prevent large dot products when \(d_k\) is large, which could push softmax into regions with very small gradients.

---

### Q8. What is Multi-Head Attention?
Instead of one attention head, multiple heads learn diverse representations, improving model expressiveness.

---

### Q9. Why do we need positional encoding?
Transformers lack inherent sequence order (unlike RNNs). Positional encoding injects order information into embeddings.

---

### Q10. What is sinusoidal positional encoding?
It encodes positions using sine and cosine functions at different frequencies, enabling extrapolation to longer sequences.

---

### Q11. What are learned positional embeddings?
Instead of sinusoidal functions, positions are learned as trainable embedding vectors.

---

### Q12. What is the encoder in a Transformer?
The encoder maps an input sequence to contextual representations using stacked self-attention + feedforward layers.

---

### Q13. What is the decoder in a Transformer?
The decoder generates output sequence autoregressively, attending to encoder outputs and past predictions.

---

### Q14. What is Masked Self-Attention in the decoder?
A mask prevents attending to future tokens during training, ensuring causality.

---

### Q15. Why are residual connections used?
They allow gradients to flow more easily and prevent vanishing gradient issues in deep models.

---

### Q16. What is Layer Normalization in Transformers?
Normalization applied across feature dimensions, stabilizing training and speeding convergence.

---

### Q17. What is the Feed-Forward Network (FFN) in Transformers?
A 2-layer MLP applied independently to each token’s representation.

---

### Q18. What are the advantages of Transformers over RNNs?
- Parallelism  
- Better long-range dependency modeling  
- Scalability to large datasets  

---

### Q19. What is the main limitation of Transformers?
Quadratic complexity in attention computation (\(O(n^2)\)) with sequence length.

---

### Q20. What are some solutions to Transformer’s quadratic complexity?
- Sparse Attention  
- Linformer  
- Performer (linear attention)  
- Longformer / BigBird  

---

### Q21. What are Encoder-only Transformers?
Models like **BERT** used for understanding tasks (classification, QA, NER).

---

### Q22. What are Decoder-only Transformers?
Models like **GPT** used for generative tasks (text generation, completion).

---

### Q23. What are Encoder-Decoder Transformers?
Models like **T5** and **BART**, used for seq2seq tasks (translation, summarization).

---

### Q24. What is BERT?
Bidirectional Encoder Representations from Transformers — pre-trained encoder-only model.

---

### Q25. What is GPT?
Generative Pre-trained Transformer — autoregressive, decoder-only model.

---

### Q26. What is T5?
Text-to-Text Transfer Transformer — treats every NLP task as text-to-text generation.

---

### Q27. What is BART?
Bidirectional Auto-Regressive Transformers — combines BERT-like encoder and GPT-like decoder.

---

### Q28. What is XLNet?
A Transformer variant that learns bidirectional context using permutation-based training.

---

### Q29. What is ALBERT?
A lightweight BERT with parameter sharing and factorized embeddings.

---

### Q30. What is DistilBERT?
A distilled, smaller version of BERT for faster inference with minimal performance drop.

---

### Q31. What is Electra?
A pre-trained model where a discriminator learns to distinguish real vs replaced tokens (more efficient than MLM).

---

### Q32. What is RoBERTa?
A robustly optimized BERT with longer training, more data, and dynamic masking.

---

### Q33. What is GPT-2?
A large autoregressive Transformer trained on web text for text generation.

---

### Q34. What is GPT-3?
A 175B parameter model enabling few-shot, zero-shot, and one-shot learning.

---

### Q35. What is GPT-4?
A more advanced multimodal model (text + images) with improved reasoning and alignment.

---

### Q36. What is attention masking?
Masks certain tokens (padding or future positions) to avoid attending incorrectly.

---

### Q37. What is cross-attention?
In decoder, queries come from decoder states, keys and values from encoder outputs.

---

### Q38. What is the role of the embedding layer?
Maps tokens into dense vector representations for model processing.

---

### Q39. What is subword tokenization in Transformers?
Splitting words into smaller units (BPE, WordPiece, SentencePiece) to handle rare words.

---

### Q40. What is Byte Pair Encoding (BPE)?
A tokenization method merging most frequent pairs of characters/subwords iteratively.

---

### Q41. What is WordPiece?
A subword tokenization used in BERT.

---

### Q42. What is SentencePiece?
A language-independent tokenization method treating input as raw bytes.

---

### Q43. What is masked language modeling (MLM)?
A pretraining task where random tokens are masked and model predicts them (used in BERT).

---

### Q44. What is causal language modeling?
Autoregressive task predicting the next token given previous tokens (used in GPT).

---

### Q45. What is Next Sentence Prediction (NSP)?
BERT pretraining objective predicting if one sentence follows another.

---

### Q46. Why was NSP criticized?
It adds limited benefit, and alternatives (like sentence order prediction) perform better.

---

### Q47. What is span corruption pretraining?
Replacing spans of text with a single mask token (used in T5).

---

### Q48. What is denoising autoencoding in Transformers?
Corrupt input and train model to reconstruct original (used in BART).

---

### Q49. What is pretraining-finetuning paradigm?
Pretrain on large corpus → fine-tune on specific downstream tasks.

---

### Q50. What is zero-shot learning in Transformers?
Performing tasks without task-specific fine-tuning, relying on prompt engineering.

---

### Q51. What is few-shot learning?
Providing few examples in context to guide model predictions.

---

### Q52. What is one-shot learning?
Performing tasks from a single example demonstration.

---

### Q53. What is transfer learning in Transformers?
Using pre-trained representations on new downstream tasks.

---

### Q54. What is prompt engineering?
Designing input prompts to steer Transformer outputs.

---

### Q55. What are soft prompts?
Learnable continuous embeddings prepended to input rather than manual text prompts.

---

### Q56. What is prefix tuning?
Freezing model weights and training only small prefix embeddings for tasks.

---

### Q57. What is LoRA (Low-Rank Adaptation)?
Parameter-efficient fine-tuning method that inserts low-rank matrices into model layers.

---

### Q58. What is PEFT (Parameter-Efficient Fine-Tuning)?
Framework including LoRA, prefix tuning, adapters for efficient task adaptation.

---

### Q59. What are adapters in Transformers?
Small bottleneck layers added between Transformer layers for task-specific tuning.

---

### Q60. What is knowledge distillation in Transformers?
Training a smaller student model to mimic outputs of a large teacher model.

---

### Q61. What is pruning in Transformers?
Removing less important weights/neurons to reduce model size and inference cost.

---

### Q62. What is quantization in Transformers?
Reducing precision (e.g., FP32 → INT8) to shrink memory and speed inference.

---

### Q63. What is mixed precision training?
Using FP16 + FP32 computations to speed training and reduce memory usage.

---

### Q64. What is gradient checkpointing?
Trading compute for memory by recomputing activations during backpropagation.

---

### Q65. What is attention visualization?
Heatmaps showing which tokens attend to which, helping interpretability.

---

### Q66. What is explainability in Transformers?
Methods like attention analysis, probing, SHAP, LIME to interpret model predictions.

---

### Q67. What is fine-tuning vs feature extraction?
- Fine-tuning: train entire model on downstream task  
- Feature extraction: freeze pretrained layers, use embeddings  

---

### Q68. What is domain adaptation in Transformers?
Adapting models to specific domain data (biomedical, legal, finance).

---

### Q69. What is continual learning?
Training models incrementally without forgetting previous tasks.

---

### Q70. What is catastrophic forgetting?
When fine-tuning causes forgetting of previous knowledge.

---

### Q71. What is multilingual Transformers?
Models like mBERT, XLM-R trained on multiple languages for cross-lingual tasks.

---

### Q72. What is cross-lingual transfer?
Using multilingual models to perform tasks in low-resource languages.

---

### Q73. What is code-switching?
Mixing multiple languages in a single sequence, handled by multilingual Transformers.

---

### Q74. What is multimodal Transformers?
Models combining text, vision, audio (e.g., CLIP, Flamingo, GPT-4).

---

### Q75. What is CLIP?
Contrastive Language-Image Pretraining — aligns images and text in shared space.

---

### Q76. What is Vision Transformer (ViT)?
Applies Transformer architecture to image patches instead of text tokens.

---

### Q77. What is DETR?
Detection Transformer — object detection model using Transformers.

---

### Q78. What is Speech Transformer?
Applies Transformer architecture to speech recognition.

---

### Q79. What is Perceiver?
A general-purpose Transformer for multimodal inputs.

---

### Q80. What is Transformer-XL?
Adds recurrence and relative position encoding for longer sequences.

---

### Q81. What is Reformer?
Efficient Transformer using locality-sensitive hashing for attention.

---

### Q82. What is Linformer?
Approximates self-attention with low-rank projections for efficiency.

---

### Q83. What is Longformer?
Sparse attention pattern enabling long context windows.

---

### Q84. What is BigBird?
Transformer with block + random + global sparse attention, scalable to long text.

---

### Q85. What is Performer?
Linear attention using kernel methods, reducing attention complexity.

---

### Q86. What is Routing Transformer?
Uses locality-sensitive hashing routing for sparse attention.

---

### Q87. What is Switch Transformer?
Mixture-of-Experts Transformer with conditional computation.

---

### Q88. What is GShard?
Large-scale training framework for Mixture-of-Experts Transformers.

---

### Q89. What is FNet?
Replaces self-attention with Fourier Transform for efficiency.

---

### Q90. What is ALiBi?
Attention with Linear Biases, a positional encoding method.

---

### Q91. What is Rotary Positional Embedding (RoPE)?
A rotational embedding method for better generalization across sequence lengths.

---

### Q92. What is relative positional encoding?
Encoding relative distances instead of absolute positions.

---

### Q93. What is retrieval-augmented Transformers?
Augmenting model with external knowledge retrieval (e.g., RAG).

---

### Q94. What is memory-augmented Transformers?
Adding external memory modules for longer context.

---

### Q95. What is efficient Transformer inference?
Techniques like kv-caching, quantization, batching for fast inference.

---

### Q96. What is beam search in Transformers?
Decoding method exploring multiple candidate sequences at each step.

---

### Q97. What is nucleus sampling?
Sampling decoding method with top-p probability mass.

---

### Q98. What is temperature in decoding?
Scaling logits before softmax to control randomness.

---

### Q99. What is reinforcement learning with Transformers?
Using RL (like RLHF) to align Transformers with human feedback.

---

### Q100. What are the main applications of Transformers?
- NLP: translation, summarization, QA, text generation  
- Vision: classification, detection, segmentation  
- Multimodal AI: CLIP, GPT-4  
- Speech: ASR, TTS  
- Bioinformatics, code, recommender systems  

---
