# LSTM & GRU Interview Questions & Answers (1–100)

---

### Q1. What is an LSTM?
LSTM (Long Short-Term Memory) is a type of RNN designed to capture long-term dependencies using memory cells and gating mechanisms.

---

### Q2. What problem does LSTM solve?
LSTMs mitigate the vanishing/exploding gradient problem in vanilla RNNs, enabling learning of long-range dependencies.

---

### Q3. What are the gates in LSTM?
- **Input gate**: Controls how much new info enters the cell.
- **Forget gate**: Decides what to discard from memory.
- **Output gate**: Controls how much memory is exposed as hidden state.

---

### Q4. What is the cell state in LSTM?
The cell state carries information across time steps with minimal modifications, serving as long-term memory.

---

### Q5. What is a GRU?
GRU (Gated Recurrent Unit) is a simplified LSTM with only reset and update gates, merging cell and hidden states.

---

### Q6. Compare LSTM vs GRU.
- **LSTM**: More expressive, better for complex tasks.
- **GRU**: Faster, fewer parameters, suitable for smaller datasets.

---

### Q7. When to use GRU instead of LSTM?
When computation resources are limited or dataset is small/medium in size.

---

### Q8. What is the forget gate used for?
To control which past information should be discarded from the cell state.

---

### Q9. What are peephole connections?
Connections where gates access the cell state directly, improving performance in timing-sensitive tasks.

---

### Q10. Why are GRUs faster than LSTMs?
Because they have fewer gates and no separate cell state, reducing computations.

---

### Q11. What is sequence-to-sequence with LSTM?
An encoder LSTM encodes input into a context vector, and a decoder LSTM generates the output sequence.

---

### Q12. What are applications of LSTMs?
Speech recognition, language modeling, translation, text generation, and time-series forecasting.

---

### Q13. What are applications of GRUs?
Similar to LSTMs but preferred in real-time/low-latency systems due to efficiency.

---

### Q14. What is vanishing gradient mitigation in LSTM?
Cell states with additive updates reduce exponential decay of gradients.

---

### Q15. Why is gradient clipping used with LSTMs?
To prevent exploding gradients during backpropagation through time.

---

### Q16. What is truncated BPTT in LSTMs?
Limiting backpropagation to a fixed number of time steps for efficiency.

---

### Q17. How do you initialize LSTM states?
Typically with zeros, but learnable initial states are possible.

---

### Q18. What are stacked LSTMs?
Multiple LSTM layers stacked for deeper representations.

---

### Q19. What is bidirectional LSTM?
Processes input sequence both forward and backward to capture past and future context.

---

### Q20. What are residual LSTMs?
LSTMs with skip connections to help gradient flow in deep networks.

---

### Q21. What is an LSTM cell equation?
Key equations involve input, forget, output gates and cell update using `sigmoid` and `tanh`.

---

### Q22. What is the update gate in GRU?
Controls how much past information to keep versus new information.

---

### Q23. What is the reset gate in GRU?
Controls how much past information to forget when computing new candidate hidden state.

---

### Q24. Why is GRU better for small datasets?
Fewer parameters reduce overfitting risk.

---

### Q25. What is attention in seq2seq LSTMs?
Allows decoder to attend to all encoder hidden states, not just the final context vector.

---

### Q26. How do LSTMs handle variable sequence lengths?
Using padding and masking.

---

### Q27. What is masking in LSTM training?
Ignoring padded tokens so they don’t influence learning.

---

### Q28. What is an encoder-decoder LSTM?
An architecture where encoder LSTM compresses input sequence and decoder LSTM generates output sequence.

---

### Q29. What is the main difference between CNNs and LSTMs?
CNNs excel at spatial data, LSTMs excel at sequential/temporal data.

---

### Q30. What are LSTMs used for in finance?
Stock prediction, fraud detection, and transaction modeling.

---

### Q31. What are LSTMs used for in healthcare?
Disease progression prediction, ECG/EEG signal analysis, and patient monitoring.

---

### Q32. What is teacher forcing in LSTMs?
Using ground truth tokens as decoder input during training.

---

### Q33. What is exposure bias?
When the model performs poorly at inference due to relying on teacher forcing during training.

---

### Q34. What is scheduled sampling?
Gradually replacing ground-truth tokens with model predictions during training.

---

### Q35. What is a character-level LSTM?
An LSTM trained at character level instead of word level.

---

### Q36. What are hierarchical LSTMs?
LSTMs operating at multiple levels (e.g., word-level and sentence-level).

---

### Q37. What is attention vs self-attention in LSTM models?
- **Attention**: Focuses on encoder hidden states.
- **Self-attention**: Token attends to all tokens in same sequence.

---

### Q38. What are memory networks with LSTM?
Combining external memory with LSTMs for enhanced reasoning.

---

### Q39. What is an LSTM autoencoder?
An LSTM that compresses sequences into latent vectors and reconstructs them.

---

### Q40. What are applications of LSTM autoencoders?
Anomaly detection, sequence reconstruction, and dimensionality reduction.

---

### Q41. How do LSTMs perform time-series forecasting?
By modeling sequential dependencies and predicting future values.

---

### Q42. What is the role of dropout in LSTMs?
To reduce overfitting by randomly dropping neurons.

---

### Q43. What is variational dropout in LSTMs?
Applying the same dropout mask across all time steps.

---

### Q44. What is zoneout in LSTMs?
A regularization technique where hidden states are randomly preserved.

---

### Q45. What is gradient clipping in GRUs?
Bounding gradients to stabilize training.

---

### Q46. What are applications of GRUs in NLP?
Language modeling, translation, and sentiment analysis.

---

### Q47. What is the role of `sigmoid` in LSTM/GRU gates?
To squash values between 0 and 1, representing proportions.

---

### Q48. What is the role of `tanh` in LSTM/GRU?
To regulate new candidate values between -1 and 1.

---

### Q49. What are contextual embeddings from LSTMs?
Embeddings generated dynamically from hidden states, context-aware.

---

### Q50. What is perplexity in LSTM language models?
A measure of prediction quality; lower perplexity is better.

---

### Q51. What is beam search in LSTM decoding?
A search strategy keeping top-k candidate sequences.

---

### Q52. What is greedy decoding?
Selecting the highest probability token at each step.

---

### Q53. What is coverage in attention-based LSTMs?
A mechanism tracking which parts of input have been attended to.

---

### Q54. What is copy mechanism?
Allowing LSTM models to directly copy words from input to output.

---

### Q55. What are pointer networks with LSTM?
Networks that output positions in the input sequence instead of tokens.

---

### Q56. What are hierarchical attention LSTMs?
LSTMs with attention applied at multiple levels (e.g., word/sentence).

---

### Q57. What is reinforcement learning with LSTMs?
Training LSTMs using reward signals, e.g., in text generation.

---

### Q58. What are contextual GRUs in recommender systems?
GRUs modeling user sessions for recommendations.

---

### Q59. How do GRUs handle missing data?
Variants like GRU-D incorporate decay mechanisms for missing values.

---

### Q60. What is GRU-D?
A GRU designed for irregular time-series with missing values.

---

### Q61. What are dilated LSTMs?
LSTMs where connections skip time steps for long-term memory.

---

### Q62. What is hierarchical softmax in LSTMs?
A tree-based approximation for efficient softmax over large vocabularies.

---

### Q63. What is dynamic unrolling in LSTMs?
Unrolling only up to actual sequence length instead of fixed length.

---

### Q64. What is curriculum learning in LSTMs?
Training with simple sequences first, then harder ones.

---

### Q65. What is a residual GRU?
A GRU with skip connections for deeper architectures.

---

### Q66. What is an LSTM with attention?
A seq2seq model where attention improves alignment.

---

### Q67. What is self-critical sequence training?
An RL-based training approach for sequence generation tasks.

---

### Q68. What are quantized LSTMs?
LSTMs with compressed weights for efficient inference.

---

### Q69. What is pruning in LSTMs?
Removing unnecessary weights or neurons for model compression.

---

### Q70. What is a lightweight GRU?
Simplified GRU for mobile/edge applications.

---

### Q71. What is neural Turing machine with LSTM?
LSTM augmented with external memory and read/write heads.

---

### Q72. What is attention visualization in LSTMs?
Visualizing attention weights to interpret model focus.

---

### Q73. What is transfer learning with LSTMs?
Reusing pretrained LSTM layers for new tasks.

---

### Q74. What are pretrained LSTM language models?
LSTMs trained on large corpora used for fine-tuning.

---

### Q75. What is the role of embeddings in LSTMs?
Dense representations of words improve training efficiency.

---

### Q76. What is fine-tuning an LSTM?
Adjusting pretrained weights on a new dataset.

---

### Q77. What is multilingual LSTM?
LSTMs trained on multiple languages for cross-lingual tasks.

---

### Q78. What are hybrid CNN-LSTM models?
CNN extracts local features, LSTM models temporal dependencies.

---

### Q79. What are hybrid RNN-LSTM models?
Combining RNNs and LSTMs for hierarchical sequence modeling.

---

### Q80. What are LSTM ensembles?
Combining multiple LSTMs for better generalization.

---

### Q81. What is adversarial training for LSTMs?
Training LSTMs to resist adversarial perturbations.

---

### Q82. What is dropout in GRUs?
Randomly deactivating units to prevent overfitting.

---

### Q83. What is recurrent batch normalization?
Normalizing hidden activations in recurrent layers.

---

### Q84. What is layer normalization in LSTMs?
Normalizing across features to stabilize training.

---

### Q85. What is online training for LSTMs?
Updating model after each sequence.

---

### Q86. What is offline training for LSTMs?
Training using full dataset batches.

---

### Q87. What is real-time inference with GRUs?
Deploying GRUs for streaming data like speech.

---

### Q88. What is streaming LSTM?
LSTM processing continuous sequences without resetting states.

---

### Q89. What are hierarchical LSTMs for dialogue?
Two-level LSTMs: one for utterances, one for conversations.

---

### Q90. What is reinforcement dialogue generation with LSTM?
Using RL to optimize dialogue rewards.

---

### Q91. What is a continuous-time LSTM?
LSTMs adapted to irregular intervals using differential equations.

---

### Q92. What are Neural ODE LSTMs?
Continuous-time LSTM variants with ODE dynamics.

---

### Q93. What is a multiplicative LSTM?
LSTMs with input-modulated recurrent connections.

---

### Q94. What is an echo state LSTM?
LSTM variant inspired by reservoir computing.

---

### Q95. What are reservoir LSTMs?
LSTMs with fixed recurrent dynamics and trained output layers.

---

### Q96. What is a memory-augmented GRU?
A GRU with external memory for enhanced sequence modeling.

---

### Q97. What is the future of LSTMs?
Still useful for small-scale and streaming tasks, though Transformers dominate.

---

### Q98. What is the future of GRUs?
Likely to remain relevant in real-time and low-latency tasks.

---

### Q99. How do LSTMs compare with Transformers?
Transformers parallelize better; LSTMs are lighter and stream-friendly.

---

### Q100. How do GRUs compare with Transformers?
GRUs are efficient but less powerful; Transformers dominate large-scale NLP.
