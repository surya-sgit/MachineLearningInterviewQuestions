# Large Language Models (LLMs) - 100 Interview Questions & Answers

---

### Q1. What is a Large Language Model (LLM)?
An LLM is a deep learning model with billions of parameters trained on vast text data to understand and generate human-like language.

---

### Q2. What are examples of LLMs?
GPT-3, GPT-4, LLaMA, PaLM, Claude, Gemini, Falcon, Mistral, BLOOM.

---

### Q3. What makes a model "large" in LLMs?
The scale of parameters (billions+), large training data, and extensive compute resources.

---

### Q4. What is the architecture used in LLMs?
Most LLMs are based on the **Transformer** architecture, especially decoder-only models.

---

### Q5. What are embeddings in LLMs?
Dense vector representations of tokens, words, or sequences capturing semantic meaning.

---

### Q6. What is tokenization in LLMs?
The process of splitting text into smaller units (tokens) for model input.

---

### Q7. What tokenization methods are common in LLMs?
- Byte Pair Encoding (BPE)  
- WordPiece  
- SentencePiece  
- Byte-level tokenization  

---

### Q8. What is context length in LLMs?
The maximum number of tokens the model can process at once (e.g., 4k, 32k, 1M tokens).

---

### Q9. What is autoregressive generation?
LLMs predict the next token sequentially given prior tokens.

---

### Q10. What is causal masking in LLMs?
A mask ensuring each token only attends to past tokens during training.

---

### Q11. What are parameters in an LLM?
Trainable weights (billions+) controlling how input data is transformed.

---

### Q12. What is pretraining in LLMs?
Training on massive unlabeled text data to learn general language patterns.

---

### Q13. What are common pretraining objectives?
- Causal Language Modeling (CLM)  
- Masked Language Modeling (MLM)  
- Permutation LM (XLNet)  

---

### Q14. What is fine-tuning in LLMs?
Training a pretrained LLM further on domain/task-specific data.

---

### Q15. What is instruction tuning?
Fine-tuning LLMs on instruction-response pairs to follow user prompts.

---

### Q16. What is reinforcement learning with human feedback (RLHF)?
A method where human preferences guide training using reinforcement learning to align outputs.

---

### Q17. What are preference models in RLHF?
Models trained to rank responses by quality, guiding reinforcement learning.

---

### Q18. What is supervised fine-tuning (SFT)?
Fine-tuning LLMs with curated instruction-response datasets.

---

### Q19. What are PEFT methods for LLMs?
Parameter-efficient fine-tuning methods like LoRA, adapters, prefix tuning.

---

### Q20. What is LoRA?
Low-Rank Adaptation — fine-tuning LLMs with small low-rank matrices.

---

### Q21. What is prompt engineering?
Designing input prompts to steer LLM outputs toward desired results.

---

### Q22. What is in-context learning?
Providing few examples in the prompt so LLM adapts without weight updates.

---

### Q23. What is chain-of-thought prompting?
Encouraging models to reason step by step before answering.

---

### Q24. What is self-consistency decoding?
Generating multiple reasoning paths and choosing the most consistent answer.

---

### Q25. What is retrieval-augmented generation (RAG)?
Combining LLMs with external retrieval to ground responses in documents.

---

### Q26. What is knowledge grounding?
Anchoring model outputs with reliable external sources.

---

### Q27. What is hallucination in LLMs?
When the model generates confident but factually incorrect information.

---

### Q28. How to reduce hallucinations in LLMs?
- Retrieval augmentation  
- Fact-checking layers  
- Alignment fine-tuning  
- Better prompts  

---

### Q29. What is a parameter-efficient adaptation?
Updating only small parts of the model instead of all parameters.

---

### Q30. What is zero-shot learning in LLMs?
Performing a task without explicit examples in the prompt.

---

### Q31. What is few-shot learning?
Providing a few examples in context for better performance.

---

### Q32. What is one-shot learning?
Adapting from a single demonstration example in the prompt.

---

### Q33. What is knowledge distillation for LLMs?
Training smaller models (students) to mimic large LLMs (teachers).

---

### Q34. What is quantization in LLMs?
Reducing precision (e.g., FP32 → INT8/4-bit) to compress model.

---

### Q35. What is pruning in LLMs?
Removing redundant weights or neurons to optimize efficiency.

---

### Q36. What is model sparsity?
Encouraging many weights to be zero for efficient computation.

---

### Q37. What is mixture of experts (MoE)?
A model with multiple experts, only activating subsets per input.

---

### Q38. What is Switch Transformer?
An MoE variant that activates only one expert per token.

---

### Q39. What is GShard?
A framework for large-scale distributed MoE training.

---

### Q40. What is distributed training in LLMs?
Splitting model/data across GPUs/TPUs for parallel training.

---

### Q41. What are data parallelism and model parallelism?
- Data parallelism: split data batches across devices  
- Model parallelism: split model layers across devices  

---

### Q42. What is pipeline parallelism?
Splitting model layers across GPUs in a pipeline to optimize training.

---

### Q43. What is tensor parallelism?
Splitting large matrix operations across GPUs.

---

### Q44. What is ZeRO optimization?
Zero Redundancy Optimizer — partitions optimizer states, gradients, and parameters.

---

### Q45. What is gradient checkpointing?
Saving memory by recomputing intermediate activations during backprop.

---

### Q46. What is flash attention?
A memory-efficient attention algorithm for LLMs.

---

### Q47. What is KV caching?
Caching key-value pairs of attention for faster autoregressive decoding.

---

### Q48. What is speculative decoding?
Using a smaller model to propose tokens and larger model to verify.

---

### Q49. What is beam search?
Decoding method maintaining top-k sequences at each step.

---

### Q50. What is nucleus sampling (top-p)?
Sampling from tokens whose cumulative probability exceeds threshold p.

---

### Q51. What is top-k sampling?
Sampling from the top-k most likely tokens.

---

### Q52. What is temperature in decoding?
Scaling logits before softmax to control randomness.

---

### Q53. What is greedy decoding?
Choosing the most probable token at each step deterministically.

---

### Q54. What is alignment in LLMs?
Process of aligning model outputs with human values/preferences.

---

### Q55. What is red-teaming in LLM evaluation?
Actively probing model for harmful or unsafe outputs.

---

### Q56. What are safety layers in LLMs?
Filters and guardrails preventing harmful or biased outputs.

---

### Q57. What is content moderation in LLMs?
Using classifiers or rules to block unsafe generations.

---

### Q58. What are jailbreaks in LLMs?
Attempts to bypass safety restrictions of LLMs.

---

### Q59. What is watermarking in LLM outputs?
Embedding signals in generated text to detect AI-generated content.

---

### Q60. What is detection of AI-generated text?
Using classifiers to distinguish human vs AI text.

---

### Q61. What is evaluation in LLMs?
Measuring performance using benchmarks (accuracy, BLEU, ROUGE, MMLU).

---

### Q62. What is MMLU?
Massive Multitask Language Understanding benchmark for LLMs.

---

### Q63. What is HELM?
Holistic Evaluation of Language Models — evaluates LLMs across scenarios.

---

### Q64. What is BIG-bench?
A large collaborative benchmark for language models.

---

### Q65. What is human evaluation of LLMs?
Humans rating model outputs on correctness, fluency, helpfulness.

---

### Q66. What is perplexity?
A metric measuring how well a model predicts test data.

---

### Q67. What are embeddings in LLMs used for?
Search, clustering, classification, semantic similarity.

---

### Q68. What is vector search?
Retrieving documents using vector similarity of embeddings.

---

### Q69. What is FAISS?
A library for efficient similarity search with embeddings.

---

### Q70. What is semantic search?
Retrieving relevant results using semantic meaning instead of keywords.

---

### Q71. What is RAG pipeline?
Retrieval-Augmented Generation pipeline combining retrieval with LLMs.

---

### Q72. What is LangChain?
A framework for building applications with LLMs and retrieval.

---

### Q73. What is LlamaIndex?
A library for LLM applications with data connectors and indexes.

---

### Q74. What are guardrails in LLM apps?
Frameworks to enforce structured, safe, and controlled responses.

---

### Q75. What is prompt injection?
A malicious attempt to manipulate LLMs through crafted prompts.

---

### Q76. What is data poisoning in LLMs?
Injecting malicious data into training corpora to bias outputs.

---

### Q77. What is model inversion attack?
Inferring sensitive training data from model outputs.

---

### Q78. What is membership inference attack?
Determining whether a sample was in the training dataset.

---

### Q79. What is adversarial prompting?
Crafting prompts to make models produce unsafe responses.

---

### Q80. What is differential privacy in LLM training?
Adding noise to protect individual training data privacy.

---

### Q81. What is federated learning with LLMs?
Training across distributed devices without centralizing data.

---

### Q82. What are open-source LLMs?
Community-released models like LLaMA, Falcon, Mistral, BLOOM.

---

### Q83. What are closed-source LLMs?
Proprietary models like GPT-4, Claude, Gemini.

---

### Q84. What is fine-tuned domain LLM?
LLM trained for specialized domains (biomedical, legal, finance).

---

### Q85. What is multi-modal LLM?
Models trained on multiple modalities (text + image + audio).

---

### Q86. What is GPT-4V?
GPT-4 with vision, able to process images as input.

---

### Q87. What is AudioLM?
An LLM designed for audio and speech modeling.

---

### Q88. What is VideoGPT?
An autoregressive Transformer applied to video frames.

---

### Q89. What is Code LLM?
Models like Codex, CodeLlama specialized for programming tasks.

---

### Q90. What is tool use in LLMs?
LLMs calling APIs, calculators, or external tools for tasks.

---

### Q91. What are agents in LLMs?
LLM-powered systems that plan, reason, and execute multi-step actions.

---

### Q92. What is ReAct prompting?
Combining reasoning and acting steps in prompts for agents.

---

### Q93. What is AutoGPT?
An autonomous LLM agent that iteratively plans and executes tasks.

---

### Q94. What is BabyAGI?
A lightweight autonomous agent framework powered by LLMs.

---

### Q95. What is function calling in LLMs?
Structured output enabling LLMs to call external functions/tools.

---

### Q96. What is structured output parsing?
Parsing LLM outputs into JSON, SQL, or other formats.

---

### Q97. What are system prompts?
Hidden instructions setting model behavior and role.

---

### Q98. What are user prompts?
Direct inputs provided by users for task execution.

---

### Q99. What are assistant prompts?
Model responses guided by system + user prompts.

---

### Q100. What are main applications of LLMs?
- Text generation, summarization, translation  
- Question answering  
- Programming/code generation  
- Chatbots/assistants  
- Knowledge management  
- Scientific discovery  
- Multimodal reasoning  

---
