# Natural Language Processing (NLP) Interview Questions & Answers

---

### Q1. What is NLP?
NLP (Natural Language Processing) is a field of AI that enables machines to understand, interpret, and generate human language.

---

### Q2. What are common applications of NLP?
- Machine translation  
- Chatbots and virtual assistants  
- Sentiment analysis  
- Speech recognition  
- Text summarization  

---

### Q3. What are the main challenges in NLP?
- Ambiguity in language  
- Context understanding  
- Sarcasm detection  
- Multilingual processing  
- Out-of-vocabulary words  

---

### Q4. What is tokenization?
The process of splitting text into smaller units like words, subwords, or sentences.

---

### Q5. What are types of tokenization?
- Word-level  
- Subword-level (BPE, WordPiece)  
- Sentence-level  

---

### Q6. What is stemming?
Reducing words to their root form by chopping off suffixes. Example: “playing” → “play”.

---

### Q7. What is lemmatization?
Reducing words to their base dictionary form using linguistic rules. Example: “better” → “good”.

---

### Q8. What is the difference between stemming and lemmatization?
- **Stemming**: rule-based, crude truncation.  
- **Lemmatization**: uses vocabulary and morphology, more accurate.

---

### Q9. What is stop word removal?
Eliminating common words (like “the”, “is”, “in”) that add little meaning.

---

### Q10. What is POS tagging?
Part-of-Speech tagging assigns grammatical categories (noun, verb, adjective) to words.

---

### Q11. What is Named Entity Recognition (NER)?
Identifying and classifying entities like names, organizations, locations, dates.

---

### Q12. What is chunking in NLP?
Grouping words into syntactic phrases, such as noun phrases (NPs).

---

### Q13. What is dependency parsing?
Analyzing grammatical structure and dependencies between words.

---

### Q14. What is constituency parsing?
Breaking sentences into sub-phrases based on grammatical structure.

---

### Q15. What is Bag of Words (BoW)?
A simple text representation counting word frequencies without considering order.

---

### Q16. What are limitations of BoW?
- Ignores context  
- Ignores word order  
- High dimensionality  

---

### Q17. What is TF-IDF?
Term Frequency-Inverse Document Frequency, a weighting method highlighting important words.

---

### Q18. Why is TF-IDF better than BoW?
It reduces the impact of common words and emphasizes discriminative terms.

---

### Q19. What are word embeddings?
Dense vector representations of words capturing semantic meaning.

---

### Q20. What are examples of word embedding methods?
- Word2Vec  
- GloVe  
- FastText  

---

### Q21. What is Word2Vec?
A model that learns word embeddings using Skip-gram or CBOW.

---

### Q22. What is GloVe?
Global Vectors, a word embedding method trained on co-occurrence statistics.

---

### Q23. What is FastText?
A word embedding method that represents words as character n-grams to handle rare words.

---

### Q24. What is contextual embedding?
Embeddings that change depending on sentence context (e.g., BERT embeddings).

---

### Q25. What is a language model?
A model that predicts the probability of a sequence of words.

---

### Q26. What are types of language models?
- N-gram models  
- Neural language models  
- Transformer-based models  

---

### Q27. What is an n-gram model?
A statistical LM predicting the next word based on the last n-1 words.

---

### Q28. What is perplexity in NLP?
A measure of how well a language model predicts text; lower is better.

---

### Q29. What is smoothing in n-gram models?
Techniques (like Laplace, Kneser-Ney) to handle unseen words.

---

### Q30. What is OOV in NLP?
Out-of-Vocabulary words not seen during training.

---

### Q31. How can OOV be handled?
- Subword tokenization  
- Character-level models  
- FastText embeddings  

---

### Q32. What is a word embedding space?
A vector space where semantically similar words are closer.

---

### Q33. What is cosine similarity?
A metric to measure similarity between two word vectors.

---

### Q34. What is semantic similarity?
Measuring closeness in meaning between texts.

---

### Q35. What is syntactic similarity?
Measuring similarity in structure and grammar.

---

### Q36. What is distributional hypothesis in NLP?
Words with similar contexts tend to have similar meanings.

---

### Q37. What is Latent Semantic Analysis (LSA)?
A technique using matrix factorization (SVD) for dimensionality reduction in text.

---

### Q38. What is Latent Dirichlet Allocation (LDA)?
A topic modeling algorithm that discovers topics in documents.

---

### Q39. What is topic modeling?
Unsupervised method to find hidden topics in text.

---

### Q40. What is text classification?
Assigning predefined categories to text (e.g., spam detection).

---

### Q41. What is sentiment analysis?
Classifying text as positive, negative, or neutral.

---

### Q42. What is text summarization?
Generating concise versions of long text.

---

### Q43. What are types of summarization?
- Extractive  
- Abstractive  

---

### Q44. What is machine translation?
Automatically translating text from one language to another.

---

### Q45. What is BLEU score?
An evaluation metric for machine translation comparing n-grams with references.

---

### Q46. What is ROUGE score?
Evaluation metric for summarization based on recall of overlapping n-grams.

---

### Q47. What is perplexity used for?
Evaluating language models’ predictive power.

---

### Q48. What is speech recognition?
Converting spoken language into text.

---

### Q49. What is speech synthesis?
Generating speech from text (TTS systems).

---

### Q50. What are transformers in NLP?
Models using self-attention mechanisms to process text.

---

### Q51. What is attention mechanism?
A technique to focus on important parts of input sequences.

---

### Q52. What is self-attention?
Attention applied within a sequence to relate different positions.

---

### Q53. What is BERT?
Bidirectional Encoder Representations from Transformers, a pre-trained NLP model.

---

### Q54. What is GPT?
Generative Pre-trained Transformer, an autoregressive language model.

---

### Q55. What is the difference between BERT and GPT?
- BERT: bidirectional, good for understanding tasks.  
- GPT: unidirectional, good for generation.  

---

### Q56. What is masked language modeling?
Training method where words are masked and predicted by the model (used in BERT).

---

### Q57. What is next sentence prediction?
A pretraining task in BERT predicting if one sentence follows another.

---

### Q58. What is fine-tuning in NLP?
Adapting pre-trained models to specific downstream tasks.

---

### Q59. What is zero-shot learning in NLP?
Performing a task without any task-specific training data.

---

### Q60. What is few-shot learning in NLP?
Learning from a small number of labeled examples.

---

### Q61. What is transfer learning in NLP?
Using pre-trained models for new tasks.

---

### Q62. What is multi-lingual NLP?
Models trained to handle multiple languages (e.g., mBERT).

---

### Q63. What is code-switching?
Mixing multiple languages in a single text.

---

### Q64. What is machine reading comprehension?
Answering questions based on provided text passages.

---

### Q65. What is question answering (QA)?
An NLP task of finding answers to user queries.

---

### Q66. What is open-domain QA?
Answering questions from any domain without restrictions.

---

### Q67. What is closed-domain QA?
Answering questions limited to a specific domain.

---

### Q68. What is extractive QA?
Extracting answers directly from text.

---

### Q69. What is abstractive QA?
Generating new sentences as answers.

---

### Q70. What is text entailment?
Determining if one sentence logically follows from another.

---

### Q71. What is coreference resolution?
Identifying when different expressions refer to the same entity.

---

### Q72. What is relation extraction?
Identifying semantic relationships between entities.

---

### Q73. What is event extraction?
Detecting and classifying events from text.

---

### Q74. What is intent detection?
Classifying user intentions in conversational systems.

---

### Q75. What is slot filling?
Extracting specific information from user queries in dialogue systems.

---

### Q76. What are dialogue systems?
Systems that can converse with humans (chatbots, assistants).

---

### Q77. What is the difference between rule-based and ML-based chatbots?
- Rule-based: scripted responses  
- ML-based: learn from data  

---

### Q78. What is a knowledge graph?
A structured representation of entities and their relationships.

---

### Q79. What is semantic parsing?
Mapping natural language to a formal representation (like SQL queries).

---

### Q80. What is information retrieval (IR)?
Finding relevant documents from a large collection.

---

### Q81. What is BM25?
A ranking function used in IR to retrieve relevant documents.

---

### Q82. What is vector search in NLP?
Retrieving documents by comparing embedding vectors.

---

### Q83. What is semantic search?
Search based on meaning rather than keyword matching.

---

### Q84. What is text generation?
Automatically generating human-like text.

---

### Q85. What is autoregressive generation?
Generating text one token at a time conditioned on previous tokens.

---

### Q86. What is beam search?
A decoding strategy exploring multiple candidate sequences.

---

### Q87. What is greedy decoding?
A decoding method choosing the most probable token at each step.

---

### Q88. What is top-k sampling?
Sampling from the top k most probable tokens.

---

### Q89. What is nucleus sampling?
Sampling from tokens whose cumulative probability exceeds a threshold (p).

---

### Q90. What is hallucination in NLP models?
When models generate plausible but incorrect information.

---

### Q91. What is domain adaptation in NLP?
Adapting models to perform well on specific domains.

---

### Q92. What is active learning in NLP?
Selecting informative examples for labeling to improve models efficiently.

---

### Q93. What is semi-supervised learning in NLP?
Training using both labeled and unlabeled data.

---

### Q94. What is unsupervised learning in NLP?
Learning patterns from text without labels (e.g., clustering).

---

### Q95. What is self-supervised learning?
Creating labels from raw data (e.g., masked LM) for pretraining.

---

### Q96. What is knowledge distillation in NLP?
Training smaller student models from large teacher models.

---

### Q97. What is low-resource NLP?
Developing models for languages with limited data.

---

### Q98. What is multilingual transfer learning?
Using models trained on one language to benefit others.

---

### Q99. What is ethical concern in NLP?
Bias, misinformation, privacy, and fairness in language models.

---

### Q100. Summarize NLP in one line.
NLP bridges human language and machines by enabling understanding, generation, and interaction with text and speech.

---
