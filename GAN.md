# Generative Adversarial Networks (GANs) Interview Questions & Answers (1–100)

---

### Q1. What is a GAN?
A Generative Adversarial Network (GAN) consists of two neural networks, a **Generator** and a **Discriminator**, competing in a minimax game to generate realistic data.

---

### Q2. Who introduced GANs and when?
GANs were introduced by Ian Goodfellow and colleagues in **2014**.

---

### Q3. What is the role of the Generator in GANs?
The generator creates synthetic data that attempts to mimic real data distribution.

---

### Q4. What is the role of the Discriminator in GANs?
The discriminator evaluates whether input data is real (from dataset) or fake (from generator).

---

### Q5. What is the training objective of GANs?
A minimax game:
\[
\min_G \max_D V(D,G) = E_{x~p_{data}}[\log D(x)] + E_{z~p_z}[\log(1-D(G(z)))]
\]

---

### Q6. What is mode collapse in GANs?
When the generator produces limited diversity of samples, ignoring some data modes.

---

### Q7. What is vanishing gradient problem in GANs?
When the discriminator becomes too strong, gradients for the generator diminish.

---

### Q8. How to mitigate vanishing gradients in GANs?
Use **Wasserstein loss**, label smoothing, or balanced training.

---

### Q9. What is Wasserstein GAN (WGAN)?
A GAN variant using Wasserstein distance for stable training.

---

### Q10. What is gradient penalty in WGAN-GP?
A penalty ensuring Lipschitz constraint, stabilizing GAN training.

---

### Q11. What is DCGAN?
Deep Convolutional GAN, a GAN using CNNs for generator and discriminator.

---

### Q12. What are applications of DCGAN?
Image synthesis, super-resolution, and style transfer.

---

### Q13. What is Conditional GAN (cGAN)?
A GAN conditioned on labels or auxiliary information to control generation.

---

### Q14. Applications of cGAN?
Image-to-image translation, text-to-image generation, data augmentation.

---

### Q15. What is CycleGAN?
A GAN for unpaired image-to-image translation using cycle consistency loss.

---

### Q16. Applications of CycleGAN?
Style transfer, domain adaptation, medical imaging.

---

### Q17. What is Pix2Pix?
A paired image-to-image translation GAN using conditional adversarial loss.

---

### Q18. What is StyleGAN?
A GAN architecture by Nvidia for high-quality image generation with style control.

---

### Q19. Applications of StyleGAN?
Face generation, art synthesis, data augmentation.

---

### Q20. What is BigGAN?
A large-scale GAN architecture for class-conditional high-resolution image synthesis.

---

### Q21. What is Progressive Growing of GANs (ProGAN)?
GAN training technique where resolution is increased progressively.

---

### Q22. What is the main advantage of ProGAN?
It stabilizes training and generates high-resolution images.

---

### Q23. What is StarGAN?
A GAN for multi-domain image-to-image translation using a single model.

---

### Q24. Applications of StarGAN?
Face attribute manipulation, multi-domain transfer.

---

### Q25. What is ESRGAN?
Enhanced Super-Resolution GAN for high-quality image super-resolution.

---

### Q26. What is SRGAN?
Super-Resolution GAN for generating high-resolution images from low-resolution inputs.

---

### Q27. What is the difference between GAN and VAE?
- GAN: Adversarial training, sharper outputs.
- VAE: Probabilistic modeling, smoother outputs.

---

### Q28. What is latent space in GANs?
A random vector (usually Gaussian) input to the generator that is mapped to data space.

---

### Q29. What is latent space interpolation?
Interpolating between two latent vectors to generate smooth transformations.

---

### Q30. What is feature matching in GANs?
A training technique where the generator matches feature statistics of real data.

---

### Q31. What is minibatch discrimination?
A technique where the discriminator considers sample diversity to avoid mode collapse.

---

### Q32. What is spectral normalization?
A weight normalization technique ensuring Lipschitz continuity in the discriminator.

---

### Q33. What is self-attention GAN (SAGAN)?
A GAN using self-attention to capture long-range dependencies in images.

---

### Q34. What is attention GAN?
A GAN incorporating attention mechanisms for improved feature focus.

---

### Q35. What is InfoGAN?
A GAN variant maximizing mutual information between latent codes and generated outputs.

---

### Q36. Applications of InfoGAN?
Disentangled representation learning.

---

### Q37. What is energy-based GAN (EBGAN)?
A GAN using energy-based models where discriminator assigns energy to samples.

---

### Q38. What is Boundary Equilibrium GAN (BEGAN)?
A GAN variant balancing generator and discriminator losses dynamically.

---

### Q39. What is Least Squares GAN (LSGAN)?
A GAN variant using least-squares loss instead of cross-entropy for stability.

---

### Q40. What is Relativistic GAN (RGAN)?
A GAN where the discriminator predicts relative realism between real and fake samples.

---

### Q41. What is MMD-GAN?
A GAN minimizing Maximum Mean Discrepancy (MMD) between real and fake distributions.

---

### Q42. What is Laplacian Pyramid GAN (LAPGAN)?
A GAN generating high-resolution images progressively using Laplacian pyramids.

---

### Q43. What is Coupled GAN (CoGAN)?
GANs trained with shared weights to learn joint distributions without paired data.

---

### Q44. What is StackGAN?
A GAN generating high-resolution images from text in two stages.

---

### Q45. Applications of StackGAN?
Text-to-image synthesis for birds, flowers, and faces.

---

### Q46. What is Text-to-Image GAN?
GANs that generate images conditioned on text descriptions.

---

### Q47. What are applications of GANs in healthcare?
Medical image synthesis, augmentation, and anomaly detection.

---

### Q48. What are applications of GANs in finance?
Synthetic data generation, fraud detection, anomaly detection.

---

### Q49. What are applications of GANs in NLP?
Text generation, data augmentation, dialogue systems.

---

### Q50. What are applications of GANs in cybersecurity?
Malware detection, adversarial example generation, intrusion detection.

---

### Q51. What is adversarial training?
Training models to resist adversarial attacks by augmenting with adversarial samples.

---

### Q52. What is evaluation metric for GANs?
Common metrics: Inception Score (IS), FID (Fréchet Inception Distance), Precision-Recall.

---

### Q53. What is Inception Score?
A metric evaluating quality and diversity of generated images using Inception network.

---

### Q54. What is FID score?
Fréchet Inception Distance measures similarity between real and generated feature distributions.

---

### Q55. What is precision-recall in GAN evaluation?
Precision: sample quality.  
Recall: sample diversity.

---

### Q56. What is overfitting in GANs?
When generator memorizes training data, reducing diversity.

---

### Q57. How to prevent overfitting in GANs?
Data augmentation, dropout, spectral normalization, and regularization.

---

### Q58. What is GAN inversion?
Mapping real images back into latent space for editing.

---

### Q59. Applications of GAN inversion?
Face editing, image retrieval, style transfer.

---

### Q60. What is latent space arithmetic in GANs?
Adding/subtracting latent vectors to modify attributes (e.g., smiling + glasses).

---

### Q61. What is mode seeking GAN?
A GAN variant that encourages covering diverse modes.

---

### Q62. What is unrolled GAN?
A GAN that unrolls discriminator optimization to stabilize training.

---

### Q63. What is Bayesian GAN?
A GAN incorporating Bayesian inference for uncertainty estimation.

---

### Q64. What is federated GAN?
GANs trained in distributed settings preserving data privacy.

---

### Q65. What is privacy-preserving GAN?
GAN ensuring differential privacy during training.

---

### Q66. What is Medical GAN?
GANs specialized for healthcare applications like MRI reconstruction.

---

### Q67. What is Data Augmentation with GANs?
Using generated synthetic data to improve model generalization.

---

### Q68. What is adversarial example generation with GANs?
Using GANs to craft adversarial inputs for robustness testing.

---

### Q69. What is semi-supervised GAN (SGAN)?
GANs trained with limited labels, where discriminator also classifies data.

---

### Q70. What is feature pyramid GAN?
GANs with multi-scale feature learning for better image synthesis.

---

### Q71. What is 3D-GAN?
GAN generating 3D voxel data from latent space.

---

### Q72. What is video GAN?
GANs generating video sequences with temporal consistency.

---

### Q73. What is music GAN?
GANs generating music or audio sequences.

---

### Q74. What is speech GAN?
GANs generating realistic speech samples.

---

### Q75. What is reinforcement GAN?
GANs combined with reinforcement learning for sequence generation.

---

### Q76. What is policy gradient GAN?
GAN variant where generator is optimized with policy gradients.

---

### Q77. What is SeqGAN?
A GAN for text generation using reinforcement learning.

---

### Q78. What is MaliGAN?
A GAN variant for sequence generation addressing training instability.

---

### Q79. What is LeakGAN?
A GAN where discriminator leaks information to guide generator.

---

### Q80. What is DialogueGAN?
GANs for dialogue and conversational AI.

---

### Q81. What is GraphGAN?
GANs designed for graph representation learning.

---

### Q82. What is BioGAN?
GANs applied in bioinformatics and drug discovery.

---

### Q83. What is RelGAN?
GANs using relational modeling for text generation.

---

### Q84. What is StackGAN++?
An improved version of StackGAN with better stability and quality.

---

### Q85. What is Few-shot GAN?
GANs that can generate data with very few samples.

---

### Q86. What is One-shot GAN?
GAN trained from a single sample.

---

### Q87. What is Continual GAN?
GANs trained incrementally without forgetting.

---

### Q88. What is Memory GAN?
GANs with memory modules to improve long-term consistency.

---

### Q89. What is evaluation challenge in GANs?
Hard to quantify diversity and realism simultaneously.

---

### Q90. What are practical challenges in GAN training?
Instability, mode collapse, vanishing gradients, hyperparameter sensitivity.

---

### Q91. What is two-time scale update rule (TTUR)?
Updating discriminator and generator at different learning rates.

---

### Q92. What is consensus optimization?
A GAN optimization technique stabilizing training by modifying gradients.

---

### Q93. What is Nash equilibrium in GANs?
An optimal state where generator and discriminator cannot improve.

---

### Q94. Why are GANs hard to train?
Due to unstable adversarial dynamics and non-convex objectives.

---

### Q95. What is curriculum GAN?
Training GANs with progressively harder tasks.

---

### Q96. What is attention-guided GAN?
GAN with attention mechanisms focusing on critical regions.

---

### Q97. What is multimodal GAN?
GANs generating across multiple modalities (e.g., text + image).

---

### Q98. What is cross-modal GAN?
GANs translating between modalities (e.g., speech-to-image).

---

### Q99. What is hybrid GAN?
Combining GANs with other architectures like VAE-GAN.

---

### Q100. What is the future of GANs?
Applications in **AI-generated media, drug discovery, simulation, personalization, and adversarial robustness**, though newer models like **diffusion models** are increasingly popular.
