# Convolutional Neural Networks (CNN) – 100 Interview Questions & Answers

---

### Q1. What is a Convolutional Neural Network (CNN)?
A CNN is a deep learning architecture designed to process data with grid-like topology (e.g., images) using convolutional, pooling, and fully connected layers.

---

### Q2. What are the key components of a CNN?
* Convolutional layers  
* Activation functions  
* Pooling layers  
* Fully connected layers  
* Dropout/normalization layers

---

### Q3. What is convolution in CNN?
A mathematical operation applying a filter (kernel) to input data to extract local features.

---

### Q4. What is a kernel/filter in CNN?
A small matrix of weights used in convolution to detect features like edges, textures, or shapes.

---

### Q5. What is stride in CNN?
The number of steps the kernel moves during convolution. Larger stride reduces output size.

---

### Q6. What is padding in CNN?
Adding extra pixels around the input to preserve spatial dimensions after convolution.

---

### Q7. What is valid vs same padding?
* **Valid**: No padding, output smaller than input.  
* **Same**: Padding ensures output size equals input size.

---

### Q8. What is pooling in CNN?
Downsampling technique that reduces spatial dimensions, improves efficiency, and makes representations invariant.

---

### Q9. Types of pooling in CNN?
* Max pooling  
* Average pooling  
* Global pooling (global max/average)

---

### Q10. What is the purpose of pooling?
To reduce computation, prevent overfitting, and introduce spatial invariance.

---

### Q11. What are feature maps?
The outputs of convolution operations, representing detected features at different spatial locations.

---

### Q12. Why use multiple filters in CNN?
Each filter learns to detect a different pattern (e.g., edges, corners, textures).

---

### Q13. What is receptive field in CNN?
The region of the input image that influences a particular neuron’s activation.

---

### Q14. What is the role of activation functions in CNN?
Introduce non-linearity to allow CNNs to learn complex mappings beyond linear filters.

---

### Q15. Common activation functions in CNN?
ReLU, Leaky ReLU, ELU, Sigmoid, Tanh, Swish.

---

### Q16. Why is ReLU commonly used in CNNs?
It prevents vanishing gradients, is computationally efficient, and accelerates convergence.

---

### Q17. What is a fully connected layer in CNN?
A dense layer where each neuron connects to all inputs, typically used for classification after feature extraction.

---

### Q18. What is the typical CNN architecture flow?
Input → Convolution → Activation → Pooling → (repeat) → Fully connected → Output.

---

### Q19. What is the difference between CNN and MLP?
* **CNN**: Exploits spatial locality with shared weights and fewer parameters.  
* **MLP**: Fully connected with no spatial feature extraction.

---

### Q20. Why do CNNs have fewer parameters than MLPs?
Weight sharing in convolution layers drastically reduces parameters compared to fully connected layers.

---

### Q21. What is weight sharing in CNN?
Using the same filter across the input, reducing parameters and capturing translational invariance.

---

### Q22. What are channels in CNN?
Dimensions representing different features (e.g., RGB channels in images).

---

### Q23. What is 1x1 convolution?
A filter of size 1×1 that mixes information across channels without affecting spatial dimensions.

---

### Q24. Applications of 1x1 convolutions?
* Dimensionality reduction  
* Channel mixing  
* Increasing non-linearity

---

### Q25. What is stride > 1 used for?
To reduce spatial dimensions and control receptive field growth.

---

### Q26. What is dilated convolution?
Convolution with gaps between kernel weights, expanding receptive field without increasing parameters.

---

### Q27. Where is dilated convolution useful?
In segmentation tasks or where large receptive fields are needed (e.g., WaveNet).

---

### Q28. What is depthwise separable convolution?
Factorization of standard convolution into depthwise and pointwise (1x1) convolutions, reducing computation.

---

### Q29. Which architectures use depthwise separable convolutions?
MobileNet, Xception.

---

### Q30. What is a transposed convolution?
Also called deconvolution, it increases spatial dimensions, often used in generative models or segmentation.

---

### Q31. What are skip connections in CNN?
Links that bypass intermediate layers, used in ResNets/UNets to mitigate vanishing gradients.

---

### Q32. What is a residual block?
A block with skip connections allowing identity mapping plus transformation (used in ResNet).

---

### Q33. Why are residual connections important?
They enable training of very deep CNNs by solving vanishing gradient issues.

---

### Q34. What is a bottleneck block in CNN?
A block using 1x1 convolutions to reduce/increase dimensionality, commonly used in deep ResNets.

---

### Q35. What is batch normalization in CNN?
A layer that normalizes activations, stabilizes training, reduces internal covariate shift.

---

### Q36. Advantages of batch normalization?
* Faster training  
* Higher learning rates possible  
* Regularization effect

---

### Q37. What is layer normalization?
Normalization across features within a layer, often used in transformers.

---

### Q38. What is group normalization?
Dividing channels into groups and normalizing within each group, useful in small batch training.

---

### Q39. What is instance normalization?
Normalizing each sample independently, often used in style transfer.

---

### Q40. What is dropout in CNN?
A regularization technique where neurons are randomly disabled during training.

---

### Q41. Why use dropout in CNN?
To prevent overfitting and improve generalization.

---

### Q42. What is data augmentation in CNN?
Artificially expanding training data with transformations (e.g., flips, rotations, crops).

---

### Q43. Why is data augmentation important in CNN?
Improves generalization, prevents overfitting, and teaches invariance.

---

### Q44. What is transfer learning in CNN?
Using a pre-trained CNN (e.g., VGG, ResNet) on a new task with fine-tuning.

---

### Q45. Advantages of transfer learning?
* Saves training time  
* Requires less data  
* Benefits from large-scale pretraining

---

### Q46. What is fine-tuning in CNN?
Adjusting weights of a pre-trained CNN on new task data.

---

### Q47. What is feature extraction in CNN?
Using CNN layers to extract features and train a classifier on top.

---

### Q48. What are pretrained CNN models?
VGG, ResNet, Inception, MobileNet, DenseNet, EfficientNet.

---

### Q49. What is VGGNet?
A CNN architecture with 3x3 convolutions, deep layers, and simple design.

---

### Q50. What is ResNet?
A deep CNN with residual connections enabling very deep architectures.

---

### Q51. What is Inception network?
A CNN using multi-scale convolutions (1x1, 3x3, 5x5) in parallel.

---

### Q52. What is DenseNet?
A CNN where each layer connects to every other layer, promoting feature reuse.

---

### Q53. What is MobileNet?
A lightweight CNN using depthwise separable convolutions for mobile devices.

---

### Q54. What is EfficientNet?
A CNN that scales depth, width, and resolution systematically for optimal performance.

---

### Q55. What is UNet?
A CNN for image segmentation with encoder-decoder and skip connections.

---

### Q56. What is SegNet?
A CNN architecture for segmentation using encoder-decoder with pooling indices.

---

### Q57. What is AlexNet?
One of the earliest deep CNNs that won ImageNet 2012 with ReLU, dropout, and GPU training.

---

### Q58. Why was AlexNet significant?
It demonstrated the power of deep CNNs for large-scale image recognition.

---

### Q59. What is ZFNet?
An improvement over AlexNet with larger receptive fields and better visualization.

---

### Q60. What is GoogLeNet?
An Inception-based network with efficiency improvements and auxiliary classifiers.

---

### Q61. What is a global average pooling layer?
Reduces each feature map to a single value, replacing fully connected layers in classification.

---

### Q62. Why use global average pooling?
Reduces parameters, prevents overfitting, and improves interpretability.

---

### Q63. What are attention mechanisms in CNN?
Modules that allow CNNs to focus on important features (e.g., channel/spatial attention).

---

### Q64. What is Squeeze-and-Excitation (SE) block?
A channel attention mechanism that adaptively recalibrates channel weights.

---

### Q65. What is spatial attention?
Attention mechanism focusing on important spatial regions in feature maps.

---

### Q66. What is channel attention?
Mechanism that assigns different weights to feature channels.

---

### Q67. What is an R-CNN?
Region-based CNN for object detection using selective search + CNN features.

---

### Q68. What is Fast R-CNN?
Improved R-CNN with RoI pooling and single-stage training.

---

### Q69. What is Faster R-CNN?
Uses a Region Proposal Network (RPN) for faster object detection.

---

### Q70. What is Mask R-CNN?
Extends Faster R-CNN by adding a mask prediction branch for instance segmentation.

---

### Q71. What is YOLO?
“You Only Look Once” — a real-time object detection CNN that predicts bounding boxes and classes in one pass.

---

### Q72. What is SSD (Single Shot Detector)?
An object detection model predicting bounding boxes at multiple feature map scales.

---

### Q73. What is RetinaNet?
An object detection CNN using focal loss to handle class imbalance.

---

### Q74. What is focal loss?
A loss function that downweights easy examples and focuses on hard examples in detection.

---

### Q75. What is semantic segmentation in CNN?
Classifying each pixel into a category (e.g., road, car, building).

---

### Q76. What is instance segmentation in CNN?
Detecting and segmenting each object instance separately.

---

### Q77. What is panoptic segmentation?
Combines semantic and instance segmentation.

---

### Q78. What is object detection?
Locating and classifying objects within an image.

---

### Q79. What is image classification?
Assigning a single class label to an entire image.

---

### Q80. What is image localization?
Predicting bounding box coordinates for an object in an image.

---

### Q81. What is saliency map?
A visualization highlighting important pixels influencing a CNN prediction.

---

### Q82. What is Grad-CAM?
A visualization technique using gradients to highlight important image regions.

---

### Q83. What is occlusion sensitivity analysis?
Masking parts of input to see how CNN predictions change, revealing important regions.

---

### Q84. What is adversarial attack in CNN?
Adding small perturbations to inputs to fool CNN predictions.

---

### Q85. How to defend CNNs against adversarial attacks?
Adversarial training, input preprocessing, robust optimization.

---

### Q86. What is knowledge distillation in CNN?
Compressing a large CNN into a smaller one using soft labels from the teacher model.

---

### Q87. What is pruning in CNN?
Removing unimportant weights or filters to reduce model size.

---

### Q88. What is quantization in CNN?
Reducing weight precision (e.g., FP32 → INT8) to speed up inference.

---

### Q89. What is model compression in CNN?
Techniques like pruning, quantization, distillation to reduce model size.

---

### Q90. What are lightweight CNN models?
CNNs optimized for mobile/edge: MobileNet, ShuffleNet, SqueezeNet.

---

### Q91. What is ShuffleNet?
A lightweight CNN using group convolutions and channel shuffling.

---

### Q92. What is SqueezeNet?
A lightweight CNN with "fire modules" reducing parameters.

---

### Q93. What is NASNet?
A CNN discovered using Neural Architecture Search for optimal design.

---

### Q94. What is AutoML in CNNs?
Automated search for best CNN architectures/hyperparameters.

---

### Q95. What is multimodal CNN?
CNNs that process multiple modalities (e.g., image + text).

---

### Q96. What is 3D convolution?
Convolution applied in 3D space (e.g., video, volumetric data).

---

### Q97. Applications of 3D CNNs?
Video classification, medical imaging (MRI, CT scans).

---

### Q98. What is a capsule network in CNN context?
Architecture using capsules that encode spatial relationships, proposed to overcome CNN limitations.

---

### Q99. What are limitations of CNN?
Data hungry, computationally expensive, lack of interpretability, vulnerable to adversarial attacks.

---

### Q100. Future trends in CNN research?
* Efficient CNNs for edge devices  
* Integration with transformers  
* Better interpretability  
* Robustness to adversarial attacks  
* Self-supervised learning
