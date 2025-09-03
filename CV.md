# Computer Vision (CV) - 100 Interview Questions & Answers

---

### Q1. What is Computer Vision?
A field of AI that enables machines to interpret and understand visual data such as images and videos.

---

### Q2. What are common applications of Computer Vision?
Object detection, face recognition, self-driving cars, medical imaging, OCR, video surveillance, and augmented reality.

---

### Q3. What are the key tasks in Computer Vision?
Image classification, object detection, image segmentation, pose estimation, tracking, and image generation.

---

### Q4. What is image classification?
Assigning a label to an entire image (e.g., cat vs. dog).

---

### Q5. What is object detection?
Identifying and localizing objects within an image using bounding boxes.

---

### Q6. What is image segmentation?
Partitioning an image into regions (semantic segmentation) or distinguishing object instances (instance segmentation).

---

### Q7. What is pose estimation?
Detecting the orientation and position of objects or humans (e.g., skeleton keypoints).

---

### Q8. What is OCR in CV?
Optical Character Recognition — extracting text from images or scanned documents.

---

### Q9. What are convolutional neural networks (CNNs)?
Deep learning models designed for processing image data by learning spatial hierarchies of features.

---

### Q10. What is the role of convolution in CNNs?
Extracts features by applying filters over local regions of the image.

---

### Q11. What is pooling in CNNs?
Downsampling operation (max/average pooling) to reduce dimensionality and control overfitting.

---

### Q12. What is stride in convolution?
The step size with which the filter moves across the image.

---

### Q13. What is padding in CNNs?
Adding extra pixels around image borders to control output size and preserve edge information.

---

### Q14. What is a receptive field?
The region of the input image that influences a neuron’s activation.

---

### Q15. What is data augmentation in CV?
Techniques like flipping, rotation, scaling, and cropping to increase dataset diversity.

---

### Q16. What is transfer learning in CV?
Using pre-trained models (e.g., ResNet, VGG) on large datasets (ImageNet) for new tasks.

---

### Q17. What is fine-tuning?
Adapting a pre-trained model by updating some or all layers for a specific dataset.

---

### Q18. What is feature extraction with CNNs?
Using pre-trained layers to extract image features and training a classifier on top.

---

### Q19. What is ResNet?
A CNN architecture using residual connections to train very deep networks.

---

### Q20. What is VGGNet?
A deep CNN with 16–19 layers of small (3x3) convolutions.

---

### Q21. What is InceptionNet?
A CNN architecture using inception modules with multi-scale filters.

---

### Q22. What is EfficientNet?
A CNN family that scales depth, width, and resolution efficiently.

---

### Q23. What is MobileNet?
A lightweight CNN architecture optimized for mobile and embedded devices.

---

### Q24. What are vision transformers (ViTs)?
Transformer-based architectures adapted to process image patches instead of sequences.

---

### Q25. What is object detection vs classification?
- Classification: label for entire image.  
- Detection: identify objects with bounding boxes.  

---

### Q26. What are R-CNNs?
Region-based CNNs for object detection by generating region proposals.

---

### Q27. What is Fast R-CNN?
Improved R-CNN that shares convolutional computations and uses ROI pooling.

---

### Q28. What is Faster R-CNN?
Object detection model introducing Region Proposal Networks (RPN).

---

### Q29. What is YOLO?
“You Only Look Once” — a real-time object detection model.

---

### Q30. What is SSD?
Single Shot MultiBox Detector — another real-time object detection model.

---

### Q31. What is Mask R-CNN?
Extension of Faster R-CNN adding instance segmentation masks.

---

### Q32. What is semantic segmentation?
Classifying each pixel into a category (e.g., road, sky, building).

---

### Q33. What is instance segmentation?
Differentiating between separate instances of the same class.

---

### Q34. What is panoptic segmentation?
Combines semantic and instance segmentation into a unified task.

---

### Q35. What is a bounding box?
A rectangular region enclosing an object for detection tasks.

---

### Q36. What is Intersection over Union (IoU)?
Metric measuring overlap between predicted and ground truth bounding boxes.

---

### Q37. What is mean Average Precision (mAP)?
Evaluation metric for object detection averaging precision across classes and IoU thresholds.

---

### Q38. What is non-maximum suppression (NMS)?
Algorithm for removing redundant overlapping bounding boxes.

---

### Q39. What is keypoint detection?
Locating specific points of interest (e.g., facial landmarks, human joints).

---

### Q40. What is facial recognition?
Identifying or verifying individuals based on facial features.

---

### Q41. What is feature pyramid network (FPN)?
Architecture for multi-scale feature representation in object detection.

---

### Q42. What is anchor box in detection?
Predefined box shapes used to detect objects at different scales/aspects.

---

### Q43. What is RetinaNet?
One-stage object detector using focal loss to handle class imbalance.

---

### Q44. What is focal loss?
Loss function focusing on hard-to-classify examples in object detection.

---

### Q45. What is data annotation in CV?
Labeling datasets with bounding boxes, masks, or keypoints.

---

### Q46. What is ImageNet?
A large-scale dataset with millions of labeled images across 1,000 classes.

---

### Q47. What is COCO dataset?
Common Objects in Context — dataset for detection, segmentation, and captioning.

---

### Q48. What is Pascal VOC?
Dataset for object detection and segmentation tasks.

---

### Q49. What is medical imaging in CV?
Applying CV techniques to X-rays, MRIs, CT scans for diagnosis.

---

### Q50. What is anomaly detection in CV?
Identifying unusual patterns (e.g., defects, fraud, outliers).

---

### Q51. What is GAN in CV?
Generative Adversarial Network — used for generating synthetic images.

---

### Q52. What is CycleGAN?
GAN for unpaired image-to-image translation.

---

### Q53. What is StyleGAN?
GAN architecture generating photorealistic human faces.

---

### Q54. What is neural style transfer?
Applying the artistic style of one image to another.

---

### Q55. What is image super-resolution?
Enhancing image resolution using DL models like SRCNN, ESRGAN.

---

### Q56. What is denoising autoencoder?
Autoencoder variant used to remove noise from images.

---

### Q57. What is image inpainting?
Filling missing or corrupted parts of images using DL.

---

### Q58. What is video object tracking?
Tracking object locations across video frames.

---

### Q59. What is optical flow?
Estimating motion of pixels between video frames.

---

### Q60. What is 3D computer vision?
Reconstructing 3D information from 2D images (stereo vision, depth maps).

---

### Q61. What is SLAM?
Simultaneous Localization and Mapping — used in robotics and AR.

---

### Q62. What is LiDAR in CV?
Light Detection and Ranging sensors used for 3D perception in autonomous driving.

---

### Q63. What is point cloud processing?
Analyzing 3D point data for perception tasks.

---

### Q64. What is depth estimation?
Predicting depth information from 2D images.

---

### Q65. What is monocular depth estimation?
Inferring depth from a single image.

---

### Q66. What is stereoscopic vision?
Using two cameras for depth perception.

---

### Q67. What is structure from motion (SfM)?
Reconstructing 3D structure from moving camera images.

---

### Q68. What is AR in CV?
Augmented Reality — overlaying digital objects on real-world scenes.

---

### Q69. What is VR in CV?
Virtual Reality — fully immersive digital environments.

---

### Q70. What is mixed reality (MR)?
Blending real and digital environments interactively.

---

### Q71. What is OCR pipeline in CV?
Steps: preprocessing → text detection → text recognition → postprocessing.

---

### Q72. What is EAST text detector?
Efficient and Accurate Scene Text detector for OCR.

---

### Q73. What is CRNN?
Convolutional Recurrent Neural Network used for text recognition.

---

### Q74. What is scene understanding in CV?
Interpreting overall scene context (objects, layout, relationships).

---

### Q75. What is graph-based CV?
Using Graph Neural Networks (GNNs) for scene graphs and relationships.

---

### Q76. What are explainable CV models?
Models with interpretable outputs using saliency maps, Grad-CAM, etc.

---

### Q77. What is Grad-CAM?
Gradient-weighted Class Activation Mapping for visual explanations.

---

### Q78. What are saliency maps?
Visualizations highlighting important input regions for predictions.

---

### Q79. What is adversarial attack in CV?
Perturbing images slightly to fool CV models.

---

### Q80. What is adversarial defense in CV?
Training methods to make models robust to adversarial examples.

---

### Q81. What is federated learning in CV?
Training CV models across distributed devices without sharing raw data.

---

### Q82. What is differential privacy in CV?
Adding noise to protect sensitive information during model training.

---

### Q83. What is transfer vs domain adaptation in CV?
- Transfer: apply pre-trained model to new but related task.  
- Domain adaptation: adapt model to new but similar data distributions.  

---

### Q84. What is zero-shot CV?
Generalizing to new categories without labeled training examples.

---

### Q85. What is few-shot CV?
Training with very few labeled examples.

---

### Q86. What is one-shot CV?
Learning from only one example per class.

---

### Q87. What is CLIP model?
Contrastive Language-Image Pretraining — aligns text and images.

---

### Q88. What is multimodal CV?
Combining vision with other modalities (text, audio).

---

### Q89. What is visual question answering (VQA)?
Answering text questions about images.

---

### Q90. What is image captioning?
Generating natural language descriptions of images.

---

### Q91. What is self-supervised learning in CV?
Learning representations from unlabeled data using pretext tasks.

---

### Q92. What is contrastive learning in CV?
Learning by contrasting positive and negative image pairs.

---

### Q93. What is SimCLR?
A self-supervised contrastive learning framework for CV.

---

### Q94. What is BYOL?
Bootstrap Your Own Latent — a self-supervised CV model without negatives.

---

### Q95. What is SwAV?
Self-supervised learning approach using clustering.

---

### Q96. What is MAE in CV?
Masked Autoencoders for self-supervised learning on images.

---

### Q97. What is DINO in CV?
Self-distillation with no labels for vision transformers.

---

### Q98. What are challenges in CV?
Occlusion, viewpoint changes, lighting variations, domain shifts.

---

### Q99. What is real-time CV?
Deploying efficient models for live inference (e.g., YOLO, MobileNet).

---

### Q100. What is the future of CV?
Trends: foundation models (CLIP, SAM), multimodal vision-language models, real-time edge CV, and explainable, ethical AI.

---
