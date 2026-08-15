# Identity Recognition

A deep learning–based face verification system designed to determine whether two facial images belong to the same person.

The project was developed primarily for identity verification scenarios where a user's **selfie** needs to be compared with their **National ID image**. Beyond one-time verification, the generated facial embeddings can also be used for future identity comparisons, helping detect repeated registrations by the same individual.

The project went through two major development versions. The first version focused on researching, implementing, and comparing multiple face-verification architectures, while the second focused on improving data quality and diversity to achieve better generalization on unseen identities.

---

## Project Objective

The main objective is to build a robust computer vision pipeline capable of answering:

> **Do these two images belong to the same person?**

The system takes two facial images, extracts discriminative facial representations, calculates their similarity or distance, and then applies a calibrated decision threshold to determine whether they represent the same identity.

The development process covered the complete machine-learning lifecycle:

* Dataset acquisition and exploration
* Face detection and cropping
* Data cleaning
* Data augmentation
* Deep learning model development
* Transfer learning and fine-tuning
* Facial embedding extraction
* Model comparison
* Verification threshold optimization
* Evaluation on unseen identities

---

## Development Journey

Rather than starting directly with the final architecture, several approaches were implemented and evaluated to understand their behavior on the identity-verification problem.

### 1. VGG16 Siamese Network

The first implementation used a **Siamese Network with a VGG16 backbone**.

Two face images were passed through the same feature extractor to generate embeddings. The distance between those embeddings was then used to determine whether the images belonged to the same identity.

The model was trained using **Contrastive Loss**, which encourages:

* embeddings of the same identity to become closer;
* embeddings of different identities to become farther apart.

The first version achieved approximately **81.2% validation accuracy**.

---

### 2. Data Cleaning

Evaluation of the initial model revealed significant noise in the original dataset. Some images grouped under the same identity actually represented different people.

Instead of manually inspecting tens of thousands of images, facial embeddings were used to help identify inconsistent samples.

For every identity:

1. Embeddings were extracted from its images.
2. A representative mean embedding was calculated.
3. Images significantly different from that representation were treated as noisy samples.
4. A strict filtering threshold was used to remove potentially incorrect images.

This process reduced the original dataset from roughly **176K images to around 60K cleaner samples**.

---

### 3. Improved VGG Siamese Network

After cleaning the dataset, the Siamese model was retrained with an adjusted contrastive-loss margin.

The cleaner training data improved the validation accuracy to approximately **85%**.

This experiment highlighted how strongly face-recognition performance depends on dataset quality, not only model architecture.

---

### 4. IResNet Siamese Network

The next experiment replaced VGG16 with **IResNet50**, a backbone better suited for high-precision facial feature extraction.

The same Siamese learning approach was retained while using the stronger feature extractor.

The model reached approximately **88% validation accuracy**, improving both feature discrimination and computational efficiency compared with the VGG-based implementation.

---

### 5. IResNet50 + ArcFace

The final architectural direction replaced distance-based Siamese training with **ArcFace**.

ArcFace learns highly discriminative facial embeddings using **Additive Angular Margin Loss**. Instead of only optimizing Euclidean separation between pairs, the model learns identities in a normalized angular embedding space.

This encourages:

* low intra-class variation;
* high inter-class separation;
* more discriminative facial representations;
* better generalization to identities that were not present during training.

The model used an **IResNet50 backbone** for feature extraction and an ArcFace classification head during training.

The training process was performed in stages:

1. Train the ArcFace head while keeping the backbone frozen.
2. Unfreeze the backbone.
3. Fine-tune the complete network.

ArcFace significantly outperformed the previous Siamese approaches and became the selected architecture for the system.

---

# Version 2 — Improving Generalization

After Version 1, the main limitation was no longer the architecture. The strongest opportunity for improvement was the **amount and diversity of training data**.

Version 2 therefore focused primarily on building a larger and cleaner dataset before retraining the ArcFace model.

## Dataset Expansion

Multiple data sources were combined:

* Cleaned VGGFace training data
* Cleaned VGGFace validation data
* Filtered LFW data
* Additional manually collected identities

The objective was to expose the model to a wider range of:

* identities
* poses
* facial expressions
* lighting conditions
* backgrounds
* image resolutions
* image quality levels

---

## Identity Deduplication

Simply merging multiple facial datasets can introduce an important problem: the **same person may exist in more than one dataset under different class labels**.

To prevent these label conflicts, a pretrained ArcFace model was used to compare identities before accepting them into the combined dataset.

For each identity:

1. Several facial images were processed.
2. Face embeddings were extracted.
3. A representative identity embedding was generated.
4. New identities were compared against previously accepted identities using a strict similarity threshold.

This process detected and removed **11 duplicate identities**.

The resulting Version 2 dataset contained **786 unique identities**.

---

## Face Detection and Preprocessing

**RetinaFace** was used to detect and crop faces before model training.

Removing irrelevant background information helps the network focus on the facial characteristics that are useful for identity discrimination.

Training images were also augmented to better represent real-world conditions.

Augmentations included variations such as:

* color transformations
* brightness and lighting changes
* image noise
* blur
* rotation and geometric transformations
* image-quality degradation

These transformations are particularly useful for identity verification because selfies and ID-card photographs often differ substantially in lighting, sharpness, pose, compression, and overall image quality.

---

## Training

The Version 2 ArcFace model was retrained on the expanded dataset using **PyTorch** and **TorchVision**.

The model retained the **IResNet + ArcFace** architecture selected during Version 1.

Training again followed a two-stage transfer-learning strategy:

1. Freeze the IResNet backbone and train the ArcFace classification head.
2. Unfreeze the backbone and fine-tune the complete model.

The final training process ran for **29 effective epochs** before reaching the best validation performance.

---

## Verification and Threshold Selection

Face verification ultimately requires converting a continuous similarity score into a decision.

Several thresholds were therefore calculated and evaluated:

### EER Threshold

The threshold at which:

**False Acceptance Rate (FAR) ≈ False Rejection Rate (FRR)**

This provides a useful operating point for biometric verification systems.

### Accuracy Threshold

The threshold that produces the highest validation accuracy.

### F1 Threshold

The threshold that maximizes the F1 score and balances precision and recall.

The final Version 2 ArcFace thresholds were approximately:

| Threshold | Value |
| --------- | ----: |
| EER       | 0.189 |
| Accuracy  | 0.224 |
| F1        | 0.224 |

A similarity score above the selected threshold indicates that the two images are likely to belong to the same identity.

---

## Results on Unseen Data

The final model was evaluated using image pairs that were not involved in model training.

| Threshold | Correct Predictions | Total Pairs |   Accuracy |
| --------- | ------------------: | ----------: | ---------: |
| EER       |                  58 |          61 | **95.08%** |
| Accuracy  |                  59 |          61 | **96.72%** |
| F1        |                  59 |          61 | **96.72%** |

These results demonstrate that improving the diversity and cleanliness of the training data enhanced the model's ability to generalize to unseen identities.

---

## Model Evolution

| Model               | Main Improvement                  | Approx. Validation Performance |
| ------------------- | --------------------------------- | -----------------------------: |
| VGG16 Siamese V1    | Initial baseline                  |                          81.2% |
| VGG16 Siamese V2    | Cleaned dataset                   |                            85% |
| IResNet50 Siamese   | Stronger facial feature extractor |                            88% |
| IResNet50 + ArcFace | Angular-margin identity learning  |                           ~98% |

The experiments show a clear development path from conventional metric learning toward a more discriminative ArcFace-based embedding model.

---

## Technology Stack

### Deep Learning

* Python
* PyTorch
* TorchVision
* ArcFace
* IResNet50
* VGG16
* Siamese Networks
* Transfer Learning
* Fine-Tuning

### Computer Vision

* RetinaFace
* OpenCV
* Pillow

### Data Processing & Evaluation

* NumPy
* Pandas
* Scikit-learn

### Visualization

* Matplotlib
* Seaborn

---

## Key Concepts Implemented

This project covers several practical deep-learning and computer-vision concepts:

* Face Verification
* Facial Embeddings
* Metric Learning
* Contrastive Loss
* Additive Angular Margin Loss
* Siamese Networks
* ArcFace
* Transfer Learning
* Fine-Tuning
* Data Augmentation
* Dataset Cleaning
* Identity Deduplication
* Cosine Similarity
* Decision Threshold Optimization
* False Acceptance Rate (FAR)
* False Rejection Rate (FRR)
* Equal Error Rate (EER)
* Evaluation on Unseen Identities

---

## Final Outcome

The project evolved from an initial VGG16-based Siamese verification model into a considerably stronger **IResNet50 + ArcFace identity-verification system**.

The most important improvement was not simply switching architectures. The development process demonstrated the combined importance of:

* selecting an appropriate facial embedding architecture;
* identifying and removing noisy training samples;
* increasing identity diversity;
* preventing duplicate-label contamination;
* realistic data augmentation;
* and carefully calibrating verification thresholds.

The final Version 2 model achieved **96.72% accuracy on unseen verification pairs**, providing a strong foundation for real-world identity verification scenarios.
