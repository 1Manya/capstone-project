# Rolling-Shutter Laser Attack Detection Using U-Net, SSD and Feature Fusion

## Overview

This project focuses on detecting laser-induced rolling-shutter interference in camera images. Such attacks can introduce visual distortions and affect the reliability of camera-based perception systems.

The updated architecture uses a **U-Net segmentation model, a pretrained SSD object detector, and a feature-fusion neural network** to classify an input image as either:

- **Clean** – No laser attack detected.
- **Corrupt** – Laser-induced interference detected.

Unlike the earlier multi-class classification approach, the updated system performs binary clean/corrupt classification while combining information from both image segmentation and object detection.

The system is also evaluated on an independent nuScenes-based dataset generated using a separate Köhler rolling-shutter attack engine, without retraining or fine-tuning the trained fusion model.

---

## 1. Updated System Architecture

The proposed architecture follows a parallel feature-extraction and fusion approach.

```text
                  Input Camera Image
                          |
              -------------------------
              |                       |
       U-Net Segmentation       SSD300 Object Detection
              |                       |
       Segmentation Mask        Detection Information
              |                       |
       U-Net Features             SSD Features
              |                       |
              -----------+------------
                         |
                  Feature Fusion
                         |
                    Fusion MLP
                         |
                Binary Classification
                         |
                 ----------------
                 |              |
               Clean          Corrupt
```

### Component Description

**1. Mask Generation**

- Generates masks representing laser-affected regions.
- Uses simulated attack information to create segmentation targets.
- These masks are used to train the U-Net segmentation model.

**2. U-Net Segmentation Model**

- Identifies potentially affected regions in an input image.
- Produces a segmentation probability map.
- Extracts five numerical features:
  - Mask coverage
  - Mean segmentation probability
  - Maximum segmentation probability
  - Connected component count
  - Largest connected component

**3. SSD Object Detection**

- Uses pretrained SSD300 with a VGG16 backbone and COCO weights.
- Extracts object detection confidence information.
- Uses a confidence floor of 0.15.
- Extracts five numerical features:
  - Number of detections
  - Mean confidence
  - Maximum confidence
  - Standard deviation of confidence
  - Number of low-confidence detections

**4. Feature Fusion and Classification**

The five U-Net features and five SSD features are concatenated into a single 10-dimensional feature vector.

A Fusion MLP processes the combined representation and predicts the final binary class.

The model outputs:
- Clean probability
- Corrupt probability
- Predicted class

### Fusion MLP Architecture

```text
Input: 10 Features
       |
Linear (10 → 128)
       |
Batch Normalization
       |
ReLU + Dropout (0.30)
       |
Linear (128 → 64)
       |
Batch Normalization
       |
ReLU + Dropout (0.20)
       |
Linear (64 → 2)
       |
Clean / Corrupt
```

---

## 2. Dataset

### Training and Internal Evaluation Dataset

The simulated dataset contains **97,500 images** across six original categories.

| Original Category | Number of Images |
|---|---:|
| Clean | 12,500 |
| Attack Low | 12,500 |
| Attack Mid | 12,500 |
| Attack High | 12,500 |
| Attack Ultra | 12,500 |
| Attack Full Range | 35,000 |
| **Total** | **97,500** |

For the updated binary classification task, the labels are mapped as follows:

| Original Label | Binary Label |
|---|---|
| Clean | 0 – Clean |
| All Attack Categories | 1 – Corrupt |

The dataset is imbalanced, with 12,500 clean images and 85,000 attacked images.

To address this imbalance, class-weighted Cross Entropy Loss is used during fusion model training. The class weights are calculated using the training split.

### Independent External Evaluation Dataset

An additional dataset containing **2,400 images** is used to evaluate generalization.

Dataset characteristics:

- 1,200 clean images
- 1,200 attacked images
- Six nuScenes camera views
- Attack patterns generated using a separate Köhler rolling-shutter attack engine
- Attack frequencies: 25 Hz, 259 Hz, 511 Hz and 750 Hz

The external dataset is kept separate from model training.

---

## 3. Training Methodology

### U-Net Training

- Trained using generated laser-affected region masks.
- Training completed for 30 epochs.
- Best validation loss: 0.5766
- Final training loss: 0.4263

### SSD Object Detection

- Uses pretrained SSD300-VGG16 COCO weights.
- No SSD retraining is performed.
- Detection features are extracted using the same preprocessing and confidence thresholds used during fusion feature generation.

### Fusion Model Training

- Input features: 10
- Output classes: 2
- Loss function: Weighted Cross Entropy Loss
- Optimizer: As configured in the training script.
- Best checkpoint selected using validation F1-score.
- Feature scaling is fitted only on the training split and reused during evaluation.

**Best validation results:**

- Best epoch: 7
- Validation F1-score: 99.75%

---

## 4. Internal Test Results

The trained binary fusion model achieved the following results on the internal test dataset.

| Metric | Result |
|---|---:|
| Accuracy | 99.49% |
| Clean Precision | 97% |
| Clean Recall | 99% |
| Clean F1-score | 98% |
| Corrupt Precision | 100% |
| Corrupt Recall | 100% |
| Corrupt F1-score | 100% |

### Internal Test Confusion Matrix

```text
                 Predicted
               Clean  Corrupt

Actual Clean    1897     22
Actual Corrupt    53   12725
```

Total test samples: 14,697.

---

## 5. Independent External Evaluation

To evaluate generalization beyond the simulated training dataset, the trained fusion model was tested on the independent nuScenes-based attack dataset.

**Important:** The external evaluation was performed without retraining, fine-tuning, or refitting the feature scaler.

### External Evaluation Results

| Metric | Result |
|---|---:|
| Accuracy | 82.50% |
| Precision | 79.28% |
| Recall | 88.00% |
| F1-score | 83.41% |
| Specificity | 77.00% |
| ROC-AUC | 90.66% |

### External Test Confusion Matrix

```text
                 Predicted
               Clean  Corrupt

Actual Clean     924    276
Actual Corrupt   144   1056
```

Total external test samples: 2,400.

### Results Interpretation

- The model correctly identified 88% of attacked images.
- It correctly classified 77% of clean images.
- The ROC-AUC of 90.66% indicates good separation between clean and attacked samples.
- The performance difference between internal and external evaluation indicates a domain gap between the simulated training data and the independently generated external data.

These results demonstrate promising cross-dataset generalization, although further testing across different attack conditions and real camera environments is necessary.

---

## 6. Model Outputs

For each input image, the system produces:

```text
Predicted Class: Clean / Corrupt

Clean Probability: 0.XX
Corrupt Probability: 0.XX
```

The predicted class is selected using the model's output probabilities.

---

## 7. Project Structure

```text
capstone-project/
│
├── Unet_ssd_fusion/
│   ├── train_fusion.py
│   ├── fusion_model.py
│   ├── extract_detection_features.py
│   └── ...
│
├── outputs/
│   ├── unet/
│   │   └── unet_features.csv
│   │
│   ├── ssd/
│   │   └── ssd_features.csv
│   │
│   ├── fusion/
│   │   ├── fusion_binary_final.pth
│   │   ├── fusion_binary_checkpoint.pth
│   │   ├── fusion_binary_metadata.json
│   │   └── binary_test_predictions.csv
│   │
│   └── external_test/
│       ├── external_unet_features.csv
│       ├── external_ssd_features.csv
│       └── external_fusion_predictions.csv
│
└── data/
    └── final_dataset/
        └── labels.csv
```

*Note: The structure above highlights the main files and may not include every project file. Large datasets and model artifacts may be maintained separately from the GitHub repository.*

---

## 8. Technologies Used

- Python
- PyTorch
- Torchvision
- U-Net
- SSD300 with VGG16
- NumPy
- Pandas
- Scikit-learn
- OpenCV
- Matplotlib
- Kaggle GPU Environment

---

## 9. Key Contributions

- Developed a segmentation-based approach to identify laser-affected image regions.
- Integrated U-Net segmentation features with SSD object detection features.
- Designed a feature-fusion MLP for binary clean/corrupt classification.
- Addressed class imbalance using weighted Cross Entropy Loss.
- Maintained consistent feature preprocessing and scaling during evaluation.
- Evaluated the trained model on an independent dataset generated using a separate rolling-shutter attack engine.
- Demonstrated external evaluation performance of 82.50% accuracy and 90.66% ROC-AUC without additional training.

---

## 10. Future Improvements

- Improve robustness against different laser frequencies, intensities and attack patterns.
- Investigate the domain gap between simulated and independently generated attack datasets.
- Improve clean-image specificity while maintaining high attack detection recall.
- Evaluate performance under different camera exposure settings and real-world conditions.
- Extend testing to additional camera datasets and unseen attack configurations.

---

## Conclusion

The updated system combines U-Net segmentation and SSD object detection through a feature-fusion network to detect laser-induced rolling-shutter interference.

The model achieves **99.49% internal test accuracy** and **82.50% accuracy on an independent external dataset**, with an external ROC-AUC of **90.66%**.

The external evaluation provides an important assessment of the model's ability to generalize beyond its original simulated training data.
