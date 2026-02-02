# Brain MRI Classification (Healthy vs Glioma) – CNN & Transfer Learning

## Overview
This project tackles **binary classification of brain MRI images** into two classes:
- **Healthy**
- **Glioma**

The goal is to compare two approaches:
1. **Custom CNN** trained from scratch
2. **Transfer learning (EfficientNetB0)** using a **two-stage strategy**:
   - **Phase 1: Feature extraction** (backbone frozen)
   - **Phase 2: Fine-tuning** (partial unfreezing)

Beyond overall accuracy, the evaluation focuses on clinically relevant behavior using **confusion matrices**, **precision**, **recall**, **F1-score**, and **AUC**, supported by training curves.

> **Note:** The dataset is **not included** in this repository for privacy and size reasons.

---

## Repository Structure
BRAIN-CNN/
│
├── dataset/ # NOT included (ignored)
│ ├── healthy/
│ └── glioma/
│
├── experiments/ # Additional experiment scripts
│ ├── exp1Train.py
│ ├── exp2Train.py
│ └── exp3Train.py
│
├── src/ # Main pipeline scripts
│ ├── data_loader.py
│ ├── train.py # Custom CNN training
│ ├── train_pretrained_effnet.py # EfficientNetB0 transfer learning (2 phases)
│ ├── evaluate.py
│ └── predict.py
│
├── results/ # Outputs (plots + reports)
├── requirements.txt
└── README.md


---

## Methods

### 1) Custom CNN (Training from Scratch)
A CNN is trained end-to-end on the MRI dataset, learning features directly from the training data.

Typical components:
- Convolution + ReLU
- Pooling
- Fully connected classification head
- Regularization (e.g., dropout / L2) depending on experiment

---

### 2) Transfer Learning with EfficientNetB0 (Two-Stage Strategy)

#### Phase 1: Feature Extraction (Backbone Frozen)
The EfficientNetB0 backbone is frozen (`trainable=False`) and only the classification head is trained.
This allows adaptation to MRI images while preserving pretrained features.

#### Phase 2: Fine-tuning (Partial Unfreezing)
The backbone is partially unfrozen (usually the last layers) and trained with a smaller learning rate to better adapt to the target dataset while reducing the risk of overfitting.

---

## Evaluation
The evaluation includes:
- **Confusion Matrix** (healthy vs glioma)
- **Precision / Recall / F1-score**
- **AUC**
- Training curves (**loss** and **accuracy**)

This is important in medical AI because **how the model fails** (false positives vs false negatives) can matter as much as headline accuracy.

---

## How to Run the Project
```bash
# 1) Install dependencies
pip install -r requirements.txt

# 2) Train Custom CNN
python src/train.py

# 3) Train Transfer Learning (EfficientNetB0: phase 1 then phase 2)
python src/train_pretrained_effnet.py

# 4) Evaluate the model
python src/evaluate.py

# 5) Run prediction on a single image (optional)
python src/predict.py
