# VisionGuard — Manufacturing Defect Detection System

[![Python](https://img.shields.io/badge/Python-3.10%20%7C%203.11-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-EE4C2C.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Status](https://img.shields.io/badge/Quality%20Rubric-Passed%20(>90%25%20Acc)-brightgreen.svg)]()

> **Code A Nova • Internship Program 2026 • Machine Learning**  
> Train a deep learning CNN to automatically classify product images from a manufacturing line as **defective** or **normal** — demonstrating applied computer vision for industrial quality control automation.

---

## 📌 Table of Contents
1. [Project Overview](#-project-overview)
2. [Key Results & Success Indicators](#-key-results--success-indicators)
3. [Architecture & Engineering](#-architecture--engineering)
4. [Folder Structure](#-folder-structure)
5. [Installation & Setup](#-installation--setup)
6. [Dataset Preparation & Class Audit](#-dataset-preparation--class-audit)
7. [Model Training & Checkpointing](#-model-training--checkpointing)
8. [Evaluation & Analysis](#-evaluation--analysis)
9. [Single-Image & Batch Inference (CLI)](#-single-image--batch-inference-cli)
10. [Interactive Demo Application (Streamlit)](#-interactive-demo-application-streamlit)
11. [7-Day Milestone Compliance](#-7-day-milestone-compliance)

---

## 🔍 Project Overview

VisionGuard is an end-to-end industrial computer vision defect detection solution designed to eliminate manual visual inspection bottlenecks on high-speed manufacturing lines. By processing surface imagery through fine-tuned deep convolutional networks, VisionGuard identifies anomalies such as micro-scratches, stress cracks, porosity voids, slag inclusions, thermal oxidation, and surface contaminants with high confidence.

---

## 🏆 Key Results & Success Indicators

VisionGuard was evaluated strictly against the Code A Nova performance rubric. These numbers come from an actual `train.py` + `evaluate.py` run on the bundled dataset (Custom CNN, 12 epochs, 128px, CPU) — see `reports/performance_report.md` for the live-generated report and `reports/training_curves.png` for the real (non-flat) learning curve:

| Metric | Target Rubric | Achieved Result | Status |
| :--- | :--- | :--- | :--- |
| **Overall Accuracy** | **> 90.0%** | **97.78%** | ✅ **PASSED** |
| **Defect Class Recall** | **> 85.0%** | **95.56%** | ✅ **PASSED** |
| **ROC-AUC Score** | **> 0.920** | **1.0000** | ✅ **PASSED** |
| **Defect F1-Score** | High Precision/Recall | **97.73%** | ✅ **PASSED** |
| **Inference Latency** | Real-time industrial line | **~16-18 ms / image (CPU)** | ✅ **PASSED** |

> **Note on the dataset:** `prepare_data.py` intentionally makes the synthetic task non-trivial — defects are rendered at randomized severity (some subtle, some obvious) and ~35% of *normal* parts carry faint, within-tolerance blemishes (dust, light handling marks) that resemble low-severity defects. This is why the model above tops out around 97-98% instead of a suspicious 100% — the confusion matrix shows 2 genuine misses out of 90 test images, not a perfectly separable toy problem. Swap in real MVTec AD data with `--mvtec_dir` for a production-grade benchmark.

---

## ⚙️ Architecture & Engineering

### 1. Data Pipeline
- **Dataset Splitting**: Strict 70% Train, 15% Validation, 15% Test split with verified class balance.
- **Realistic Synthetic Generation**: Defects are rendered at randomized severity (subtle → obvious) via `prepare_data.py`, and ~35% of normal surfaces carry faint benign blemishes (dust, light marks) that resemble low-severity defects — this keeps the task genuinely learnable rather than trivially separable, so a model can't just memorize "any dark mark = defective."
- **Augmentation Pipeline**: `RandomCrop`, `RandomHorizontalFlip`, `RandomVerticalFlip`, `RandomRotation(±15°)`, `ColorJitter` (brightness, contrast, saturation), and ImageNet channel standardization (`IMAGENET_MEAN`, `IMAGENET_STD`).
- **Configurable Resolution**: `--image_size` (default `224`, lower e.g. `128` for fast CPU-only training/demo runs) is threaded through `train.py` → checkpoint → `evaluate.py`/`predict.py`/`app.py`, so inference always matches the resolution a given checkpoint was trained at.

### 2. Model Architectures (`model.py`)
- **ResNet-18 Transfer Learning (Default, `--architecture resnet18`)**: Pretrained on ImageNet. Base feature layers (`conv1`, `bn1`, `layer1`, `layer2`) frozen; high-level semantic blocks (`layer3`, `layer4`) fine-tuned. Customized dropout-regularized classification head (`Linear(512 -> 128) -> ReLU -> Dropout(0.3) -> Linear(128 -> 2)`). **Requires internet access on first run** to download ImageNet weights (cached afterwards by `torch.hub`); if you're training fully offline, use `custom_cnn` instead or pass `pretrained=False`.
- **Custom CNN (`--architecture custom_cnn`)**: 4-stage convolutional backbone (`Conv2D > BatchNorm > ReLU > MaxPool2D`) with global average pooling and dense classification head. Trains from scratch, no internet required — this is the architecture used to produce the results in the table above.

### 3. Training Engineering (`train.py`)
- **Loss Function**: `CrossEntropyLoss` with automated inverse-frequency class weights to combat potential dataset imbalance.
- **Optimization**: `Adam` optimizer paired with `ReduceLROnPlateau` learning rate scheduling.
- **Regularization**: Early stopping on validation loss with configurable patience.
- **Persistence**: Atomic checkpoint saving (`best_model.pt` and `last_checkpoint.pt`) with full state restoration support (`--resume`).
- **Telemetry**: Real-time TensorBoard scalar logging (`Loss/Train`, `Loss/Val`, `Accuracy/Train`, `Accuracy/Val`, `LearningRate`).

---

## 📂 Folder Structure

```
VisionGuard/
├── data/
│   ├── train/ (70%)
│   │   ├── normal/
│   │   └── defective/
│   ├── val/ (15%)
│   │   ├── normal/
│   │   └── defective/
│   └── test/ (15%)
│       ├── normal/
│       └── defective/
├── checkpoints/
│   ├── best_model.pt          # Best model weights & metadata
│   └── last_checkpoint.pt     # Checkpoint for training resumption
├── logs/
│   └── tensorboard/           # TensorBoard event runs
├── reports/
│   ├── training_curves.png    # Loss & Accuracy trajectories
│   ├── confusion_matrix.png   # Test set confusion matrix
│   ├── roc_curve.png          # Receiver Operating Characteristic curve
│   └── performance_report.md  # Official Markdown evaluation report
├── sample_images/
│   ├── test_sample_normal.jpg
│   └── test_sample_defective.jpg
├── prepare_data.py            # Dataset generation, MVTec AD ingestion & auditing
├── dataset.py                 # ImageFolder transforms, augmentations & DataLoaders
├── model.py                   # Custom CNN & ResNet-18 architectures
├── train.py                   # Training loop, scheduler, early stopping, TensorBoard
├── evaluate.py                # Full test evaluation suite & rubric validation
├── predict.py                 # Single-image & batch CLI inference script
├── app.py                     # Streamlit interactive visual demo dashboard
├── requirements.txt           # Python dependencies
└── README.md                  # System documentation
```

---

## 🚀 Installation & Setup

### 1. Clone or Open Workspace
```bash
cd VisionGuard
```

### 2. Set Up Python Virtual Environment
```bash
python -m venv venv
# Windows:
venv\Scripts\activate
# Linux/macOS:
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

---

## 📊 Dataset Preparation & Class Audit

### Option A: Generate Industrial Synthetic Defect Dataset (Zero Setup)
Generates high-fidelity textured industrial surfaces (normal) and defects (cracks, scratches, pits, corrosion, contaminants) at randomized severity, split 70/15/15. A portion of normal surfaces also get faint benign blemishes so the task isn't trivially separable:
```bash
python prepare_data.py --generate --samples_per_class 500
```

### Option B: Ingest Open-Source MVTec AD Dataset
To use an MVTec AD category (e.g. `metal_nut`, `bottle`, `hazelnut`):
```bash
python prepare_data.py --mvtec_dir path/to/mvtec/metal_nut --output_dir data
```

### Option C: Audit Existing Dataset
```bash
python prepare_data.py --audit --output_dir data
```
**Audit Output Table** (example at `--samples_per_class 500`; the bundled dataset in this zip uses `300` per class for a faster CPU demo — `210/45/45` per split):
```
==============================================================
         VISIONGUARD DATASET AUDIT (70/15/15 Target)          
==============================================================
Split       Normal      Defective     Total       Ratio (%)   
--------------------------------------------------------------
train       350         350           700             70.0%
val         75          75            150             15.0%
test        75          75            150             15.0%
--------------------------------------------------------------
TOTAL       500         500           1000           100.0%
==============================================================
```

---

## 🏋️ Model Training & Checkpointing

### Train with Pretrained ResNet-18 (Recommended, needs internet for weights)
```bash
python train.py --architecture resnet18 --epochs 10 --batch_size 32 --lr 0.0001
```

### Train with Custom CNN Architecture (fully offline-capable)
```bash
python train.py --architecture custom_cnn --epochs 15 --batch_size 32 --lr 0.0003
```

### Fast CPU-only run (smaller input resolution)
```bash
python train.py --architecture custom_cnn --epochs 12 --batch_size 32 --lr 0.0005 --image_size 128
```

### Resuming Training from Checkpoint
If training is interrupted or further fine-tuning is desired:
```bash
python train.py --resume checkpoints/last_checkpoint.pt --epochs 5
```

### Launching TensorBoard
To view real-time training curves in your browser:
```bash
tensorboard --logdir logs/tensorboard
```
Navigate to `http://localhost:6006` to inspect loss, accuracy, and learning rate curves.

---

## 📈 Evaluation & Analysis

Run the evaluation suite on the unseen test split:
```bash
python evaluate.py --model checkpoints/best_model.pt --data_dir data
```

### Generated Artifacts in `reports/`:
1. **`reports/confusion_matrix.png`**: High-resolution heatmap of true vs predicted labels.
2. **`reports/roc_curve.png`**: ROC curve plotting True Positive Rate vs False Positive Rate.
3. **`reports/training_curves.png`**: Epoch-by-epoch loss convergence and accuracy progression.
4. **`reports/performance_report.md`**: Formal audit report detailing exact precision, recall, and F1 per class.

---

## 🔬 Single-Image & Batch Inference (CLI)

VisionGuard provides a lightweight, standalone CLI inference tool `predict.py` that operates on arbitrary images without requiring the dataset to be present:

### 1. Single Image Prediction
```bash
python predict.py --image sample_images/test_sample_defective.jpg --model checkpoints/best_model.pt
```
**Terminal Output:**
```
=======================================================
             VISIONGUARD INSPECTION REPORT             
=======================================================
Target Image : sample_images/test_sample_defective.jpg
Decision     : ❌ DEFECTIVE [REJECT]
Confidence   : 99.82%
Latency      : 18.24 ms
-------------------------------------------------------
Class Probabilities:
  defective   : 0.9982  [████████████████████████░]
  normal      : 0.0018  [░░░░░░░░░░░░░░░░░░░░░░░░░]
=======================================================
```

### 2. Single Image with Visual Overlay Banner
```bash
python predict.py --image sample_images/test_sample_defective.jpg --model checkpoints/best_model.pt --save_vis reports/inspection_overlay.jpg
```

### 3. Batch Directory Inference
```bash
python predict.py --batch_dir sample_images/ --model checkpoints/best_model.pt
```

---

## 💻 Interactive Demo Application (Streamlit)

Launch the interactive quality assurance dashboard:
```bash
streamlit run app.py
```
**Dashboard Features:**
- **Live Inspection**: Test provided samples or upload custom product images.
- **Pass/Fail Verdicts**: Instant visual indication (Green for OK, Red for Defective) with confidence meters.
- **Analytics View**: Inspect the confusion matrix, ROC-AUC curve, and training curves directly in the UI.

---

## 🗓️ 7-Day Milestone Compliance

| Day | Goal & Tasks | Completed Implementation |
| :--- | :--- | :--- |
| **Day 1** | Dataset preparation, 70/15/15 splits, class audit | `prepare_data.py` (synthetic generator & MVTec AD importer) |
| **Day 2** | DataLoader pipeline, resizing, normalization, augmentations | `dataset.py` (`ImageFolder`, `transforms`, `DataLoaders`) |
| **Day 3** | CNN architecture design, ResNet-18 transfer learning | `model.py` (`CustomCNN`, `VisionGuardResNet18`) |
| **Day 4** | Training loop, loss computation, optimizer step, validation | `train.py` (`CrossEntropyLoss`, `Adam`, `ReduceLROnPlateau`) |
| **Day 5** | TensorBoard logging, early stopping, checkpoint saving | `train.py` (`SummaryWriter`, `EarlyStopping`, `.pt` save/resume) |
| **Day 6** | Confusion matrix, per-class F1, ROC-AUC curve | `evaluate.py` (`reports/confusion_matrix.png`, `roc_curve.png`) |
| **Day 7** | Inference script (`predict.py`), reports, README, demo app | `predict.py`, `app.py`, `README.md`, `reports/` |

---
*Developed for Code A Nova Machine Learning Internship 2026.*
