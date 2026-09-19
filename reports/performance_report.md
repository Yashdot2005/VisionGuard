# VisionGuard — Manufacturing Defect Detection System
## Official Model Performance & Evaluation Report

**Model Architecture:** `CUSTOM_CNN`  
**Evaluation Set:** Test Split (90 samples)  
**Checkpoint:** `checkpoints/best_model.pt`  

---

### 1. Rubric Success Indicator Verification

| Metric | Target Rubric | Achieved Result | Status |
| :--- | :--- | :--- | :--- |
| **Accuracy** | **> 90.0%** | **97.78%** | ✅ PASSED |
| **Defect Class Recall** | **> 85.0%** | **95.56%** | ✅ PASSED |
| **ROC-AUC Score** | **> 0.920** | **1.0000** | ✅ PASSED |

**Overall Status:** **PASSED ALL RUBRIC CRITERIA**

---

### 2. Comprehensive Per-Class Performance Metrics

| Class Label | Precision | Recall | F1-Score | Support |
| :--- | :--- | :--- | :--- | :--- |
| **Defective** | 100.00% | 95.56% | 97.73% | 45 |
| **Normal** | 95.74% | 100.00% | 97.83% | 45 |
| **Macro Average** | 97.87% | 97.78% | 97.78% | 90 |
| **Weighted Average** | - | - | 97.78% | 90 |

---

### 3. Confusion Matrix Breakdown

- **True Negatives (Normal correctly identified):** 45
- **False Positives (Normal falsely flagged as Defective):** 0
- **False Negatives (Defective missed / Escapes):** 2
- **True Positives (Defects accurately detected):** 43

### 4. Evaluation Visualizations

- **Confusion Matrix:** Saved to `reports/confusion_matrix.png`
- **ROC Curve:** Saved to `reports/roc_curve.png`
- **Training Trajectories:** Saved to `reports/training_curves.png`
