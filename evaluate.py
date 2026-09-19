"""
VisionGuard - Model Evaluation & Reporting
Day 6 Milestone: Confusion matrix, per-class F1 score, ROC-AUC curve generation.

Features:
- Evaluates trained checkpoint on test dataset
- Calculates Accuracy, Precision, Recall, F1-Score, and ROC-AUC
- Generates Confusion Matrix plot (reports/confusion_matrix.png)
- Generates ROC-AUC curve plot (reports/roc_curve.png)
- Saves comprehensive Performance Report (reports/performance_report.md)
- Verifies rubric criteria: Accuracy >90%, Defect Recall >85%, AUC >0.92
"""

import os
import json
import argparse
from pathlib import Path
from typing import Dict, Any, Tuple

import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    roc_curve,
    confusion_matrix,
    classification_report
)

from dataset import get_dataloaders
from model import build_model


def evaluate_model(
    checkpoint_path: str = "checkpoints/best_model.pt",
    data_dir: str = "data",
    reports_dir: str = "reports",
    batch_size: int = 32
) -> Dict[str, Any]:
    """Run full evaluation on test split and generate all artifacts."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n[VisionGuard Evaluation] Evaluating on Device: {device}")
    
    reports_path = Path(reports_dir)
    reports_path.mkdir(parents=True, exist_ok=True)

    # Load checkpoint
    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found at: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=device)
    architecture = checkpoint.get("architecture", "resnet18")
    class_names = checkpoint.get("class_names", ["defective", "normal"])
    class_to_idx = checkpoint.get("class_to_idx", {"defective": 0, "normal": 1})
    defect_idx = class_to_idx.get("defective", 0)
    image_size = checkpoint.get("image_size", 224)

    print(f"[VisionGuard Evaluation] Loaded checkpoint: {checkpoint_path}")
    print(f"  - Architecture: {architecture.upper()}")
    print(f"  - Trained Epochs: {checkpoint.get('epoch', 'N/A')}")
    print(f"  - Classes: {class_names} (defect index: {defect_idx})")
    print(f"  - Image Size: {image_size}")

    # Load test data
    dataloaders, sizes, _ = get_dataloaders(data_dir=data_dir, batch_size=batch_size, image_size=image_size, num_workers=0)
    test_loader = dataloaders["test"]

    # Build and load model
    model = build_model(architecture=architecture, num_classes=len(class_names))
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    model.eval()

    all_targets = []
    all_preds = []
    all_probs = []

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            outputs = model(images)
            probs = F.softmax(outputs, dim=1)

            _, preds = torch.max(outputs, 1)

            all_targets.extend(labels.cpu().numpy().tolist())
            all_preds.extend(preds.cpu().numpy().tolist())
            all_probs.extend(probs.cpu().numpy().tolist())

    y_true = np.array(all_targets)
    y_pred = np.array(all_preds)
    y_prob = np.array(all_probs)

    # Compute Metrics
    accuracy = accuracy_score(y_true, y_pred)
    
    # Binary metrics specifically targeting the 'defective' class (positive label)
    defect_pos_label = defect_idx
    precision_defect = precision_score(y_true, y_pred, pos_label=defect_pos_label, zero_division=0)
    recall_defect = recall_score(y_true, y_pred, pos_label=defect_pos_label, zero_division=0)
    f1_defect = f1_score(y_true, y_pred, pos_label=defect_pos_label, zero_division=0)

    # Normal class metrics
    normal_idx = 1 if defect_idx == 0 else 0
    precision_normal = precision_score(y_true, y_pred, pos_label=normal_idx, zero_division=0)
    recall_normal = recall_score(y_true, y_pred, pos_label=normal_idx, zero_division=0)
    f1_normal = f1_score(y_true, y_pred, pos_label=normal_idx, zero_division=0)

    # Macro and weighted F1
    f1_macro = f1_score(y_true, y_pred, average="macro", zero_division=0)
    f1_weighted = f1_score(y_true, y_pred, average="weighted", zero_division=0)

    # ROC-AUC calculation (using probabilities for class 1 if binary)
    # If defect is index 0, positive class for AUC could be defect probability
    defect_probabilities = y_prob[:, defect_idx]
    # For standard binary roc_auc_score, encode true targets as 1 for defect, 0 for normal
    y_true_defect_binary = (y_true == defect_idx).astype(int)
    roc_auc = roc_auc_score(y_true_defect_binary, defect_probabilities)

    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)

    # 1. Save Confusion Matrix Plot
    cm_path = reports_path / "confusion_matrix.png"
    plt.figure(figsize=(7, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
        cbar=True,
        annot_kws={"size": 14, "weight": "bold"}
    )
    plt.title("VisionGuard Confusion Matrix (Test Set)", fontsize=14, fontweight="bold", pad=12)
    plt.xlabel("Predicted Class", fontsize=12)
    plt.ylabel("Ground Truth Class", fontsize=12)
    plt.tight_layout()
    plt.savefig(str(cm_path), dpi=300)
    plt.close()
    print(f"[VisionGuard Evaluation] Saved Confusion Matrix to '{cm_path}'")

    # 2. Save ROC-AUC Curve Plot
    roc_path = reports_path / "roc_curve.png"
    fpr, tpr, _ = roc_curve(y_true_defect_binary, defect_probabilities)
    plt.figure(figsize=(7, 6))
    plt.plot(fpr, tpr, color="#2563EB", lw=2.5, label=f"ROC Curve (AUC = {roc_auc:.4f})")
    plt.plot([0, 1], [0, 1], color="#9CA3AF", lw=1.5, linestyle="--", label="Random Chance (AUC = 0.50)")
    plt.xlim([-0.02, 1.02])
    plt.ylim([-0.02, 1.05])
    plt.xlabel("False Positive Rate (1 - Specificity)", fontsize=12)
    plt.ylabel("True Positive Rate (Recall / Sensitivity)", fontsize=12)
    plt.title("Receiver Operating Characteristic (ROC) Curve", fontsize=14, fontweight="bold", pad=12)
    plt.legend(loc="lower right", fontsize=11)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(str(roc_path), dpi=300)
    plt.close()
    print(f"[VisionGuard Evaluation] Saved ROC Curve to '{roc_path}'")

    # 3. Check Success Rubric Criteria
    rubric_accuracy_pass = accuracy >= 0.90
    rubric_recall_pass = recall_defect >= 0.85
    rubric_auc_pass = roc_auc >= 0.92
    overall_pass = rubric_accuracy_pass and rubric_recall_pass and rubric_auc_pass

    # 4. Generate Performance Report Markdown
    report_md_path = reports_path / "performance_report.md"
    report_content = f"""# VisionGuard — Manufacturing Defect Detection System
## Official Model Performance & Evaluation Report

**Model Architecture:** `{architecture.upper()}`  
**Evaluation Set:** Test Split ({len(y_true)} samples)  
**Checkpoint:** `{checkpoint_path}`  

---

### 1. Rubric Success Indicator Verification

| Metric | Target Rubric | Achieved Result | Status |
| :--- | :--- | :--- | :--- |
| **Accuracy** | **> 90.0%** | **{accuracy * 100:.2f}%** | {"✅ PASSED" if rubric_accuracy_pass else "❌ FAILED"} |
| **Defect Class Recall** | **> 85.0%** | **{recall_defect * 100:.2f}%** | {"✅ PASSED" if rubric_recall_pass else "❌ FAILED"} |
| **ROC-AUC Score** | **> 0.920** | **{roc_auc:.4f}** | {"✅ PASSED" if rubric_auc_pass else "❌ FAILED"} |

**Overall Status:** {"**PASSED ALL RUBRIC CRITERIA**" if overall_pass else "**NEEDS REFINEMENT**"}

---

### 2. Comprehensive Per-Class Performance Metrics

| Class Label | Precision | Recall | F1-Score | Support |
| :--- | :--- | :--- | :--- | :--- |
| **Defective** | {precision_defect * 100:.2f}% | {recall_defect * 100:.2f}% | {f1_defect * 100:.2f}% | {np.sum(y_true == defect_idx)} |
| **Normal** | {precision_normal * 100:.2f}% | {recall_normal * 100:.2f}% | {f1_normal * 100:.2f}% | {np.sum(y_true == normal_idx)} |
| **Macro Average** | {(precision_defect + precision_normal)/2 * 100:.2f}% | {(recall_defect + recall_normal)/2 * 100:.2f}% | {f1_macro * 100:.2f}% | {len(y_true)} |
| **Weighted Average** | - | - | {f1_weighted * 100:.2f}% | {len(y_true)} |

---

### 3. Confusion Matrix Breakdown

- **True Negatives (Normal correctly identified):** {cm[normal_idx, normal_idx]}
- **False Positives (Normal falsely flagged as Defective):** {cm[normal_idx, defect_idx]}
- **False Negatives (Defective missed / Escapes):** {cm[defect_idx, normal_idx]}
- **True Positives (Defects accurately detected):** {cm[defect_idx, defect_idx]}

### 4. Evaluation Visualizations

- **Confusion Matrix:** Saved to `reports/confusion_matrix.png`
- **ROC Curve:** Saved to `reports/roc_curve.png`
- **Training Trajectories:** Saved to `reports/training_curves.png`
"""
    with open(report_md_path, "w", encoding="utf-8") as f:
        f.write(report_content)
    print(f"[VisionGuard Evaluation] Performance report saved to '{report_md_path}'")

    # Print summary to stdout
    print("\n" + "=" * 65)
    print(f"{'VISIONGUARD TEST EVALUATION SUMMARY':^65}")
    print("=" * 65)
    print(f"  Test Accuracy         : {accuracy * 100:.2f}% (Target: >90.0% -> {'PASS' if rubric_accuracy_pass else 'FAIL'})")
    print(f"  Defect Recall         : {recall_defect * 100:.2f}% (Target: >85.0% -> {'PASS' if rubric_recall_pass else 'FAIL'})")
    print(f"  Defect Precision      : {precision_defect * 100:.2f}%")
    print(f"  Defect F1-Score       : {f1_defect * 100:.2f}%")
    print(f"  ROC-AUC Score         : {roc_auc:.4f} (Target: >0.920 -> {'PASS' if rubric_auc_pass else 'FAIL'})")
    print("=" * 65)
    print(f"  Confusion Matrix:\n{cm}")
    print("=" * 65 + "\n")

    return {
        "accuracy": accuracy,
        "precision_defect": precision_defect,
        "recall_defect": recall_defect,
        "f1_defect": f1_defect,
        "roc_auc": roc_auc,
        "confusion_matrix": cm.tolist()
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="VisionGuard Model Evaluation")
    parser.add_argument("--model", type=str, default="checkpoints/best_model.pt", help="Path to checkpoint file")
    parser.add_argument("--data_dir", type=str, default="data", help="Root data directory")
    parser.add_argument("--reports_dir", type=str, default="reports", help="Directory to save evaluation reports")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")

    args = parser.parse_args()
    evaluate_model(
        checkpoint_path=args.model,
        data_dir=args.data_dir,
        reports_dir=args.reports_dir,
        batch_size=args.batch_size
    )
