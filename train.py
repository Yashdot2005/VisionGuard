"""
VisionGuard - Model Training Pipeline
Day 4 & 5 Milestones: Loss computation, optimizer, LR scheduler, validation, early stopping,
checkpoint saving/resuming, and TensorBoard logging.

Features:
- CrossEntropyLoss with dynamic class weighting
- Adam optimizer with differential learning rates for transfer learning
- ReduceLROnPlateau learning rate scheduler
- Early stopping based on validation loss
- Checkpoint persistence (best model & last state for resuming)
- TensorBoard scalar logging (loss, accuracy, lr per epoch)
- Training curves visual generation (loss & accuracy plots saved as PNG)
"""

import os
import time
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt
import numpy as np

from dataset import get_dataloaders
from model import build_model, get_model_summary


class EarlyStopping:
    """Early stops training when validation loss stops improving."""

    def __init__(self, patience: int = 5, min_delta: float = 1e-4):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = float("inf")
        self.early_stop = False

    def __call__(self, val_loss: float) -> bool:
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        return self.early_stop


def compute_class_weights(dataset, device: torch.device) -> torch.Tensor:
    """Compute balanced class weights inversely proportional to class frequencies."""
    targets = [s[1] for s in dataset.samples]
    class_counts = np.bincount(targets)
    total_samples = len(targets)
    weights = total_samples / (len(class_counts) * class_counts.astype(np.float32))
    return torch.tensor(weights, dtype=torch.float, device=device)


def save_checkpoint(
    state: dict,
    is_best: bool,
    checkpoint_dir: str = "checkpoints"
) -> Tuple[str, Optional[str]]:
    """Save training checkpoint and update best model file."""
    ckpt_path = Path(checkpoint_dir)
    ckpt_path.mkdir(parents=True, exist_ok=True)

    last_path = str(ckpt_path / "last_checkpoint.pt")
    best_path = str(ckpt_path / "best_model.pt")

    torch.save(state, last_path)
    if is_best:
        torch.save(state, best_path)
        return last_path, best_path
    return last_path, None


def plot_training_curves(history: Dict[str, List[float]], output_path: str = "reports/training_curves.png"):
    """Plot and save Loss and Accuracy curves across training epochs."""
    out_dir = Path(output_path).parent
    out_dir.mkdir(parents=True, exist_ok=True)

    epochs = range(1, len(history["train_loss"]) + 1)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Loss subplot
    ax1.plot(epochs, history["train_loss"], "b-o", label="Train Loss", linewidth=2)
    ax1.plot(epochs, history["val_loss"], "r--s", label="Val Loss", linewidth=2)
    ax1.set_title("Training & Validation Loss", fontsize=13, fontweight="bold")
    ax1.set_xlabel("Epoch", fontsize=11)
    ax1.set_ylabel("Loss", fontsize=11)
    ax1.grid(True, linestyle="--", alpha=0.6)
    ax1.legend(fontsize=11)

    # Accuracy subplot
    ax2.plot(epochs, [a * 100 for a in history["train_acc"]], "b-o", label="Train Accuracy", linewidth=2)
    ax2.plot(epochs, [a * 100 for a in history["val_acc"]], "g--s", label="Val Accuracy", linewidth=2)
    ax2.set_title("Training & Validation Accuracy", fontsize=13, fontweight="bold")
    ax2.set_xlabel("Epoch", fontsize=11)
    ax2.set_ylabel("Accuracy (%)", fontsize=11)
    ax2.grid(True, linestyle="--", alpha=0.6)
    ax2.legend(fontsize=11)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"[VisionGuard Training] Training curves saved to '{output_path}'")


def train_epoch(
    model: nn.Module,
    dataloader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device
) -> Tuple[float, float]:
    """Execute one training epoch over the training dataloader."""
    model.train()
    running_loss = 0.0
    correct_preds = 0
    total_samples = 0

    for images, labels in dataloader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        _, preds = torch.max(outputs, 1)
        correct_preds += torch.sum(preds == labels.data).item()
        total_samples += images.size(0)

    epoch_loss = running_loss / total_samples
    epoch_acc = correct_preds / total_samples
    return epoch_loss, epoch_acc


def validate_epoch(
    model: nn.Module,
    dataloader,
    criterion: nn.Module,
    device: torch.device
) -> Tuple[float, float]:
    """Execute one evaluation pass over the validation dataloader."""
    model.eval()
    running_loss = 0.0
    correct_preds = 0
    total_samples = 0

    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * images.size(0)
            _, preds = torch.max(outputs, 1)
            correct_preds += torch.sum(preds == labels.data).item()
            total_samples += images.size(0)

    epoch_loss = running_loss / total_samples
    epoch_acc = correct_preds / total_samples
    return epoch_loss, epoch_acc


def train(
    data_dir: str = "data",
    architecture: str = "resnet18",
    epochs: int = 15,
    batch_size: int = 32,
    lr: float = 1e-4,
    patience: int = 5,
    checkpoint_dir: str = "checkpoints",
    log_dir: str = "logs/tensorboard",
    reports_dir: str = "reports",
    resume_path: Optional[str] = None,
    image_size: int = 224
) -> Dict[str, List[float]]:
    """Full VisionGuard training loop with evaluation, checkpointing, and logging."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n[VisionGuard Training] Initializing on Device: {device}")

    # Load DataLoaders
    dataloaders, dataset_sizes, class_names = get_dataloaders(
        data_dir=data_dir,
        batch_size=batch_size,
        image_size=image_size,
        num_workers=0
    )

    # Class balance weighting
    class_weights = compute_class_weights(dataloaders["train"].dataset, device)
    print(f"[VisionGuard Training] Computed class weights: {class_weights.tolist()}")
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    # Build model
    model = build_model(architecture=architecture, num_classes=len(class_names))
    model = model.to(device)
    total_p, train_p = get_model_summary(model)
    print(f"[VisionGuard Training] Model Architecture: {architecture.upper()} | Total: {total_p:,} | Trainable: {train_p:,}")

    # Optimizer & Scheduler
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr, weight_decay=1e-4)
    scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=2)
    early_stopping = EarlyStopping(patience=patience)

    # History & State
    history: Dict[str, List[float]] = {
        "train_loss": [],
        "val_loss": [],
        "train_acc": [],
        "val_acc": [],
        "lr": []
    }
    start_epoch = 1
    best_val_loss = float("inf")

    # Resume from checkpoint if specified
    if resume_path and os.path.isfile(resume_path):
        print(f"[VisionGuard Training] Resuming training from checkpoint: {resume_path}")
        checkpoint = torch.load(resume_path, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        if "scheduler_state_dict" in checkpoint and checkpoint["scheduler_state_dict"]:
            scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        start_epoch = checkpoint["epoch"] + 1
        best_val_loss = checkpoint.get("best_val_loss", float("inf"))
        if "history" in checkpoint:
            history = checkpoint["history"]
        print(f"[VisionGuard Training] Resumed from epoch {start_epoch - 1}. Best Val Loss: {best_val_loss:.4f}")

    # TensorBoard Writer
    writer = SummaryWriter(log_dir=log_dir)
    print(f"[VisionGuard Training] TensorBoard logging enabled at '{log_dir}'")

    print("\n" + "=" * 70)
    print(f"{'EPOCH':<8}{'TRAIN LOSS':<14}{'TRAIN ACC (%)':<16}{'VAL LOSS':<14}{'VAL ACC (%)':<14}{'LR':<10}")
    print("=" * 70)

    total_start_time = time.time()

    for epoch in range(start_epoch, start_epoch + epochs):
        current_lr = optimizer.param_groups[0]["lr"]

        t_loss, t_acc = train_epoch(model, dataloaders["train"], criterion, optimizer, device)
        v_loss, v_acc = validate_epoch(model, dataloaders["val"], criterion, device)

        # Scheduler step
        scheduler.step(v_loss)

        # Record history
        history["train_loss"].append(t_loss)
        history["val_loss"].append(v_loss)
        history["train_acc"].append(t_acc)
        history["val_acc"].append(v_acc)
        history["lr"].append(current_lr)

        # TensorBoard Logging
        writer.add_scalar("Loss/Train", t_loss, epoch)
        writer.add_scalar("Loss/Val", v_loss, epoch)
        writer.add_scalar("Accuracy/Train", t_acc, epoch)
        writer.add_scalar("Accuracy/Val", v_acc, epoch)
        writer.add_scalar("LearningRate", current_lr, epoch)

        # Console Progress
        is_best = v_loss < best_val_loss
        best_marker = " * BEST" if is_best else ""
        print(f"{epoch:<8}{t_loss:<14.4f}{t_acc * 100:<16.2f}{v_loss:<14.4f}{v_acc * 100:<14.2f}{current_lr:<10.6f}{best_marker}")

        if is_best:
            best_val_loss = v_loss

        # Save checkpoint
        checkpoint_state = {
            "epoch": epoch,
            "architecture": architecture,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "best_val_loss": best_val_loss,
            "history": history,
            "class_names": class_names,
            "class_to_idx": dataloaders["train"].dataset.class_to_idx,
            "image_size": image_size,
        }
        save_checkpoint(checkpoint_state, is_best=is_best, checkpoint_dir=checkpoint_dir)

        # Check Early Stopping
        if early_stopping(v_loss):
            print(f"\n[VisionGuard Training] Early stopping triggered after {epoch} epochs (patience={patience}).")
            break

    total_duration = time.time() - total_start_time
    print("=" * 70)
    print(f"[VisionGuard Training] Completed in {total_duration / 60:.2f} minutes.")
    print(f"[VisionGuard Training] Best Validation Loss: {best_val_loss:.4f}")

    writer.close()

    # Generate training curves plot
    plot_training_curves(history, output_path=f"{reports_dir}/training_curves.png")

    return history


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="VisionGuard Model Training")
    parser.add_argument("--data_dir", type=str, default="data", help="Root data directory")
    parser.add_argument("--architecture", type=str, default="resnet18", choices=["resnet18", "custom_cnn"])
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--patience", type=int, default=5, help="Early stopping patience")
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints", help="Directory to save checkpoints")
    parser.add_argument("--log_dir", type=str, default="logs/tensorboard", help="TensorBoard log directory")
    parser.add_argument("--reports_dir", type=str, default="reports", help="Directory for reports & curves")
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume training from")
    parser.add_argument("--image_size", type=int, default=224, help="Input image resolution (e.g. 128, 224)")

    args = parser.parse_args()

    train(
        data_dir=args.data_dir,
        architecture=args.architecture,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        patience=args.patience,
        checkpoint_dir=args.checkpoint_dir,
        log_dir=args.log_dir,
        reports_dir=args.reports_dir,
        resume_path=args.resume,
        image_size=args.image_size
    )
