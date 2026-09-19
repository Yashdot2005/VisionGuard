"""
VisionGuard - Single-Image & Batch CLI Inference Script
Day 7 Milestone: predict.py for single-image classification.

Features:
- Single-image CLI prediction on arbitrary images without requiring dataset.
- Clear formatted probability and confidence breakdown.
- Latency measurement.
- Optional visual inspection overlay banner (Green: NORMAL, Red: DEFECTIVE).
- Batch folder prediction mode for assembly-line batches.
"""

import os
import sys
import time
import argparse
from pathlib import Path
from typing import Dict, Any, Optional

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFont
import numpy as np

from dataset import load_and_preprocess_image
from model import build_model


def predict_single_image(
    image_path: str,
    checkpoint_path: str = "checkpoints/best_model.pt",
    device: Optional[torch.device] = None,
    save_vis: Optional[str] = None
) -> Dict[str, Any]:
    """Perform single-image inference and return formatted prediction dictionary."""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(f"Model checkpoint not found at: {checkpoint_path}")

    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    architecture = checkpoint.get("architecture", "resnet18")
    class_names = checkpoint.get("class_names", ["defective", "normal"])
    class_to_idx = checkpoint.get("class_to_idx", {"defective": 0, "normal": 1})
    idx_to_class = {v: k for k, v in class_to_idx.items()}
    image_size = checkpoint.get("image_size", 224)

    # Initialize model
    model = build_model(architecture=architecture, num_classes=len(class_names))
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    model.eval()

    # Preprocess image
    tensor = load_and_preprocess_image(image_path, image_size=image_size).to(device)

    # Measure latency
    start_time = time.perf_counter()
    with torch.no_grad():
        outputs = model(tensor)
        probabilities = F.softmax(outputs, dim=1).squeeze(0)
        confidence, predicted_idx = torch.max(probabilities, dim=0)
    end_time = time.perf_counter()

    latency_ms = (end_time - start_time) * 1000.0
    pred_label = idx_to_class.get(predicted_idx.item(), class_names[predicted_idx.item()])
    conf_pct = confidence.item() * 100.0

    prob_dict = {
        class_names[i]: float(probabilities[i].item())
        for i in range(len(class_names))
    }

    # Optional visualization overlay
    if save_vis:
        create_prediction_overlay(
            image_path=image_path,
            pred_label=pred_label,
            confidence_pct=conf_pct,
            output_path=save_vis
        )

    return {
        "image_path": str(image_path),
        "prediction": pred_label,
        "confidence_pct": conf_pct,
        "probabilities": prob_dict,
        "latency_ms": latency_ms,
        "status": "FAIL" if pred_label.lower() == "defective" else "PASS"
    }


def create_prediction_overlay(
    image_path: str,
    pred_label: str,
    confidence_pct: float,
    output_path: str
):
    """Draw quality inspection banner onto the image."""
    img = Image.open(image_path).convert("RGB")
    w, h = img.size

    # Banner dimensions
    banner_height = max(50, int(h * 0.18))
    canvas = Image.new("RGB", (w, h + banner_height), (20, 24, 33))
    canvas.paste(img, (0, 0))

    draw = ImageDraw.Draw(canvas)
    is_defective = (pred_label.lower() == "defective")
    badge_color = (220, 38, 38) if is_defective else (16, 185, 129)  # Red vs Green
    status_text = f"[{'FAIL - DEFECT DETECTED' if is_defective else 'PASS - QUALITY OK'}]"

    # Banner header background
    draw.rectangle([(0, h), (w, h + banner_height)], fill=(24, 28, 38))
    # Accent indicator line
    draw.rectangle([(0, h), (w, h + 4)], fill=badge_color)

    # Text summary
    line1 = f"PREDICTION: {pred_label.upper()} {status_text}"
    line2 = f"CONFIDENCE: {confidence_pct:.2f}%"
    draw.text((12, h + 10), line1, fill=badge_color)
    draw.text((12, h + 28), line2, fill=(240, 240, 240))

    out_p = Path(output_path)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(str(out_p))
    print(f"[VisionGuard Predict] Visual overlay saved to '{output_path}'")


def print_prediction_report(result: Dict[str, Any]):
    """Format prediction cleanly to terminal."""
    is_defect = result["prediction"].lower() == "defective"
    status_icon = "[FAIL] DEFECTIVE [REJECT]" if is_defect else "[PASS] NORMAL [APPROVE]"

    print("\n" + "=" * 55)
    print(f"{'VISIONGUARD INSPECTION REPORT':^55}")
    print("=" * 55)
    print(f"Target Image : {result['image_path']}")
    print(f"Decision     : {status_icon}")
    print(f"Confidence   : {result['confidence_pct']:.2f}%")
    print(f"Latency      : {result['latency_ms']:.2f} ms")
    print("-" * 55)
    print("Class Probabilities:")
    for cls_name, prob in result["probabilities"].items():
        bar_len = int(prob * 25)
        bar = "#" * bar_len + "-" * (25 - bar_len)
        print(f"  {cls_name:<12}: {prob:>6.4f}  [{bar}]")
    print("=" * 55 + "\n")


def batch_predict(batch_dir: str, checkpoint_path: str = "checkpoints/best_model.pt"):
    """Run batch inference over all images in a directory."""
    folder = Path(batch_dir)
    image_files = [f for f in folder.glob("*.*") if f.suffix.lower() in [".jpg", ".jpeg", ".png", ".bmp"]]

    if not image_files:
        print(f"No valid image files found in {batch_dir}")
        return

    print(f"\n[VisionGuard Batch] Processing {len(image_files)} images from '{batch_dir}'...")
    defects = 0
    normals = 0

    for img_path in image_files:
        res = predict_single_image(str(img_path), checkpoint_path)
        if res["prediction"].lower() == "defective":
            defects += 1
            icon = "[FAIL]"
        else:
            normals += 1
            icon = "[PASS]"
        print(f"  {icon} {img_path.name:<28} -> {res['prediction'].upper()} ({res['confidence_pct']:.1f}%) | {res['latency_ms']:.1f}ms")

    print("-" * 55)
    print(f"Batch Summary: {len(image_files)} processed | {normals} Normal | {defects} Defective")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="VisionGuard Single-Image & Batch Inference")
    parser.add_argument("--image", type=str, default=None, help="Path to single image for inference")
    parser.add_argument("--batch_dir", type=str, default=None, help="Directory of images for batch inference")
    parser.add_argument("--model", type=str, default="checkpoints/best_model.pt", help="Path to model checkpoint")
    parser.add_argument("--save_vis", type=str, default=None, help="Save image with prediction overlay banner")

    args = parser.parse_args()

    if args.image:
        res = predict_single_image(args.image, checkpoint_path=args.model, save_vis=args.save_vis)
        print_prediction_report(res)
    elif args.batch_dir:
        batch_predict(args.batch_dir, checkpoint_path=args.model)
    else:
        # Default test against sample image if available
        sample_img = "sample_images/test_sample_defective.jpg"
        if os.path.exists(sample_img):
            print(f"No image supplied. Running demonstration on '{sample_img}'...")
            res = predict_single_image(sample_img, checkpoint_path=args.model, save_vis="reports/sample_prediction.jpg")
            print_prediction_report(res)
        else:
            parser.print_help()
