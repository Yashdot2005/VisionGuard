"""
VisionGuard - Dataset Preparation & Class Audit
Day 1 Milestone: Folder structure, train/val/test splits (70/15/15), class audit.

Supports:
1. High-fidelity synthetic manufacturing defect image generation (metal/machined surface).
2. Reorganization of open-source MVTec AD datasets into standard 70/15/15 splits.
3. Automated class distribution auditing and verification.
"""

import os
import random
import shutil
import argparse
from pathlib import Path
from typing import Dict, Tuple, List

import cv2
import numpy as np
from PIL import Image


def generate_normal_surface(width: int = 256, height: int = 256) -> np.ndarray:
    """Generate a realistic brushed metal/machined component surface.
    
    Includes directional grain, subtle illumination gradient, and micro-texture.
    A fraction of "normal" surfaces also carry faint, within-tolerance blemishes
    (light handling marks, dust, minor sheen variation) so the normal class is not
    a perfectly clean background — this keeps the classification task realistic
    instead of trivially separable from the defective class.
    """
    # Base metallic gray with slight tone variation
    base_color = random.randint(170, 205)
    img = np.full((height, width), base_color, dtype=np.float32)

    # Directional machining/brush marks (horizontal or slight angle)
    grain_noise = np.random.normal(0, random.uniform(8, 14), (height, 1)).astype(np.float32)
    grain = np.repeat(grain_noise, width, axis=1)

    # Add Gaussian micro-roughness
    micro_texture = np.random.normal(0, 4.0, (height, width)).astype(np.float32)

    # Lighting gradient (vignetting / cylindrical reflection)
    x = np.linspace(-1, 1, width)
    y = np.linspace(-1, 1, height)
    xx, yy = np.meshgrid(x, y)
    light_gradient = -25.0 * (xx ** 2) + random.uniform(-10, 10) * yy

    img = img + grain + micro_texture + light_gradient
    img = np.clip(img, 20, 245).astype(np.uint8)

    # Convert to 3-channel BGR
    img_bgr = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    # Subtle industrial tint (e.g. slight warm or cool steel tint)
    tint = np.array([random.uniform(0.95, 1.05), random.uniform(0.97, 1.02), random.uniform(0.98, 1.03)])
    img_bgr = np.clip(img_bgr * tint, 0, 255).astype(np.uint8)

    # ~35% of normal parts carry a faint, within-tolerance blemish so the model
    # must learn defect *severity*, not just "any mark present".
    if random.random() < 0.35:
        img_bgr = add_benign_blemish(img_bgr)

    return img_bgr


def add_benign_blemish(img: np.ndarray) -> np.ndarray:
    """Add a faint, low-contrast mark that mimics normal handling wear
    (light dust, a shallow sheen variation, a faint smudge) at an intensity
    that stays within acceptable QA tolerance — i.e. it should NOT flip the
    label to defective, but it does make the surface visually noisier and
    closer in appearance to a genuine (low-severity) defect.
    """
    h, w, _ = img.shape
    out = img.copy()
    kind = random.choice(["faint_line", "faint_spot", "sheen"])

    if kind == "faint_line":
        x1, y1 = random.randint(15, w - 15), random.randint(15, h - 15)
        length = random.randint(20, 60)
        angle = random.uniform(0, 2 * np.pi)
        x2 = int(np.clip(x1 + length * np.cos(angle), 0, w - 1))
        y2 = int(np.clip(y1 + length * np.sin(angle), 0, h - 1))
        overlay = out.copy()
        shade = random.randint(150, 175)
        cv2.line(overlay, (x1, y1), (x2, y2), (shade, shade, shade), 1, lineType=cv2.LINE_AA)
        alpha = random.uniform(0.15, 0.35)
        out = cv2.addWeighted(overlay, alpha, out, 1 - alpha, 0)
    elif kind == "faint_spot":
        cx, cy = random.randint(20, w - 20), random.randint(20, h - 20)
        mask = np.zeros((h, w), dtype=np.uint8)
        cv2.circle(mask, (cx, cy), random.randint(4, 9), 255, -1)
        mask = cv2.GaussianBlur(mask, (15, 15), 6)
        norm_mask = (mask / 255.0)[:, :, np.newaxis] * random.uniform(0.10, 0.25)
        shade = np.array([random.randint(140, 175)] * 3, dtype=np.float32)
        out = np.clip(out.astype(np.float32) * (1 - norm_mask) + shade * norm_mask, 0, 255).astype(np.uint8)
    else:  # sheen
        cx, cy = random.randint(30, w - 30), random.randint(30, h - 30)
        mask = np.zeros((h, w), dtype=np.uint8)
        cv2.ellipse(mask, (cx, cy), (random.randint(30, 55), random.randint(20, 40)), random.randint(0, 180), 0, 360, 255, -1)
        mask = cv2.GaussianBlur(mask, (41, 41), 15)
        norm_mask = (mask / 255.0)[:, :, np.newaxis] * random.uniform(0.08, 0.18)
        brighten = out.astype(np.float32) * (1 + norm_mask)
        out = np.clip(brighten, 0, 255).astype(np.uint8)

    return out


def inject_defects(img: np.ndarray) -> Tuple[np.ndarray, str]:
    """Inject realistic manufacturing defects into a normal surface.
    
    Defect types:
    - scratch: Sharp curvilinear abrasion with groove shadow & ridge highlight
    - crack: Jagged, branching fracture line with depth
    - pit: Pore / crater / slag inclusion depression
    - corrosion: Oxidized blotch / thermal discoloration
    - contaminant: Foreign particulate / oil stain
    """
    defect_img = img.copy()
    h, w, _ = defect_img.shape
    defect_type = random.choice(["scratch", "crack", "pit", "corrosion", "contaminant"])

    # Severity controls how visually obvious the defect is: low-severity defects
    # are near the decision boundary with benign blemishes, which is what makes
    # >90% accuracy meaningful instead of a trivially separable dataset.
    severity = random.uniform(0.35, 1.0)

    if defect_type == "scratch":
        # Draw 1-3 sharp curvilinear scratches
        num_scratches = random.randint(1, 3)
        for _ in range(num_scratches):
            start_x, start_y = random.randint(20, w - 20), random.randint(20, h - 20)
            length = random.randint(40, 130)
            angle = random.uniform(0, 2 * np.pi)
            end_x = int(start_x + length * np.cos(angle))
            end_y = int(start_y + length * np.sin(angle))
            ctrl_x = int((start_x + end_x) / 2 + random.randint(-25, 25))
            ctrl_y = int((start_y + end_y) / 2 + random.randint(-25, 25))

            # Sample quadratic Bezier curve points
            t = np.linspace(0, 1, num=50)
            bx = ((1 - t) ** 2 * start_x + 2 * (1 - t) * t * ctrl_x + t ** 2 * end_x).astype(int)
            by = ((1 - t) ** 2 * start_y + 2 * (1 - t) * t * ctrl_y + t ** 2 * end_y).astype(int)

            pts = np.vstack((bx, by)).T
            pts = pts[(pts[:, 0] >= 0) & (pts[:, 0] < w) & (pts[:, 1] >= 0) & (pts[:, 1] < h)]

            # Shadow groove (dark line)
            cv2.polylines(defect_img, [pts], isClosed=False, color=(30, 30, 30), thickness=2, lineType=cv2.LINE_AA)
            # Parallel highlight ridge
            highlight_pts = pts + np.array([1, 1])
            highlight_pts = highlight_pts[(highlight_pts[:, 0] >= 0) & (highlight_pts[:, 0] < w) & (highlight_pts[:, 1] >= 0) & (highlight_pts[:, 1] < h)]
            cv2.polylines(defect_img, [highlight_pts], isClosed=False, color=(240, 240, 240), thickness=1, lineType=cv2.LINE_AA)

    elif defect_type == "crack":
        # Jagged fracture line with multiple branch segments
        cur_x, cur_y = random.randint(40, w - 40), random.randint(40, h - 40)
        steps = random.randint(15, 30)
        crack_points = [(cur_x, cur_y)]
        for _ in range(steps):
            cur_x += random.randint(-8, 8)
            cur_y += random.randint(3, 10)
            cur_x = np.clip(cur_x, 5, w - 5)
            cur_y = np.clip(cur_y, 5, h - 5)
            crack_points.append((cur_x, cur_y))

        pts = np.array(crack_points, dtype=np.int32).reshape((-1, 1, 2))
        cv2.polylines(defect_img, [pts], isClosed=False, color=(20, 20, 20), thickness=2, lineType=cv2.LINE_AA)

        # Minor branching crack
        branch_start = crack_points[len(crack_points) // 2]
        branch_pts = [branch_start]
        bx, by = branch_start
        for _ in range(random.randint(6, 12)):
            bx += random.randint(4, 10)
            by += random.randint(-4, 4)
            bx = np.clip(bx, 5, w - 5)
            by = np.clip(by, 5, h - 5)
            branch_pts.append((bx, by))
        cv2.polylines(defect_img, [np.array(branch_pts, dtype=np.int32)], isClosed=False, color=(25, 25, 25), thickness=1, lineType=cv2.LINE_AA)

    elif defect_type == "pit":
        # Cluster of micro-pits or 1-2 major porosity voids
        num_pits = random.randint(2, 6)
        center_x, center_y = random.randint(50, w - 50), random.randint(50, h - 50)
        for _ in range(num_pits):
            px = int(np.clip(center_x + random.randint(-20, 20), 10, w - 10))
            py = int(np.clip(center_y + random.randint(-20, 20), 10, h - 10))
            radius = random.randint(3, 10)
            # Dark cavity
            cv2.circle(defect_img, (px, py), radius, (25, 25, 25), -1, lineType=cv2.LINE_AA)
            # Asymmetric highlight crescent simulating depth
            cv2.ellipse(defect_img, (px + 1, py + 1), (radius, radius // 2), 45, 0, 180, (230, 230, 230), 1, lineType=cv2.LINE_AA)

    elif defect_type == "corrosion":
        # Irregular oxidation/discoloration blob
        cx, cy = random.randint(60, w - 60), random.randint(60, h - 60)
        mask = np.zeros((h, w), dtype=np.uint8)
        cv2.ellipse(mask, (cx, cy), (random.randint(20, 45), random.randint(15, 35)), random.randint(0, 180), 0, 360, 255, -1)
        mask = cv2.GaussianBlur(mask, (31, 31), 11)
        normalized_mask = (mask / 255.0)[:, :, np.newaxis]

        corrosion_color = np.array([random.randint(25, 45), random.randint(50, 95), random.randint(140, 210)], dtype=np.float32)
        blended = defect_img.astype(np.float32) * (1.0 - normalized_mask * 0.7) + corrosion_color * (normalized_mask * 0.7)
        defect_img = np.clip(blended, 0, 255).astype(np.uint8)

    elif defect_type == "contaminant":
        # Dark speck / oil smudge / foreign debris
        cx, cy = random.randint(50, w - 50), random.randint(50, h - 50)
        num_splatters = random.randint(4, 9)
        for _ in range(num_splatters):
            sx = int(np.clip(cx + random.randint(-25, 25), 10, w - 10))
            sy = int(np.clip(cy + random.randint(-25, 25), 10, h - 10))
            sr = random.randint(2, 7)
            cv2.circle(defect_img, (sx, sy), sr, (random.randint(15, 40), random.randint(15, 40), random.randint(15, 40)), -1, lineType=cv2.LINE_AA)

    # Blend the fully-rendered defect back toward the clean surface by severity,
    # so low-severity samples are genuinely hard (defect present but subtle)
    # rather than every defective image being maximally obvious.
    blended = img.astype(np.float32) * (1 - severity) + defect_img.astype(np.float32) * severity
    defect_img = np.clip(blended, 0, 255).astype(np.uint8)

    return defect_img, defect_type


def create_synthetic_dataset(
    output_dir: str = "data",
    samples_per_class: int = 500,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42
) -> Dict[str, Dict[str, int]]:
    """Generate a reproducible synthetic manufacturing dataset split into train/val/test (70/15/15)."""
    assert abs((train_ratio + val_ratio + test_ratio) - 1.0) < 1e-5, "Splits must sum to 1.0"
    random.seed(seed)
    np.random.seed(seed)

    base_path = Path(output_dir)
    splits = ["train", "val", "test"]
    classes = ["normal", "defective"]

    # Clear and recreate destination directories
    for s in splits:
        for c in classes:
            (base_path / s / c).mkdir(parents=True, exist_ok=True)

    n_train = int(samples_per_class * train_ratio)
    n_val = int(samples_per_class * val_ratio)
    n_test = samples_per_class - n_train - n_val

    split_counts = {
        "train": n_train,
        "val": n_val,
        "test": n_test
    }

    print(f"\n[VisionGuard DataPrep] Generating synthetic dataset at '{output_dir}'...")
    print(f"Target distribution per class: Train={n_train}, Val={n_val}, Test={n_test} (Total: {samples_per_class})")

    # Generate normal and defective images
    for split_name, count in split_counts.items():
        for i in range(count):
            # Normal image
            normal_img = generate_normal_surface()
            normal_filename = base_path / split_name / "normal" / f"normal_{split_name}_{i:04d}.png"
            cv2.imwrite(str(normal_filename), normal_img)

            # Defective image (apply defect injection)
            defect_img, d_type = inject_defects(normal_img)
            defect_filename = base_path / split_name / "defective" / f"defect_{d_type}_{split_name}_{i:04d}.png"
            cv2.imwrite(str(defect_filename), defect_img)

    # Also save dedicated test sample images in sample_images/ for quick demo/CLI testing
    samples_dir = Path("sample_images")
    samples_dir.mkdir(exist_ok=True)
    sample_norm = generate_normal_surface()
    sample_def, d_type = inject_defects(sample_norm)
    cv2.imwrite(str(samples_dir / "test_sample_normal.jpg"), sample_norm)
    cv2.imwrite(str(samples_dir / "test_sample_defective.jpg"), sample_def)
    print(f"[VisionGuard DataPrep] Exported test samples to '{samples_dir}/'")

    audit_results = audit_dataset(output_dir)
    return audit_results


def audit_dataset(data_dir: str = "data") -> Dict[str, Dict[str, int]]:
    """Audit class counts across train, val, and test splits and print summary table."""
    base_path = Path(data_dir)
    splits = ["train", "val", "test"]
    classes = ["normal", "defective"]

    results: Dict[str, Dict[str, int]] = {}
    total_samples = 0

    print("\n" + "=" * 62)
    print(f"{'VISIONGUARD DATASET AUDIT (70/15/15 Target)':^62}")
    print("=" * 62)
    print(f"{'Split':<12}{'Normal':<12}{'Defective':<14}{'Total':<12}{'Ratio (%)':<12}")
    print("-" * 62)

    split_totals = {}
    for s in splits:
        results[s] = {}
        for c in classes:
            folder = base_path / s / c
            if folder.exists():
                count = len([f for f in folder.iterdir() if f.suffix.lower() in [".png", ".jpg", ".jpeg", ".bmp"]])
            else:
                count = 0
            results[s][c] = count
        split_total = sum(results[s].values())
        split_totals[s] = split_total
        total_samples += split_total

    for s in splits:
        s_total = split_totals[s]
        ratio_pct = (s_total / total_samples * 100) if total_samples > 0 else 0.0
        print(f"{s:<12}{results[s].get('normal', 0):<12}{results[s].get('defective', 0):<14}{s_total:<12}{ratio_pct:>8.1f}%")

    print("-" * 62)
    print(f"{'TOTAL':<12}{sum(results[s].get('normal', 0) for s in splits):<12}{sum(results[s].get('defective', 0) for s in splits):<14}{total_samples:<12}{100.0:>8.1f}%")
    print("=" * 62 + "\n")

    return results


def organize_mvtec_dataset(mvtec_dir: str, output_dir: str = "data", train_ratio: float = 0.70, val_ratio: float = 0.15) -> Dict[str, Dict[str, int]]:
    """Reorganize an MVTec AD category folder into the VisionGuard 70/15/15 format.
    
    MVTec structure:
    mvtec_dir/
      train/good/
      test/good/
      test/<defect_1>/
      test/<defect_2>/
    """
    src = Path(mvtec_dir)
    dst = Path(output_dir)
    assert src.exists(), f"Source path {mvtec_dir} does not exist"

    # Collect normal images
    normal_images: List[Path] = []
    if (src / "train" / "good").exists():
        normal_images.extend(list((src / "train" / "good").glob("*.*")))
    if (src / "test" / "good").exists():
        normal_images.extend(list((src / "test" / "good").glob("*.*")))

    # Collect defective images
    defect_images: List[Path] = []
    test_dir = src / "test"
    if test_dir.exists():
        for d in test_dir.iterdir():
            if d.is_dir() and d.name != "good":
                defect_images.extend(list(d.glob("*.*")))

    random.seed(42)
    random.shuffle(normal_images)
    random.shuffle(defect_images)

    def split_and_copy(img_list: List[Path], class_label: str):
        total = len(img_list)
        n_train = int(total * train_ratio)
        n_val = int(total * val_ratio)
        
        train_imgs = img_list[:n_train]
        val_imgs = img_list[n_train:n_train + n_val]
        test_imgs = img_list[n_train + n_val:]

        for split_name, imgs in [("train", train_imgs), ("val", val_imgs), ("test", test_imgs)]:
            target_folder = dst / split_name / class_label
            target_folder.mkdir(parents=True, exist_ok=True)
            for img_path in imgs:
                shutil.copy(str(img_path), str(target_folder / img_path.name))

    split_and_copy(normal_images, "normal")
    split_and_copy(defect_images, "defective")

    return audit_dataset(output_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="VisionGuard Dataset Preparation & Audit")
    parser.add_argument("--generate", action="store_true", help="Generate synthetic manufacturing defect dataset")
    parser.add_argument("--samples_per_class", type=int, default=500, help="Number of images per class (default: 500)")
    parser.add_argument("--output_dir", type=str, default="data", help="Output directory for data splits")
    parser.add_argument("--audit", action="store_true", help="Run class audit on existing dataset")
    parser.add_argument("--mvtec_dir", type=str, default=None, help="Path to MVTec AD category folder to organize")

    args = parser.parse_args()

    if args.mvtec_dir:
        organize_mvtec_dataset(args.mvtec_dir, args.output_dir)
    elif args.generate:
        create_synthetic_dataset(output_dir=args.output_dir, samples_per_class=args.samples_per_class)
    elif args.audit:
        audit_dataset(args.output_dir)
    else:
        # Default: generate synthetic dataset with audit
        create_synthetic_dataset(output_dir=args.output_dir, samples_per_class=args.samples_per_class)
