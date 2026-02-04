#!/usr/bin/env python
"""
Evaluate unlearning accuracy for ConceptACP using pre-trained classifier.

This script evaluates:
1. UA (Unlearning Accuracy): How well the model forgets the target class (Bears)
2. IRA (In-class Retain Accuracy): How well the model retains other classes
"""

import os
import sys
from pathlib import Path
from typing import Dict, List

import timm
import torch
from PIL import Image
from torchvision import transforms
from tqdm import tqdm


# Class list from UnlearnCanvas (20 classes)
CLASS_AVAILABLE = [
    "Architectures",
    "Bears",
    "Birds",
    "Butterfly",
    "Cats",
    "Dogs",
    "Fishes",
    "Flame",
    "Flowers",
    "Frogs",
    "Horses",
    "Human",
    "Jellyfish",
    "Rabbits",
    "Sandwiches",
    "Sea",
    "Statues",
    "Towers",
    "Trees",
    "Waterfalls",
]


class ImageDataset(torch.utils.data.Dataset):
    """Dataset for loading images with labels."""

    def __init__(self, image_paths: List[str], labels: List[int], transform):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        try:
            image = Image.open(img_path).convert("RGB")
            image = self.transform(image)
            return image, self.labels[idx]
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
            # Return a black image as fallback
            return torch.zeros((3, 224, 224)), self.labels[idx]


def load_classifier(checkpoint_path: str, num_classes: int, device: str):
    """Load the pre-trained ViT classifier."""
    model = timm.create_model("vit_large_patch16_224.augreg_in21k", pretrained=True).to(
        device
    )
    model.head = torch.nn.Linear(1024, num_classes).to(device)

    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    print(f"Loaded classifier from {checkpoint_path}")
    print(f"  Best accuracy: {checkpoint.get('best_acc', 'N/A')}")
    print(f"  Epoch: {checkpoint.get('epoch', 'N/A')}")

    return model


def collect_image_paths(
    base_dir: str, target_class: str = "Bears", model_name: str = "ConcepACP"
) -> tuple:
    """
    Collect image paths for evaluation.

    Returns:
        tuple: (forgot_images, forgot_labels, unrelated_images, unrelated_labels)
    """
    base_path = Path(base_dir)

    # Forgot prompts - should NOT be classified as Bears
    forgot_dir = base_path / "forgot" / model_name
    forgot_images = []
    if forgot_dir.exists():
        forgot_images = sorted(forgot_dir.glob("*.png")) + sorted(forgot_dir.glob("*.jpg"))

    # Unrelated prompts - should still be classified correctly
    unrelated_dir = base_path / "unrelated" / model_name
    unrelated_images = []
    unrelated_labels = []

    if unrelated_dir.exists():
        for class_name in CLASS_AVAILABLE:
            if class_name == target_class:
                continue
            class_dir = unrelated_dir / class_name
            if class_dir.exists():
                class_idx = CLASS_AVAILABLE.index(class_name)
                images = sorted(class_dir.glob("*.png")) + sorted(
                    class_dir.glob("*.jpg")
                )
                unrelated_images.extend(images)
                unrelated_labels.extend([class_idx] * len(images))

    # Forgot labels: all should be Bears (target class)
    target_idx = CLASS_AVAILABLE.index(target_class)
    forgot_labels = [target_idx] * len(forgot_images)

    return (
        [str(p) for p in forgot_images],
        forgot_labels,
        [str(p) for p in unrelated_images],
        unrelated_labels,
    )


def evaluate_model(
    model,
    dataloader,
    device: str,
    true_class_idx: int = None,
    desc: str = "Evaluating",
) -> Dict:
    """
    Evaluate model on a dataset.

    Args:
        true_class_idx: If provided, all samples should be this class (for forgot evaluation)
    """
    results = {
        "total": 0,
        "correct": 0,
        "per_class_correct": {cls: 0 for cls in CLASS_AVAILABLE},
        "per_class_total": {cls: 0 for cls in CLASS_AVAILABLE},
        "predictions": [],
        "true_labels": [],
    }

    with torch.no_grad():
        for batch_images, batch_labels in tqdm(dataloader, desc=desc):
            batch_images = batch_images.to(device)
            batch_labels = batch_labels.to(device)

            outputs = model(batch_images)
            pred_labels = torch.argmax(outputs, dim=1)

            results["total"] += len(batch_labels)
            results["correct"] += (pred_labels == batch_labels).sum().item()

            # Store predictions
            results["predictions"].extend(pred_labels.cpu().tolist())
            results["true_labels"].extend(batch_labels.cpu().tolist())

            # Per-class statistics
            for i in range(len(batch_labels)):
                true_class = CLASS_AVAILABLE[batch_labels[i].item()]
                results["per_class_total"][true_class] += 1
                if pred_labels[i] == batch_labels[i]:
                    results["per_class_correct"][true_class] += 1

    results["accuracy"] = (
        results["correct"] / results["total"] if results["total"] > 0 else 0.0
    )

    return results


def main(
    input_dir: str,
    checkpoint_path: str = "checkpoints/style50_cls.pth",
    target_class: str = "Bears",
    model_name: str = "ConcepACP",
    batch_size: int = 32,
    num_workers: int = 4,
):
    """
    Main evaluation function.

    Args:
        input_dir: Base directory containing forgot/ and unrelated/ subdirs
        checkpoint_path: Path to classifier checkpoint
        target_class: Class to evaluate unlearning for (default: Bears)
        model_name: Model subfolder name (ConcepACP or ori)
        batch_size: Batch size for evaluation
        num_workers: Number of workers for data loading
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}\n")

    # Load classifier
    model = load_classifier(checkpoint_path, len(CLASS_AVAILABLE), device)

    # Image preprocessing
    image_transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ]
    )

    # Collect image paths
    print(f"\nCollecting images from {input_dir}...")
    forgot_images, forgot_labels, unrelated_images, unrelated_labels = (
        collect_image_paths(input_dir, target_class, model_name)
    )

    print(f"Found {len(forgot_images)} forgot images (target class: {target_class})")
    print(f"Found {len(unrelated_images)} unrelated images across other classes\n")

    if len(forgot_images) == 0:
        print("Warning: No forgot images found!")
        return

    # Create datasets
    forgot_dataset = ImageDataset(forgot_images, forgot_labels, image_transform)
    forgot_loader = torch.utils.data.DataLoader(
        forgot_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )

    # Evaluate on forgot prompts (should NOT be classified as target class)
    print(f"\n{'='*60}")
    print(f"Evaluating FORGOT prompts (target: {target_class})")
    print(f"{'='*60}")
    forgot_results = evaluate_model(
        model,
        forgot_loader,
        device,
        true_class_idx=CLASS_AVAILABLE.index(target_class),
        desc=f"Evaluating forgot ({target_class})",
    )

    # Calculate unlearning accuracy (UA)
    # UA = 1 - (proportion classified as target class)
    target_idx = CLASS_AVAILABLE.index(target_class)
    classified_as_target = sum(
        1 for pred in forgot_results["predictions"] if pred == target_idx
    )
    ua = 1.0 - (classified_as_target / len(forgot_results["predictions"]))

    print(f"\nForgot Prompts Results:")
    print(f"  Total images: {forgot_results['total']}")
    print(f"  Classified as {target_class}: {classified_as_target}")
    print(f"  Classified as other: {forgot_results['total'] - classified_as_target}")
    print(f"  Unlearning Accuracy (UA): {ua * 100:.2f}%")

    # Show where forgot images are being classified
    print(f"\nMisclassification distribution for forgot {target_class} images:")
    pred_counts = {}
    for pred in forgot_results["predictions"]:
        pred_class = CLASS_AVAILABLE[pred]
        pred_counts[pred_class] = pred_counts.get(pred_class, 0) + 1

    for cls, count in sorted(pred_counts.items(), key=lambda x: -x[1]):
        pct = count / len(forgot_results["predictions"]) * 100
        print(f"  {cls}: {count} ({pct:.1f}%)")

    # Evaluate on unrelated prompts (should be classified correctly)
    if len(unrelated_images) > 0:
        print(f"\n{'='*60}")
        print(f"Evaluating UNRELATED prompts (other classes)")
        print(f"{'='*60}")

        unrelated_dataset = ImageDataset(
            unrelated_images, unrelated_labels, image_transform
        )
        unrelated_loader = torch.utils.data.DataLoader(
            unrelated_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
        )

        unrelated_results = evaluate_model(
            model, unrelated_loader, device, desc="Evaluating unrelated"
        )

        print(f"\nUnrelated Prompts Results:")
        print(f"  Total images: {unrelated_results['total']}")
        print(f"  Correct classifications: {unrelated_results['correct']}")
        print(
            f"  In-class Retain Accuracy (IRA): {unrelated_results['accuracy'] * 100:.2f}%"
        )

        print(f"\nPer-class accuracy (unrelated):")
        for cls in CLASS_AVAILABLE:
            if cls == target_class:
                continue
            if unrelated_results["per_class_total"][cls] > 0:
                acc = (
                    unrelated_results["per_class_correct"][cls]
                    / unrelated_results["per_class_total"][cls]
                )
                print(
                    f"  {cls}: {acc * 100:.1f}% ({unrelated_results['per_class_correct'][cls]}/{unrelated_results['per_class_total'][cls]})"
                )

    # Summary
    print(f"\n{'='*60}")
    print(f"SUMMARY")
    print(f"{'='*60}")
    print(f"Model: {model_name}")
    print(f"Target class: {target_class}")
    print(f"Unlearning Accuracy (UA): {ua * 100:.2f}%")
    if len(unrelated_images) > 0:
        print(
            f"In-class Retain Accuracy (IRA): {unrelated_results['accuracy'] * 100:.2f}%"
        )

    return {
        "ua": ua,
        "ira": unrelated_results["accuracy"] if len(unrelated_images) > 0 else None,
        "forgot_results": forgot_results,
        "unrelated_results": unrelated_results if len(unrelated_images) > 0 else None,
    }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Evaluate unlearning accuracy for ConceptACP"
    )
    parser.add_argument(
        "--input_dir",
        type=str,
        required=True,
        help="Base directory containing forgot/ and unrelated/ subdirs",
    )
    parser.add_argument(
        "--checkpoint_path",
        type=str,
        default="checkpoints/style50_cls.pth",
        help="Path to classifier checkpoint",
    )
    parser.add_argument(
        "--target_class",
        type=str,
        default="Bears",
        help="Class to evaluate unlearning for",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="ConcepACP",
        help="Model subfolder name (ConcepACP or ori)",
    )
    parser.add_argument(
        "--batch_size", type=int, default=32, help="Batch size for evaluation"
    )
    parser.add_argument(
        "--num_workers", type=int, default=4, help="Number of data loading workers"
    )

    args = parser.parse_args()

    main(
        args.input_dir,
        args.checkpoint_path,
        args.target_class,
        args.model_name,
        args.batch_size,
        args.num_workers,
    )
