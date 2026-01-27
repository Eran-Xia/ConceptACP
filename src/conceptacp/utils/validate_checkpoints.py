"""
Validation script to verify that block-level fine-tuning produces identical results to the original approach.

This script:
1. Loads checkpoints from both approaches
2. Compares the state dicts
3. Optionally runs inference to compare outputs
"""
import torch
import argparse
from pathlib import Path
import json


def load_checkpoint(path):
    """Load a checkpoint and return its state dict."""
    checkpoint = torch.load(path, map_location='cpu', weights_only=False)
    if 'model_state_dict' in checkpoint:
        return checkpoint['model_state_dict'], checkpoint
    return checkpoint, {}


def compare_state_dicts(state_dict1, state_dict2, tolerance=1e-5):
    """Compare two state dicts and return differences."""
    differences = []

    # Check if keys match
    keys1 = set(state_dict1.keys())
    keys2 = set(state_dict2.keys())

    if keys1 != keys2:
        differences.append(f"Key mismatch:")
        differences.append(f"  Only in dict1: {keys1 - keys2}")
        differences.append(f"  Only in dict2: {keys2 - keys1}")
        return differences

    # Compare values
    max_diff = 0.0
    max_diff_key = None

    for key in keys1:
        tensor1 = state_dict1[key]
        tensor2 = state_dict2[key]

        if tensor1.shape != tensor2.shape:
            differences.append(f"Shape mismatch for {key}: {tensor1.shape} vs {tensor2.shape}")
            continue

        diff = torch.abs(tensor1 - tensor2).max().item()
        if diff > max_diff:
            max_diff = diff
            max_diff_key = key

        if diff > tolerance:
            differences.append(f"{key}: max diff = {diff:.2e} (exceeds tolerance {tolerance:.2e})")

    if not differences:
        return [f"✓ All parameters match within tolerance {tolerance:.2e}", f"  Maximum difference: {max_diff:.2e} (at {max_diff_key})"]

    return differences


def main():
    parser = argparse.ArgumentParser(description="Compare checkpoints from different fine-tuning approaches")
    parser.add_argument("--original", type=str, help="Path to original checkpoint")
    parser.add_argument("--block", type=str, help="Path to block-level checkpoint")
    parser.add_argument("--tolerance", type=float, default=1e-5, help="Tolerance for parameter comparison")
    parser.add_argument("--list-models", action="store_true", help="List available model checkpoints")

    args = parser.parse_args()

    models_dir = Path("models")

    if args.list_models:
        print("\n=== Available Model Checkpoints ===\n")
        if not models_dir.exists():
            print(f"Models directory not found: {models_dir}")
            return

        original_ckpts = sorted(models_dir.glob("*.pt"))
        original_ckpts = [p for p in original_ckpts if not p.name.endswith("_block.pt")]
        block_ckpts = sorted(models_dir.glob("*_block.pt"))

        print("Original approach checkpoints:")
        for p in original_ckpts:
            print(f"  {p}")

        print("\nBlock-level approach checkpoints:")
        for p in block_ckpts:
            print(f"  {p}")

        print(f"\nUsage example:")
        if original_ckpts and block_ckpts:
            print(f"  python {__file__} --original {original_ckpts[0]} --block {block_ckpts[0]}")
        return

    if not args.original or not args.block:
        print("Error: Please specify both --original and --block checkpoint paths")
        print("Use --list-models to see available checkpoints")
        return

    print("\n=== Checkpoint Comparison ===\n")
    print(f"Original:   {args.original}")
    print(f"Block-level: {args.block}")
    print(f"Tolerance:   {args.tolerance:.2e}\n")

    # Load checkpoints
    print("Loading checkpoints...")
    state_dict1, meta1 = load_checkpoint(args.original)
    state_dict2, meta2 = load_checkpoint(args.block)

    # Print metadata
    print("\n--- Original Checkpoint Metadata ---")
    for k, v in meta1.items():
        if k != 'model_state_dict':
            print(f"  {k}: {v}")

    print("\n--- Block-level Checkpoint Metadata ---")
    for k, v in meta2.items():
        if k != 'model_state_dict':
            print(f"  {k}: {v}")

    # Compare state dicts
    print("\n--- Comparing Parameters ---")
    differences = compare_state_dicts(state_dict1, state_dict2, tolerance=args.tolerance)

    for diff in differences:
        print(diff)

    if not any("mismatch" in d.lower() or "exceeds" in d.lower() for d in differences):
        print("\n✓ SUCCESS: Checkpoints are equivalent!")
        print("  Block-level fine-tuning produces identical results to the original approach.")
    else:
        print("\n✗ WARNING: Checkpoints differ!")
        print("  This may indicate:")
        print("  - Different training hyperparameters")
        print("  - Different number of epochs")
        print("  - Different random seeds")
        print("  - Implementation differences")


if __name__ == "__main__":
    main()
