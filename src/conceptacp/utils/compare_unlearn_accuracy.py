#!/usr/bin/env python
"""
Compare unlearning accuracy between original model and ConceptACP.

This script evaluates both 'ori' and 'ConcepACP' model outputs and compares:
1. Unlearning Accuracy (UA) - higher is better
2. In-class Retain Accuracy (IRA) - higher is better

Usage:
    uv run src/conceptacp/utils/compare_unlearn_accuracy.py \\
        --input_dir data/output/sd15_uc_object_bears \\
        --checkpoint_path checkpoints/style50_cls.pth \\
        --target_class Bears
"""

import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from evaluate_unlearn_accuracy import main as evaluate_main


def main(
    input_dir: str,
    checkpoint_path: str = "checkpoints/style50_cls.pth",
    target_class: str = "Bears",
    batch_size: int = 32,
    num_workers: int = 4,
):
    """
    Compare original and ConceptACP model outputs.
    """
    print(f"\n{'#'*70}")
    print(f"# Unlearning Accuracy Comparison")
    print(f"# Target Class: {target_class}")
    print(f"# Input Directory: {input_dir}")
    print(f"{'#'*70}\n")

    # Evaluate original model
    print(f"\n{'*'*70}")
    print(f"* EVALUATING ORIGINAL MODEL (ori)")
    print(f"{'*'*70}")
    ori_results = evaluate_main(
        input_dir=input_dir,
        checkpoint_path=checkpoint_path,
        target_class=target_class,
        model_name="ori",
        batch_size=batch_size,
        num_workers=num_workers,
    )

    # Evaluate ConceptACP model
    print(f"\n\n{'*'*70}")
    print(f"* EVALUATING CONCEPTACP MODEL (ConcepACP)")
    print(f"{'*'*70}")
    conceptacp_results = evaluate_main(
        input_dir=input_dir,
        checkpoint_path=checkpoint_path,
        target_class=target_class,
        model_name="ConcepACP",
        batch_size=batch_size,
        num_workers=num_workers,
    )

    # Comparison summary
    print(f"\n\n{'#'*70}")
    print(f"# COMPARISON SUMMARY")
    print(f"{'#'*70}\n")

    print(f"Target Class: {target_class}\n")

    print(f"{'Metric':<40} {'Original':>12} {'ConceptACP':>12} {'Delta':>12}")
    print(f"{'-'*80}")

    # Unlearning Accuracy (UA) - higher is better
    ori_ua = ori_results["ua"] * 100
    conceptacp_ua = conceptacp_results["ua"] * 100
    ua_delta = conceptacp_ua - ori_ua

    print(
        f"{'Unlearning Accuracy (UA)':<40} {ori_ua:>11.2f}% {conceptacp_ua:>11.2f}% {ua_delta:>+11.2f}%"
    )

    # In-class Retain Accuracy (IRA) - higher is better
    if ori_results["ira"] is not None and conceptacp_results["ira"] is not None:
        ori_ira = ori_results["ira"] * 100
        conceptacp_ira = conceptacp_results["ira"] * 100
        ira_delta = conceptacp_ira - ori_ira

        print(
            f"{'In-class Retain Accuracy (IRA)':<40} {ori_ira:>11.2f}% {conceptacp_ira:>11.2f}% {ira_delta:>+11.2f}%"
        )

    print(f"{'-'*80}\n")

    # Interpretation
    print("Interpretation:")
    print(f"  - UA measures how well the model has forgotten {target_class}")
    print(f"    (higher is better: images are NOT classified as {target_class})")
    print(
        f"  - IRA measures how well the model retains other classes (higher is better)"
    )
    print()

    if ua_delta > 0:
        print(
            f"✓ ConceptACP improved unlearning by {ua_delta:.2f}% (forgot {target_class} better)"
        )
    else:
        print(
            f"✗ ConceptACP decreased unlearning by {abs(ua_delta):.2f}% (still recognizes {target_class})"
        )

    if ori_results["ira"] is not None and conceptacp_results["ira"] is not None:
        if ira_delta >= -1.0:  # Allow small degradation
            print(
                f"✓ ConceptACP maintained retention performance ({ira_delta:+.2f}%)"
            )
        else:
            print(
                f"✗ ConceptACP degraded retention by {abs(ira_delta):.2f}% (other classes affected)"
            )

    print()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Compare unlearning accuracy between original and ConceptACP"
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
        args.batch_size,
        args.num_workers,
    )
