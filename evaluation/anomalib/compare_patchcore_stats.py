# evaluation/anomalib/compare_patchcore_stats.py: informational memory-bank comparison without pairwise distances

import argparse
from collections import Counter
import hashlib
import json
import os

import torch

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def row_hashes(bank):
    """Count exact row byte hashes while retaining only one row-sized byte copy at a time."""
    return Counter(hashlib.sha256(row.contiguous().view(torch.uint8).numpy().tobytes()).digest() for row in bank)


def compare_stats(reference_path, native_path):
    """Record bank shapes, order-independent exact row overlap and channel mean differences."""
    banks = []
    for path in (reference_path, native_path):
        bank = torch.load(path, map_location="cpu", weights_only=False)["state_dict"].get("model.memory_bank")
        if not isinstance(bank, torch.Tensor) or bank.ndim != 2 or not bank.is_floating_point():
            raise ValueError("Checkpoint must contain a two-dimensional floating memory_bank")
        banks.append(bank.detach())
    left, right = banks
    comparable = left.shape[1] == right.shape[1] and left.dtype == right.dtype
    matched = sum((row_hashes(left) & row_hashes(right)).values()) if comparable else None
    reason = "channel_mismatch" if left.shape[1] != right.shape[1] else "empty_buffer" if not left.numel() or not right.numel() else None
    absolute = relative = None
    if reason is None:
        left_mean, right_mean = left.mean(dim=0, dtype=torch.float64), right.mean(dim=0, dtype=torch.float64)
        if not torch.isfinite(left_mean).all() or not torch.isfinite(right_mean).all():
            reason = "nonfinite_mean"
        else:
            difference = (left_mean - right_mean).abs()
            absolute = difference.max().item()
            relative = (difference / left_mean.abs().clamp_min(torch.finfo(left.dtype).eps)).max().item()
    return {"informational_only": True, "reference": os.path.abspath(reference_path), "native": os.path.abspath(native_path),
            "reference_shape": list(left.shape), "native_shape": list(right.shape), "shape_match": left.shape == right.shape,
            "exact_rows": {"matched_rows": matched, "reference_ratio": matched / len(left) if comparable and len(left) else None,
                           "native_ratio": matched / len(right) if comparable and len(right) else None,
                           "comparison": "sha256_row_bytes_multiset", "comparable": comparable},
            "channel_mean": {"max_absolute_difference": absolute, "max_relative_difference": relative,
                             "relative_difference_denominator": "abs(reference mean).clamp_min(dtype epsilon)",
                             "not_comparable_reason": reason}}


def main(args=None):
    """Print and save memory-bank information without an acceptance verdict."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference", required=True)
    parser.add_argument("--native", required=True)
    parser.add_argument("--out", default=os.path.join(PROJECT_ROOT, "outputs", "compare", "patchcore_stats.json"))
    args = parser.parse_args(args)
    report = compare_stats(args.reference, args.native)
    rows, means = report["exact_rows"], report["channel_mean"]
    print("Informational comparison only; no acceptance verdict.")
    print("| Reference shape | Native shape | Shape match | Exact reference rows | Exact native rows | Mean max absolute | Mean max relative |")
    print("|---|---|---|---|---|---|---|")
    print(f"| {report['reference_shape']} | {report['native_shape']} | {report['shape_match']} | "
          f"{rows['reference_ratio']} | {rows['native_ratio']} | {means['max_absolute_difference']} | {means['max_relative_difference']} |")
    path = os.path.abspath(args.out)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(f"Information saved: {path}")
    return report


if __name__ == "__main__":
    main()
