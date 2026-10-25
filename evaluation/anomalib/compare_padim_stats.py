# evaluation/anomalib/compare_padim_stats.py: informational comparison of PaDiM checkpoint indices and statistics

import argparse
import json
import os

import torch

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def compare_stats(reference_path, native_path):
    """Compare channel indices and compatible fitted buffers without assigning a verdict."""
    reference = torch.load(reference_path, map_location="cpu", weights_only=False)["state_dict"]
    native = torch.load(native_path, map_location="cpu", weights_only=False)["state_dict"]
    names = ("model.idx", "model.gaussian.mean", "model.gaussian.inv_covariance")
    for state in (reference, native):
        if any(name not in state or not isinstance(state[name], torch.Tensor) for name in names):
            raise ValueError("Checkpoint must contain PaDiM indices and Gaussian tensor buffers")
    reference_idx, native_idx = reference["model.idx"], native["model.idx"]
    same_idx = torch.equal(reference_idx, native_idx)
    overlap = torch.isin(reference_idx.unique(), native_idx.unique()).sum().item()
    report = {"informational_only": True, "reference": os.path.abspath(reference_path),
              "native": os.path.abspath(native_path),
              "idx": {"equal": same_idx, "overlapping_channels": overlap,
                      "reference_shape": list(reference_idx.shape), "native_shape": list(native_idx.shape)},
              "relative_difference_denominator": "abs(reference).clamp_min(dtype epsilon)", "buffers": {}}
    for name in names[1:]:
        left, right = reference[name], native[name]
        same_shape = left.shape == right.shape
        reason = "idx_mismatch" if not same_idx else "shape_mismatch" if not same_shape else None
        absolute = relative = None
        if reason is None:
            if not left.numel():
                reason = "empty_buffer"
            elif not torch.isfinite(left).all() or not torch.isfinite(right).all():
                reason = "nonfinite_buffer"
            else:
                difference = (left.double() - right.double()).abs()
                absolute = difference.max().item()
                relative = (difference / left.double().abs().clamp_min(torch.finfo(left.dtype).eps)).max().item()
        report["buffers"][name] = {"reference_shape": list(left.shape), "native_shape": list(right.shape),
                                   "shape_match": same_shape, "max_absolute_difference": absolute,
                                   "max_relative_difference": relative, "not_comparable_reason": reason}
    return report


def main(args=None):
    """Print and save checkpoint differences as information rather than a pass or fail result."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference", required=True)
    parser.add_argument("--native", required=True)
    parser.add_argument("--out", default=os.path.join(PROJECT_ROOT, "outputs", "compare", "padim_stats.json"))
    args = parser.parse_args(args)
    report = compare_stats(args.reference, args.native)
    print("Informational comparison only; no acceptance verdict.")
    print(f"Indices equal: {report['idx']['equal']}; overlapping channels: {report['idx']['overlapping_channels']}")
    print("| Buffer | Reference shape | Native shape | Shape match | Max absolute | Max relative | Reason |")
    print("|---|---|---|---|---|---|---|")
    for name, values in report["buffers"].items():
        print(f"| {name} | {values['reference_shape']} | {values['native_shape']} | {values['shape_match']} | "
              f"{values['max_absolute_difference']} | {values['max_relative_difference']} | "
              f"{values['not_comparable_reason'] or ''} |")
    path = os.path.abspath(args.out)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(f"Information saved: {path}")
    return report


if __name__ == "__main__":
    main()
