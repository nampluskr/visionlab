# evaluation/anomalib/compare_dfkde_stats.py: informational comparison of DFKDE PCA and KDE checkpoint statistics

import argparse
import json
import math
import os

import torch

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def tensor_difference(left, right):
    """Describe compatible scalar or mean buffers without assigning an acceptance verdict."""
    same_shape = left.shape == right.shape
    reason = "shape_mismatch" if not same_shape else "empty_buffer" if not left.numel() else None
    absolute = relative = None
    if reason is None:
        if not torch.isfinite(left).all() or not torch.isfinite(right).all():
            reason = "nonfinite_buffer"
        else:
            difference = (left.double() - right.double()).abs()
            absolute = difference.max().item()
            relative = (difference / left.double().abs().clamp_min(torch.finfo(left.dtype).eps)).max().item()
    return {"reference_shape": list(left.shape), "native_shape": list(right.shape), "shape_match": same_shape,
            "max_absolute_difference": absolute, "max_relative_difference": relative, "not_comparable_reason": reason}


def compare_stats(reference_path, native_path):
    """Compare sign-invariant statistics and restrict sign-dependent KDE buffers to their shapes."""
    reference = torch.load(reference_path, map_location="cpu", weights_only=False)["state_dict"]
    native = torch.load(native_path, map_location="cpu", weights_only=False)["state_dict"]
    prefix = "model.classifier."
    names = ("pca_model.mean", "pca_model.num_components", "pca_model.singular_vectors", "max_length",
             "kde_model.norm", "kde_model.bw_transform", "kde_model.dataset")
    for state in (reference, native):
        if any(not isinstance(state.get(prefix + name), torch.Tensor) for name in names):
            raise ValueError("Checkpoint must contain all PCA, KDE and scaling tensor buffers")
        if state[prefix + "pca_model.num_components"].numel() != 1:
            raise ValueError("PCA num_components must contain one value")
    left_k = int(reference[prefix + "pca_model.num_components"].item())
    right_k = int(native[prefix + "pca_model.num_components"].item())
    buffers = {name: tensor_difference(reference[prefix + name], native[prefix + name])
               for name in ("pca_model.mean", "max_length", "kde_model.norm")}
    left, right = reference[prefix + "pca_model.singular_vectors"], native[prefix + "pca_model.singular_vectors"]
    same_shape = left.shape == right.shape
    reason = "component_count_mismatch" if left_k != right_k else "shape_mismatch" if not same_shape else None
    overlap = distance = None
    if reason is None:
        if left.ndim != 2 or left.shape[1] != left_k:
            reason = "invalid_component_shape"
        elif not torch.isfinite(left).all() or not torch.isfinite(right).all():
            reason = "nonfinite_buffer"
        else:
            overlap = (left.double().T @ right.double()).square().sum().item()
            distance = math.sqrt(max(left_k - overlap, 0.0))
    buffers["pca_model.singular_vectors"] = {
        "reference_shape": list(left.shape), "native_shape": list(right.shape), "shape_match": same_shape,
        "cross_gram_squared_frobenius": overlap, "subspace_distance": distance, "not_comparable_reason": reason}
    for name in ("kde_model.bw_transform", "kde_model.dataset"):
        left, right = reference[prefix + name], native[prefix + name]
        buffers[name] = {"reference_shape": list(left.shape), "native_shape": list(right.shape),
                         "shape_match": left.shape == right.shape, "comparison": "shape_only_sign_dependent"}
    return {"informational_only": True, "reference": os.path.abspath(reference_path), "native": os.path.abspath(native_path),
            "num_components": {"reference": left_k, "native": right_k, "equal": left_k == right_k},
            "relative_difference_denominator": "abs(reference).clamp_min(dtype epsilon)", "buffers": buffers}


def main(args=None):
    """Print and save checkpoint statistics as information rather than a pass or fail result."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference", required=True)
    parser.add_argument("--native", required=True)
    parser.add_argument("--out", default=os.path.join(PROJECT_ROOT, "outputs", "compare", "dfkde_stats.json"))
    args = parser.parse_args(args)
    report = compare_stats(args.reference, args.native)
    print("Informational comparison only; no acceptance verdict.")
    counts = report["num_components"]
    print(f"Components: reference={counts['reference']}, native={counts['native']}, equal={counts['equal']}")
    print("| Buffer | Reference shape | Native shape | Shape match | Max absolute | Max relative | Subspace distance | Note |")
    print("|---|---|---|---|---|---|---|---|")
    for name, values in report["buffers"].items():
        note = values.get("not_comparable_reason") or values.get("comparison", "")
        print(f"| {name} | {values['reference_shape']} | {values['native_shape']} | {values['shape_match']} | "
              f"{values.get('max_absolute_difference')} | {values.get('max_relative_difference')} | "
              f"{values.get('subspace_distance')} | {note} |")
    path = os.path.abspath(args.out)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(f"Information saved: {path}")
    return report


if __name__ == "__main__":
    main()
