# evaluation/anomalib/compare_cfa_stats.py: informational CFA memory-bank and descriptor checkpoint differences

import argparse
import json
import os

import torch

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def tensor_difference(left, right, relative=False):
    """Describe compatible tensor differences without an acceptance verdict."""
    same_shape = left is not None and right is not None and left.shape == right.shape
    reason = "missing_tensor" if left is None or right is None else "shape_mismatch" if not same_shape else None
    absolute = maximum_relative = None
    if reason is None:
        if not left.numel():
            reason = "empty_buffer"
        elif not torch.isfinite(left).all() or not torch.isfinite(right).all():
            reason = "nonfinite_tensor"
        else:
            difference = (left.double() - right.double()).abs()
            absolute = difference.max().item()
            if relative:
                maximum_relative = (difference / left.double().abs().clamp_min(torch.finfo(left.dtype).eps)).max().item()
    result = {"reference_shape": None if left is None else list(left.shape),
              "native_shape": None if right is None else list(right.shape), "shape_match": same_shape,
              "max_absolute_difference": absolute, "not_comparable_reason": reason}
    if relative:
        result["max_relative_difference"] = maximum_relative
    return result


def compare_stats(reference_path, native_path):
    """Compare the memory bank and every descriptor.layer tensor as information only."""
    states = [torch.load(path, map_location="cpu", weights_only=False)["state_dict"]
              for path in (reference_path, native_path)]
    descriptor = {name for state in states for name in state if name.startswith("model.descriptor.layer.")}
    if not descriptor or any("model.memory_bank" not in state for state in states):
        raise ValueError("Checkpoints must contain a memory bank and descriptor.layer tensors")
    names = ["model.memory_bank"] + sorted(descriptor)
    buffers = {}
    for name in names:
        left, right = (state.get(name) for state in states)
        if any(value is not None and not isinstance(value, torch.Tensor) for value in (left, right)):
            raise ValueError(f"Expected checkpoint tensors: {name}")
        buffers[name.removeprefix("model.")] = tensor_difference(left, right, relative=name == "model.memory_bank")
    return {"informational_only": True, "reference": os.path.abspath(reference_path), "native": os.path.abspath(native_path),
            "relative_difference_denominator": "abs(reference).clamp_min(dtype epsilon)", "buffers": buffers}


def main(args=None):
    """Print and save checkpoint differences without assigning pass or fail."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference", required=True)
    parser.add_argument("--native", required=True)
    parser.add_argument("--out", default=os.path.join(PROJECT_ROOT, "outputs", "compare", "cfa_stats.json"))
    args = parser.parse_args(args)
    report = compare_stats(args.reference, args.native)
    print("Informational comparison only; no acceptance verdict.")
    print("| Tensor | Reference shape | Native shape | Shape match | Max absolute | Max relative | Note |")
    print("|---|---|---|---|---|---|---|")
    for name, values in report["buffers"].items():
        print(f"| {name} | {values['reference_shape']} | {values['native_shape']} | {values['shape_match']} | "
              f"{values['max_absolute_difference']} | {values.get('max_relative_difference')} | "
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
