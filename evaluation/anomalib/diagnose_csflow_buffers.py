# evaluation/anomalib/diagnose_csflow_buffers.py: read-only CSFlow checkpoint BatchNorm buffer diagnostics

import argparse
import os

import torch

BACKBONE_PREFIX = "model.feature_extractor.feature_extractor.backbone."
FX_PREFIX = "model.feature_extractor.feature_extractor.feature_extractor."
BUFFER_KINDS = ("running_mean", "running_var", "num_batches_tracked")


def parse_checkpoint(value):
    """Parse a labeled checkpoint path without splitting additional equals signs in the path."""
    label, separator, path = value.partition("=")
    if not separator or not label.strip() or not path.strip():
        raise argparse.ArgumentTypeError("Expected --checkpoint LABEL=PATH with a nonempty label and path")
    return label.strip(), os.path.abspath(path)


def select_buffers(state, prefix):
    """Select BatchNorm buffers and remove only the specified module prefix."""
    if not isinstance(state, dict):
        raise ValueError("Expected a state_dict mapping")
    buffers = {name[len(prefix):]: value for name, value in state.items()
               if name.startswith(prefix) and name.rsplit(".", 1)[-1] in BUFFER_KINDS}
    for name, value in buffers.items():
        if not isinstance(value, torch.Tensor) or value.numel() == 0 or value.is_complex():
            raise ValueError(f"Expected a nonempty real tensor for buffer {prefix}{name}")
    return buffers


def equal_with_nan(left, right):
    """Compare duplicate tensors exactly while treating NaNs at matching positions as equal."""
    return (left.shape == right.shape and left.dtype == right.dtype and
            bool(((left == right) | (torch.isnan(left) & torch.isnan(right))).all()))


def statistics(values):
    """Summarize finite elements and report how many nonfinite elements were excluded."""
    values = values.detach().reshape(-1).to(dtype=torch.float64)
    finite = values[torch.isfinite(values)]
    result = {"finite": finite.numel(), "excluded": values.numel() - finite.numel(),
              "median": float("nan"), "p90": float("nan"), "min": float("nan"), "max": float("nan")}
    if finite.numel():
        result.update(median=torch.quantile(finite, 0.5).item(), p90=torch.quantile(finite, 0.9).item(),
                      min=finite.min().item(), max=finite.max().item())
    return result


def print_statistics(name, stats, fields):
    """Print one finite-only distribution with explicit sample counts."""
    summary = " ".join(f"{field}={stats[field]:.8g}" for field in fields)
    print(f"{name}: {summary} finite_elements={stats['finite']} excluded_nonfinite={stats['excluded']}")


def diagnose(label, path, weights):
    """Compare one memory-mapped checkpoint with local buffers without changing either state."""
    checkpoint = torch.load(path, map_location="cpu", weights_only=False, mmap=True)
    if not isinstance(checkpoint, dict) or "state_dict" not in checkpoint:
        raise ValueError(f"{path}: expected a Lightning checkpoint with state_dict")
    state = checkpoint["state_dict"]
    buffers = select_buffers(state, BACKBONE_PREFIX)
    if not buffers:
        raise ValueError(f"{path}: no BatchNorm buffers under {BACKBONE_PREFIX}")
    if buffers.keys() != weights.keys():
        raise ValueError(f"{label}: buffer keys differ: missing={sorted(weights.keys() - buffers.keys())}, "
                         f"unexpected={sorted(buffers.keys() - weights.keys())}")
    for name, value in buffers.items():
        if value.shape != weights[name].shape:
            raise ValueError(f"{label}: buffer shape mismatch for {name}: {value.shape} vs {weights[name].shape}")
    fx_buffers = select_buffers(state, FX_PREFIX)
    unexpected_fx = sorted(fx_buffers.keys() - buffers.keys())
    if unexpected_fx:
        raise ValueError(f"{label}: FX buffers without backbone counterparts: {unexpected_fx}")
    different_fx = sorted(name for name, value in fx_buffers.items() if not equal_with_nan(value, buffers[name]))
    print(f"\nCheckpoint: {label} path={path}")
    print(f"FX duplicate buffers: present={len(fx_buffers)} equal={len(fx_buffers) - len(different_fx)} "
          f"different={len(different_fx)} backbone_only={len(buffers) - len(fx_buffers)} (matching NaNs count as equal)")
    for name in different_fx:
        print(f"FX mismatch: {name}")
    print("\nBackbone buffers compared with the weights file:")
    print("kind | total_buffers | different_buffers | ckpt_nan_elements | ckpt_inf_elements | "
          "file_nan_elements | file_inf_elements | ckpt_buffers_abs_gt_1e6 | file_buffers_abs_gt_1e6")
    relatives = {kind: [] for kind in ("running_mean", "running_var")}
    ratios, deltas, ranked = [], [], []
    for kind in BUFFER_KINDS:
        names = sorted(name for name in buffers if name.rsplit(".", 1)[-1] == kind)
        different = sum(not torch.equal(buffers[name], weights[name]) for name in names)
        checkpoint_values = torch.cat([buffers[name].reshape(-1).to(dtype=torch.float64) for name in names])
        file_values = torch.cat([weights[name].reshape(-1).to(dtype=torch.float64) for name in names])
        checkpoint_large = sum(bool((buffers[name].abs() > 1e6).any()) for name in names)
        file_large = sum(bool((weights[name].abs() > 1e6).any()) for name in names)
        print(f"{kind} | {len(names)} | {different} | {torch.isnan(checkpoint_values).sum().item()} | "
              f"{torch.isinf(checkpoint_values).sum().item()} | {torch.isnan(file_values).sum().item()} | "
              f"{torch.isinf(file_values).sum().item()} | {checkpoint_large} | {file_large}")
        if kind == "num_batches_tracked":
            for name in names:
                if buffers[name].numel() != 1 or buffers[name].dtype != torch.int64 or weights[name].dtype != torch.int64:
                    raise ValueError(f"{label}: expected scalar int64 counter for {name}")
                deltas.append(int(buffers[name].item()) - int(weights[name].item()))
            continue
        for name in names:
            actual = buffers[name].reshape(-1).to(dtype=torch.float64)
            expected = weights[name].reshape(-1).to(dtype=torch.float64)
            relative = (actual - expected).abs() / (expected.abs() + 1e-12)
            relatives[kind].append(relative)
            stats = statistics(relative)
            ranked.append((float("inf") if stats["excluded"] else stats["max"], name, stats))
            if kind == "running_var":
                ratios.append(actual / expected)
    print(f"Total backbone buffers: {len(buffers)}; different={sum(not torch.equal(buffers[name], weights[name]) for name in buffers)}")
    print("\nCounter differences (checkpoint minus weights, counted per buffer):")
    for delta in sorted(set(deltas)):
        print(f"delta={delta:+d} buffers={deltas.count(delta)}")
    print("\nElementwise distributions use finite values only; ratio uses direct division without an epsilon:")
    for kind, values in relatives.items():
        print_statistics(f"{kind} abs_relative_difference", statistics(torch.cat(values)), ("median", "p90", "max"))
    ratio_stats = statistics(torch.cat(ratios))
    print_statistics("running_var checkpoint/file ratio", ratio_stats, ("median", "min", "max"))
    print("\nTop 5 buffers by maximum relative difference; nonfinite differences rank first:")
    print("buffer | finite_max_relative_difference | nonfinite_relative_elements")
    for _, name, stats in sorted(ranked, key=lambda row: (-row[0], row[1]))[:5]:
        print(f"{name} | {stats['max']:.8g} | {stats['excluded']}")
    return label, ratio_stats


def self_check():
    """Check arithmetic, duplicate NaN equality and labeled path parsing using tiny in-memory tensors."""
    stats = statistics(torch.tensor([0.0, 1.0, 2.0, float("nan"), float("inf")]))
    assert stats["finite"] == 3 and stats["excluded"] == 2
    assert stats["median"] == 1 and abs(stats["p90"] - 1.8) < 1e-12
    assert stats["min"] == 0 and stats["max"] == 2
    assert equal_with_nan(torch.tensor([float("nan"), 1.0]), torch.tensor([float("nan"), 1.0]))
    assert not equal_with_nan(torch.tensor([1.0]), torch.tensor([2.0]))
    assert parse_checkpoint("example=file=name.ckpt") == ("example", os.path.abspath("file=name.ckpt"))


def main():
    """Print read-only buffer diagnostics and one factual cross-checkpoint summary."""
    parser = argparse.ArgumentParser(description="Compare CSFlow checkpoint BatchNorm buffers with local torchvision weights.")
    parser.add_argument("--checkpoint", action="append", type=parse_checkpoint, required=True, metavar="LABEL=PATH")
    parser.add_argument("--weights", default="D:/backbones/efficientnet_b5_lukemelas-1a07897c.pth")
    args = parser.parse_args()
    labels = [label for label, _ in args.checkpoint]
    if len(labels) != len(set(labels)):
        parser.error("Checkpoint labels must be unique")
    self_check()
    weights_path = os.path.abspath(args.weights)
    weights = select_buffers(torch.load(weights_path, map_location="cpu", weights_only=True), "")
    if any(not any(name.rsplit(".", 1)[-1] == kind for name in weights) for kind in BUFFER_KINDS):
        raise ValueError("Weights must contain all three BatchNorm buffer kinds")
    print(f"Weights: {weights_path}")
    print(f"Backbone prefix: {BACKBONE_PREFIX}")
    print(f"FX prefix: {FX_PREFIX}")
    summaries = [diagnose(label, path, weights) for label, path in args.checkpoint]
    distances = [(label, abs(stats["median"] - 1)) for label, stats in summaries if stats["finite"]]
    largest = max((distance for _, distance in distances), default=None)
    farthest = ",".join(label for label, distance in distances if distance == largest) if distances else "unavailable"
    values = "; ".join(f"{label}: finite_ratio_median={stats['median']:.8g}, "
                       f"abs(median-1)={abs(stats['median'] - 1):.8g}, excluded_nonfinite={stats['excluded']}"
                       for label, stats in summaries)
    print(f"\nCheckpoint comparison: {values}; farthest_finite_median_from_1={farthest}")


if __name__ == "__main__":
    main()
