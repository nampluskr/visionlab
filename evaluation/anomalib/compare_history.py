# evaluation/anomalib/compare_history.py: inspect every scalar event and compare epoch training losses

import argparse
import os

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
from tensorboard.util.tensor_util import make_ndarray


def read_history(run, tag="train_loss_epoch"):
    """Print all scalar records per event file and collect epoch losses without a verdict."""
    root = os.path.join(run, "lightning_logs")
    paths = sorted(os.path.join(directory, name) for directory, _, names in os.walk(root)
                   for name in names if name.startswith("events.out.tfevents."))
    if not paths:
        raise FileNotFoundError(f"No TensorBoard event files in {root}")
    losses = {}
    fallback_epoch = 0
    for path in paths:
        events = EventAccumulator(path, size_guidance={"scalars": 0, "tensors": 0})
        events.Reload()
        records = {}
        for name in events.Tags().get("scalars", []):
            records[name] = [(event.step, event.value, event.wall_time) for event in events.Scalars(name)]
        for name in events.Tags().get("tensors", []):
            for event in events.Tensors(name):
                value = make_ndarray(event.tensor_proto)
                if value.ndim == 0 and value.dtype.kind in "biuf":
                    records.setdefault(name, []).append((event.step, float(value), event.wall_time))
        print(f"Event file: {path}")
        for name, values in sorted(records.items()):
            print(f"  {name}: {[(step, value) for step, value, _ in values]}")
        for step, value, wall_time in sorted(records.get(tag, []), key=lambda item: item[2]):
            candidates = [event for event in records.get("epoch", []) if event[0] == step]
            epoch = int(min(candidates, key=lambda item: abs(item[2] - wall_time))[1]) if candidates else fallback_epoch
            fallback_epoch += 1
            if epoch not in losses or wall_time >= losses[epoch][1]:
                losses[epoch] = (value, wall_time)
    return {epoch: value for epoch, (value, _) in losses.items()}


def print_comparison(reference, native):
    """Print one epoch comparison table without assigning tolerances."""
    print("| Epoch | Reference | Native | Absolute difference | Relative difference |")
    print("|---:|---:|---:|---:|---:|")
    for epoch in sorted(reference.keys() | native.keys()):
        left, right = reference.get(epoch), native.get(epoch)
        if left is None or right is None:
            print(f"| {epoch} | {left if left is not None else '-'} | {right if right is not None else '-'} | - | - |")
            continue
        difference = abs(left - right)
        relative = difference / abs(left) if left != 0 else (0.0 if difference == 0 else float("inf"))
        print(f"| {epoch} | {left:.10g} | {right:.10g} | {difference:.10g} | {relative:.10g} |")


def main():
    """Display each requested epoch tag for human interpretation."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference", required=True)
    parser.add_argument("--native", required=True)
    parser.add_argument("--tags", nargs="+", default=["train_loss_epoch"])
    args = parser.parse_args()
    for tag in args.tags:
        reference = read_history(os.path.abspath(args.reference), tag=tag)
        native = read_history(os.path.abspath(args.native), tag=tag)
        print(f"Tag: {tag}")
        print_comparison(reference, native)


if __name__ == "__main__":
    main()
