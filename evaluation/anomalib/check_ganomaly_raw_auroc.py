# evaluation/anomalib/check_ganomaly_raw_auroc.py: read-only raw GANomaly checkpoint AUROC and score summaries

import argparse
import gc
import os
import sys
from itertools import combinations

sys.dont_write_bytecode = True
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "src"))

import torch

from visionlab.anomaly_detection.common.metric import AUROC
from visionlab.anomaly_detection.common.pre_processor import IMAGENET_MEAN, IMAGENET_STD, PreProcessor
from visionlab.anomaly_detection.data.mvtec.lightning_data import MVTecLitData
from visionlab.anomaly_detection.data.visa.lightning_data import VisALitData
from visionlab.anomaly_detection.models.ganomaly.model import GanomalyModel


def parse_run(value):
    """Parse a labeled dataset and checkpoint while preserving equals signs inside the path."""
    label, separator, remainder = value.partition("=")
    data, path_separator, path = remainder.partition("=")
    label, data, path = label.strip(), data.strip(), path.strip()
    if not separator or not path_separator or not label or data not in ("mvtec", "visa") or not path:
        raise argparse.ArgumentTypeError("Expected --run LABEL=DATA=PATH with DATA mvtec or visa")
    return label, data, os.path.abspath(path)


def make_data(data):
    """Use the native data module with the current GANomaly dataset roots, split and evaluation batch size."""
    data_class, root, category = {
        "mvtec": (MVTecLitData, "D:/datasets/mvtec", "bottle"),
        "visa": (VisALitData, "D:/datasets/visa_anomalib", "candle"),
    }[data]
    module = data_class(root=root, category=category, train_batch_size=32, eval_batch_size=32,
                        num_workers=0, test_split_mode="from_dir", test_split_ratio=0.2,
                        val_split_mode="from_test", val_split_ratio=0.4, seed=42)
    module.setup(stage="test")
    return module


def load_model(path):
    """Restore only model-prefixed tensors from one memory-mapped Lightning checkpoint on CPU."""
    checkpoint = torch.load(path, map_location="cpu", weights_only=False, mmap=True)
    if not isinstance(checkpoint, dict) or not isinstance(checkpoint.get("state_dict"), dict):
        raise ValueError(f"{path}: expected a Lightning checkpoint with state_dict")
    state = {name.removeprefix("model."): value for name, value in checkpoint["state_dict"].items()
             if name.startswith("model.")}
    if not state:
        raise ValueError(f"{path}: no model.* state")
    model = GanomalyModel(input_size=(256, 256), num_input_channels=3, n_features=64,
                          latent_vec_size=100, extra_layers=0, add_final_conv_layer=True)
    model.load_state_dict(state, strict=True)
    del state, checkpoint
    return model.eval()


def score_summary(scores):
    """Return count, mean, minimum and maximum without changing raw score ordering."""
    if not scores.numel():
        return 0, float("nan"), float("nan"), float("nan")
    values = scores.to(dtype=torch.float64)
    return values.numel(), values.mean().item(), values.min().item(), values.max().item()


def evaluate(label, data, path):
    """Evaluate one checkpoint on the native test split and print raw image scores without writing files."""
    module = make_data(data)
    loader = module.test_dataloader()
    pre_processor = PreProcessor(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    metric = AUROC()
    model = load_model(path)
    scores, labels = [], []
    try:
        with torch.inference_mode():
            for images, target in loader:
                prediction = model(pre_processor(images)).pred_score
                metric.update(prediction, target["label"])
                scores.append(prediction.detach().cpu())
                labels.append(target["label"].detach().cpu())
        if not scores:
            raise ValueError(f"{label}: empty test loader")
        scores, labels = torch.cat(scores), torch.cat(labels)
        value = metric.compute()
        print(f"\nLABEL={label} DATA={data}")
        print(f"Checkpoint: {path}")
        normal, anomalous = score_summary(scores[labels == 0]), score_summary(scores[labels == 1])
        print(f"Test images: total={labels.numel()} normal={normal[0]} anomalous={anomalous[0]}")
        print(f"Raw Image AUROC: {value:.8f}")
        if metric.reason is not None:
            print(f"AUROC reason: {metric.reason}")
        print("Label | Count | Mean | Min | Max")
        for name, summary in (("normal", normal), ("anomalous", anomalous)):
            count, mean, minimum, maximum = summary
            print(f"{name} | {count} | {mean:.8e} | {minimum:.8e} | {maximum:.8e}")
        return label, data, value
    finally:
        del model
        gc.collect()


def self_check():
    """Check labeled path parsing and score reductions using only tiny in-memory values."""
    assert parse_run("example=mvtec=file=name.ckpt") == ("example", "mvtec", os.path.abspath("file=name.ckpt"))
    for value in ("missing", "label=unknown=file.ckpt", "=mvtec=file.ckpt", "label=visa="):
        try:
            parse_run(value)
        except argparse.ArgumentTypeError:
            pass
        else:
            raise AssertionError(f"Accepted invalid run: {value}")
    assert score_summary(torch.tensor([1.0, 3.0])) == (2, 2.0, 1.0, 3.0)
    assert score_summary(torch.empty(0))[0] == 0


def main():
    """Print sequential checkpoint evaluations and factual within-dataset AUROC differences."""
    parser = argparse.ArgumentParser(description="Read GANomaly checkpoints and evaluate raw rank-preserving Image AUROC.")
    parser.add_argument("--run", action="append", type=parse_run, required=True, metavar="LABEL=DATA=PATH")
    args = parser.parse_args()
    labels = [label for label, _, _ in args.run]
    if len(labels) != len(set(labels)):
        parser.error("Run labels must be unique")
    self_check()
    summaries = [evaluate(label, data, path) for label, data, path in args.run]
    for data in ("mvtec", "visa"):
        runs = [(label, value) for label, dataset, value in summaries if dataset == data]
        differences = [f"abs({left_label} - {right_label})={abs(left - right):.8f}"
                       for (left_label, left), (right_label, right) in combinations(runs, 2)]
        if differences:
            print(f"DATA={data} raw Image AUROC differences: " + "; ".join(differences))


if __name__ == "__main__":
    main()
