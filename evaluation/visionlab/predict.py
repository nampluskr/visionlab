# evaluation/visionlab/predict.py: offline pure PyTorch evaluation from a native model export

import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "src"))

from visionlab.utils.io import block_network

block_network()

import argparse
import json
import math
import re

import yaml
from torch.utils.data import DataLoader

from visionlab.anomaly_detection.common.metric import AUROC, F1
from visionlab.anomaly_detection.data.common.split import split_datasets
from visionlab.anomaly_detection.data.mvtec.dataset import MVTecDataset
from visionlab.anomaly_detection.data.visa.dataset import VisADataset
from visionlab.inference.predictor import Predictor
from visionlab.utils.io import file_sha256


def latest_export(model, data, category):
    """Select the export in the highest numbered run without falling back to older runs."""
    root = os.path.join(PROJECT_ROOT, "outputs", "visionlab", model, f"{data}_{category}")
    if not os.path.isdir(root):
        raise FileNotFoundError(root)
    versions = [(int(match.group(1)), name) for name in os.listdir(root)
                if os.path.isdir(os.path.join(root, name)) and (match := re.fullmatch(r"v(\d+)", name))]
    if not versions:
        raise FileNotFoundError(f"No numbered runs in {root}")
    path = os.path.join(root, max(versions)[1], "export", "model.pt")
    if not os.path.isfile(path):
        raise FileNotFoundError(path)
    return path


def main(args=None):
    """Evaluate the configured pure test split and save explicit metric failure reasons."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=("stfpm", "efficient_ad", "padim"), required=True)
    parser.add_argument("--data", choices=("mvtec", "visa"), required=True)
    parser.add_argument("--export")
    parser.add_argument("--category")
    args = parser.parse_args(args)
    config_path = os.path.join(PROJECT_ROOT, "src", "visionlab", "anomaly_detection", "configs", "data", f"{args.data}.yaml")
    with open(config_path, encoding="utf-8") as stream:
        config = yaml.safe_load(stream)["data"]["init_args"]
    model_config_path = os.path.join(PROJECT_ROOT, "src", "visionlab", "anomaly_detection", "configs", "models",
                                     f"{args.model}.yaml")
    with open(model_config_path, encoding="utf-8") as stream:
        config.update(yaml.safe_load(stream).get("data", {}).get("init_args", {}))
    category = args.category or config["category"]
    if os.path.basename(category) != category or category in (".", ".."):
        parser.error("--category must be a single directory name")
    config["category"] = category
    path = os.path.abspath(args.export) if args.export else latest_export(args.model, args.data, category)
    dataset_class = MVTecDataset if args.data == "mvtec" else VisADataset
    train_data = dataset_class(root=config["root"], category=category, split="train")
    test_data = dataset_class(root=config["root"], category=category, split="test")
    _, _, test_data = split_datasets(
        train_data=train_data, test_data=test_data,
        test_split_mode=config.get("test_split_mode", "from_dir"),
        test_split_ratio=config.get("test_split_ratio", 0.2),
        val_split_mode=config["val_split_mode"], val_split_ratio=config["val_split_ratio"], seed=config["seed"],
    )
    loader = DataLoader(test_data, batch_size=config["eval_batch_size"], shuffle=False, num_workers=0)
    predictor = Predictor.from_export(path, device="cpu")
    metrics = [("image_AUROC", AUROC(), "pred_score", "label"),
               ("pixel_AUROC", AUROC(), "anomaly_map", "mask"),
               ("image_F1Score", F1(), "pred_label", "label"),
               ("pixel_F1Score", F1(), "pred_mask", "mask")]
    for images, target in loader:
        outputs = predictor.predict(images)
        for _, metric, prediction, truth in metrics:
            if outputs.get(prediction) is not None and target.get(truth) is not None:
                metric.update(outputs[prediction], target[truth])
    values = {}
    reasons = {}
    print("| Metric | Value | Reason |")
    print("|---|---:|---|")
    for name, metric, prediction, _ in metrics:
        value = metric.compute()
        reason = metric.reason
        if prediction in ("pred_label", "pred_mask") and predictor.post_processor is None:
            reason = "post_processor_disabled"
        values[name] = None if math.isnan(value) else value
        reasons[name] = reason
        formatted = "NaN" if math.isnan(value) else f"{value:.8f}"
        print(f"| {name} | {formatted} | {reason or ''} |")
    report = {"model": args.model, "data": args.data, "category": category, "data_root": config["root"],
              "config_file": config_path, "data_config": config,
              "export": {"path": path, "sha256": file_sha256(path)},
              "test_image_paths": [sample["image_path"] for sample in test_data.samples],
              "metrics": values, "reasons": reasons}
    output_dir = os.path.join(os.path.dirname(os.path.dirname(path)), "predict")
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "metrics.json")
    with open(output_path, "w", encoding="utf-8", newline="\n") as stream:
        json.dump(report, stream, indent=2, ensure_ascii=True, allow_nan=False)
        stream.write("\n")
    print(f"Metrics: {output_path}")
    return report


if __name__ == "__main__":
    main()
