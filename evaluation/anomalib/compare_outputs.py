# evaluation/anomalib/compare_outputs.py: compare local model weights, forwards and processing with the reference

from local_setup import setup_local

setup_local()

import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "src"))

import argparse
import inspect
import json
import math
import random

import torch
import yaml
from anomalib.data import ImageBatch
from anomalib.metrics import AUROC as ReferenceAUROC
from anomalib.metrics import F1AdaptiveThreshold, F1Score
from anomalib.models import EfficientAd, Padim, Stfpm
from anomalib.models.image.padim.torch_model import PadimModel as ReferencePadimModel
from anomalib.post_processing import PostProcessor as ReferencePostProcessor

from visionlab.anomaly_detection.common.metric import AUROC, F1
from visionlab.anomaly_detection.common.post_processor import PostProcessor
from visionlab.anomaly_detection.common.pre_processor import IMAGENET_MEAN, IMAGENET_STD, PreProcessor
from visionlab.anomaly_detection.data.mvtec.dataset import MVTecDataset
from visionlab.anomaly_detection.data.visa.dataset import VisADataset
from visionlab.anomaly_detection.models.stfpm.lightning_model import StfpmLitModel
from visionlab.anomaly_detection.models.stfpm.model import STFPMModel
from visionlab.anomaly_detection.models.efficient_ad.lightning_model import EfficientAdLitModel
from visionlab.anomaly_detection.models.efficient_ad.model import EfficientAdModel
from visionlab.anomaly_detection.models.padim.lightning_model import PadimLitModel
from visionlab.anomaly_detection.models.padim.model import PadimModel
from visionlab.utils.io import file_sha256
from local_models import LocalEfficientAd

RTOL = 1e-4
ATOL = 1e-5
METRIC_ATOL = 1e-6
EFFICIENT_AD_RTOL = 1e-5
EFFICIENT_AD_ATOL = 1e-6
PADIM_RTOL = 1e-5
PADIM_ATOL = 1e-6


def signature_fields(model_class):
    """Compare constructor names, order, kinds and default representations without annotations."""
    return [{"name": name, "kind": parameter.kind.name, "default": repr(parameter.default)}
            for name, parameter in inspect.signature(model_class.__init__).parameters.items()]


def record_check(checks, name, passed, details):
    """Print and retain one independently assessable comparison."""
    checks.append({"name": name, "passed": bool(passed), "details": details})
    print(f"{'PASS' if passed else 'FAIL'} {name}: {details}")


def compare_tensor(checks, name, reference, native, rtol=RTOL, atol=ATOL):
    """Check shape and values at the fixed forward tolerance."""
    same_shape = reference.shape == native.shape
    passed = same_shape and torch.allclose(reference, native, rtol=rtol, atol=atol)
    difference = (reference - native).abs().max().item() if same_shape and reference.numel() else None
    if difference is not None and not math.isfinite(difference):
        difference = None
    record_check(checks, name, passed, {"reference_shape": list(reference.shape),
                                      "native_shape": list(native.shape), "max_abs_difference": difference})


def compare_forwards(checks, reference, native, data, samples):
    """Use real resized test samples and a shared normalization for both pure models."""
    config_path = os.path.join(PROJECT_ROOT, "src", "visionlab", "anomaly_detection", "configs", "data", f"{data}.yaml")
    with open(config_path, encoding="utf-8") as stream:
        config = yaml.safe_load(stream)["data"]["init_args"]
    dataset_class = MVTecDataset if data == "mvtec" else VisADataset
    dataset = dataset_class(root=config["root"], category=config["category"], split="test")
    if not len(dataset):
        raise ValueError(f"Empty {data} test dataset")
    count = min(samples, len(dataset))
    images = torch.stack([dataset[index][0] for index in range(count)])
    images = PreProcessor(mean=IMAGENET_MEAN, std=IMAGENET_STD)(images)
    record_check(checks, f"FR-5 {data} inputs", True,
                 {"root": config["root"], "category": config["category"],
                  "image_paths": [sample["image_path"] for sample in dataset.samples[:count]]})
    reference.eval()
    native.eval()
    with torch.no_grad():
        expected = reference(images)
        actual = native(images)
        compare_tensor(checks, f"FR-5 {data} eval pred_score", expected.pred_score, actual.pred_score)
        compare_tensor(checks, f"FR-5 {data} eval anomaly_map", expected.anomaly_map, actual.anomaly_map)
        reference.train()
        native.train()
        expected_features = reference(images)
        actual_features = native(images)
    for branch, expected, actual in zip(("teacher", "student"), expected_features, actual_features):
        record_check(checks, f"FR-5 {data} train {branch} keys", list(expected) == list(actual),
                     {"reference": list(expected), "native": list(actual)})
        for name in sorted(expected.keys() & actual.keys()):
            compare_tensor(checks, f"FR-5 {data} train {branch} {name}", expected[name], actual[name])


def compare_processing(checks):
    """Compare metric adapters on equivalent score ordering and post-processing on identical CPU data."""
    generator = torch.Generator().manual_seed(42)
    scores = torch.randint(0, 257, (262144,), generator=generator).float() / 256
    labels = torch.randint(0, 2, scores.shape, generator=generator)
    labels[:2] = torch.tensor([0, 1])
    for name, metric, reference_metric, prediction, field in (
        ("AUROC", AUROC(), ReferenceAUROC(fields=["pred_score", "gt_label"]), scores, "pred_score"),
        ("AUROC raw", AUROC(), ReferenceAUROC(fields=["pred_score", "gt_label"]), scores * 32 - 8, "pred_score"),
        ("F1", F1(), F1Score(fields=["pred_label", "gt_label"]), scores > 0.5, "pred_label"),
    ):
        metric.update(prediction, labels)
        actual = metric.compute()
        metric_batch = ImageBatch(image=torch.zeros(scores.numel(), 3, 1, 1), gt_label=labels)
        # Keep reference AUROC inputs in [0, 1] to avoid its sigmoid collapsing raw-score ranks.
        setattr(metric_batch, field, scores if name == "AUROC raw" else prediction)
        reference_metric.update(metric_batch)
        expected = reference_metric.compute().item()
        record_check(checks, f"metrics {name}", abs(actual - expected) <= METRIC_ATOL,
                     {"reference": expected, "native": actual, "elements": scores.numel()})
    raw_scores = scores * 32 - 8
    image_scores = raw_scores[:4096].reshape(-1, 1)
    image_labels = labels[:4096]
    maps = raw_scores.reshape(4096, 1, 8, 8)
    masks = labels.reshape(4096, 8, 8)
    native = PostProcessor()
    native.update({"pred_score": image_scores, "gt_label": image_labels, "anomaly_map": maps, "gt_mask": masks})
    native.finalize()
    reference = ReferencePostProcessor()
    batch = ImageBatch(image=torch.zeros(4096, 3, 8, 8), pred_score=image_scores,
                       gt_label=image_labels, anomaly_map=maps, gt_mask=masks)
    reference.on_validation_batch_end(None, None, batch)
    reference.on_validation_epoch_end(None, None)
    reference_threshold = F1AdaptiveThreshold(fields=["pred_score", "gt_label"])
    reference_threshold.update(batch)
    expected_threshold = reference_threshold.compute().item()
    record_check(checks, "FR-24 F1AdaptiveThreshold", abs(native.image_threshold.item() - expected_threshold) <= METRIC_ATOL,
                 {"reference": expected_threshold, "native": native.image_threshold.item()})
    reference_state = {"image_threshold": reference._image_threshold, "pixel_threshold": reference._pixel_threshold,
                       "image_min": reference.image_min, "image_max": reference.image_max,
                       "pixel_min": reference.pixel_min, "pixel_max": reference.pixel_max}
    for name, actual in native.state_dict().items():
        expected = reference_state[name].detach().cpu()
        record_check(checks, f"post_processor {name}", torch.equal(expected, actual),
                     {"reference": expected.item(), "native": actual.item()})
    processed = native(image_scores, maps)
    for name, values, minimum, maximum, threshold, sensitivity in (
        ("pred_score", image_scores, reference.image_min, reference.image_max,
         reference.image_threshold, reference.normalized_image_threshold),
        ("anomaly_map", maps, reference.pixel_min, reference.pixel_max,
         reference.pixel_threshold, reference.normalized_pixel_threshold),
    ):
        expected = reference._normalize(values, minimum, maximum, threshold)
        compare_tensor(checks, f"post_processor normalized {name}", expected, processed[name])
        label = "pred_label" if name == "pred_score" else "pred_mask"
        expected_label = reference._apply_threshold(expected, sensitivity)
        record_check(checks, f"post_processor {label}", torch.equal(expected_label, processed[label]),
                     {"threshold": sensitivity.item()})


def compare_efficient_ad_forwards(checks, reference, native, data, samples):
    """Compare real 256 inputs, calibrated maps, distances and deterministic student loss."""
    config_path = os.path.join(PROJECT_ROOT, "src", "visionlab", "anomaly_detection", "configs", "data", f"{data}.yaml")
    with open(config_path, encoding="utf-8") as stream:
        config = yaml.safe_load(stream)["data"]["init_args"]
    dataset_class = MVTecDataset if data == "mvtec" else VisADataset
    dataset = dataset_class(root=config["root"], category=config["category"], split="test")
    if not len(dataset):
        raise ValueError(f"Empty {data} test dataset")
    count = min(samples, len(dataset))
    images = torch.stack([dataset[index][0] for index in range(count)])
    assert images.shape[1:] == (3, 256, 256)
    record_check(checks, f"FR-5 {data} inputs", True,
                 {"root": config["root"], "category": config["category"],
                  "image_paths": [sample["image_path"] for sample in dataset.samples[:count]]})
    for calibrated in (False, True):
        with torch.no_grad():
            reference.mean_std["mean"].fill_(0.2 if calibrated else 0)
            reference.mean_std["std"].fill_(1.3 if calibrated else 0)
            for name in reference.quantiles:
                reference.quantiles[name].fill_((0.8 if name.startswith("qb") else 0.1) if calibrated else 0)
        native.load_state_dict(reference.state_dict(), strict=True)
        reference.eval()
        native.eval()
        prefix = f"FR-5 {data} {'calibrated' if calibrated else 'unset'}"
        with torch.no_grad():
            expected, actual = reference(images), native(images)
            pairs = [("pred_score", expected.pred_score, actual.pred_score),
                     ("anomaly_map", expected.anomaly_map, actual.anomaly_map)]
            expected_distance = reference.compute_student_teacher_distance(images)
            actual_distance = native.compute_student_teacher_distance(images)
            pairs.extend((name, left, right) for name, left, right in
                         zip(("student_output", "distance_st"), expected_distance, actual_distance, strict=True))
            pairs.extend((name, left, right) for name, left, right in
                         zip(("map_st raw", "map_stae raw"), reference.get_maps(images, normalize=False),
                             native.get_maps(images, normalize=False), strict=True))
            for name, left, right in pairs:
                compare_tensor(checks, f"{prefix} {name}", left, right,
                               rtol=EFFICIENT_AD_RTOL, atol=EFFICIENT_AD_ATOL)
            reference.train()
            native.train()
            batch_imagenet = images[:1].flip(-1)
            expected_loss = reference(batch=images[:1], batch_imagenet=batch_imagenet)[0]
            actual_loss = native(batch=images[:1], batch_imagenet=batch_imagenet)[0]
            compare_tensor(checks, f"{prefix} loss_st", expected_loss, actual_loss,
                           rtol=EFFICIENT_AD_RTOL, atol=EFFICIENT_AD_ATOL)


def compare_efficient_ad(checks, samples):
    """Check EfficientAD constructor, injected teacher and original pure-model computations."""
    record_check(checks, "FR-6 constructor signature", signature_fields(EfficientAd) == signature_fields(EfficientAdLitModel),
                 {"reference": signature_fields(EfficientAd), "native": signature_fields(EfficientAdLitModel)})
    config_path = os.path.join(PROJECT_ROOT, "src", "visionlab", "anomaly_detection", "configs", "models", "efficient_ad.yaml")
    with open(config_path, encoding="utf-8") as stream:
        config = yaml.safe_load(stream)["model"]["init_args"]
    native = EfficientAdLitModel(imagenet_dir=config["imagenet_dir"], model_size=config["model_size"],
                                 pre_processor=False, post_processor=False, evaluator=False, visualizer=False)
    reference = LocalEfficientAd(teacher_file=native.teacher_weights_path, imagenet_dir=config["imagenet_dir"],
                                 model_size=config["model_size"], pre_processor=False, post_processor=False,
                                 evaluator=False, visualizer=False)
    reference.prepare_pretrained_model()
    expected, actual = reference.model.teacher.state_dict(), native.model.teacher.state_dict()
    mismatches = sorted(name for name in expected.keys() & actual.keys() if not torch.equal(expected[name], actual[name]))
    record_check(checks, "FR-23 local teacher", expected.keys() == actual.keys() and not mismatches,
                 {"missing": sorted(expected.keys() - actual.keys()), "unexpected": sorted(actual.keys() - expected.keys()),
                  "mismatches": mismatches, "weights": native.teacher_weights_path,
                  "sha256": file_sha256(native.teacher_weights_path)})
    pure = EfficientAdModel(teacher_out_channels=384, model_size=config["model_size"])
    for data in ("mvtec", "visa"):
        try:
            compare_efficient_ad_forwards(checks, reference.model, pure, data, samples)
        except Exception as error:
            record_check(checks, f"FR-5 {data}", False, f"{type(error).__name__}: {error}")


def compare_padim_forwards(checks, reference, native, data, samples):
    """Compare PaDiM maps, distances and blur on actual normalized 256-pixel images."""
    config_path = os.path.join(PROJECT_ROOT, "src", "visionlab", "anomaly_detection", "configs", "data", f"{data}.yaml")
    with open(config_path, encoding="utf-8") as stream:
        config = yaml.safe_load(stream)["data"]["init_args"]
    dataset_class = MVTecDataset if data == "mvtec" else VisADataset
    dataset = dataset_class(root=config["root"], category=config["category"], split="test")
    if not len(dataset):
        raise ValueError(f"Empty {data} test dataset")
    count = min(samples, len(dataset))
    images = PreProcessor(mean=IMAGENET_MEAN, std=IMAGENET_STD)(
        torch.stack([dataset[index][0] for index in range(count)]))
    assert images.shape[1:] == (3, 256, 256)
    record_check(checks, f"FR-5 {data} inputs", True,
                 {"root": config["root"], "category": config["category"],
                  "image_paths": [sample["image_path"] for sample in dataset.samples[:count]]})
    reference.eval()
    native.eval()
    with torch.no_grad():
        expected, actual = reference(images), native(images)
        pairs = [("pred_score", expected.pred_score, actual.pred_score),
                 ("anomaly_map", expected.anomaly_map, actual.anomaly_map)]
        reference_embedding = reference.generate_embedding(reference.feature_extractor(images))
        native_embedding = native.generate_embedding(native.feature_extractor(images))
        expected_distance = reference.anomaly_map_generator.compute_distance(
            reference_embedding, [reference.gaussian.mean, reference.gaussian.inv_covariance])
        actual_distance = native.anomaly_map_generator.compute_distance(
            native_embedding, [native.gaussian.mean, native.gaussian.inv_covariance])
        pairs.append(("compute_distance", expected_distance, actual_distance))
        blur_input = reference.anomaly_map_generator.up_sample(expected_distance, (256, 256))
        pairs.append(("GaussianBlur2d", reference.anomaly_map_generator.blur(blur_input),
                      native.anomaly_map_generator.blur(blur_input)))
    for name, left, right in pairs:
        compare_tensor(checks, f"FR-5 {data} {name}", left, right, rtol=PADIM_RTOL, atol=PADIM_ATOL)


def compare_padim(checks, samples, reference_checkpoint):
    """Compare PaDiM constructors, local backbone, seeded indices and shared fitted states."""
    record_check(checks, "FR-6 constructor signature", signature_fields(Padim) == signature_fields(PadimLitModel),
                 {"reference": signature_fields(Padim), "native": signature_fields(PadimLitModel)})
    random_state = random.getstate()
    try:
        random.seed(42)
        reference = Padim(backbone="resnet18", pre_trained=True, pre_processor=False,
                          post_processor=False, evaluator=False, visualizer=False)
        random.seed(42)
        native = PadimLitModel(backbone="resnet18", pre_trained=True, pre_processor=False,
                               post_processor=False, evaluator=False, visualizer=False)
    finally:
        random.setstate(random_state)
    expected = reference.model.feature_extractor.state_dict()
    actual = native.model.feature_extractor.state_dict()
    mismatches = sorted(name for name in expected.keys() & actual.keys() if not torch.equal(expected[name], actual[name]))
    record_check(checks, "FR-23 local backbone", expected.keys() == actual.keys() and not mismatches,
                 {"missing": sorted(expected.keys() - actual.keys()), "unexpected": sorted(actual.keys() - expected.keys()),
                  "mismatches": mismatches, "weights": native.teacher_weights_path,
                  "sha256": file_sha256(native.teacher_weights_path)})
    record_check(checks, "seeded idx", torch.equal(reference.model.idx, native.model.idx),
                 {"seed": 42, "reference": reference.model.idx.tolist(), "native": native.model.idx.tolist()})
    if reference_checkpoint:
        checkpoint = torch.load(reference_checkpoint, map_location="cpu", weights_only=False)
        state = {name.removeprefix("model."): value for name, value in checkpoint["state_dict"].items()
                 if name.startswith("model.")}
        if not state:
            raise ValueError("Reference checkpoint has no model.* state")
        channels = state["idx"].numel()
    else:
        state = reference.model.state_dict()
        channels = reference.model.n_features
    pure_reference = ReferencePadimModel(backbone="resnet18", pre_trained=False, n_features=channels)
    pure_native = PadimModel(backbone="resnet18", pre_trained=False, n_features=channels)
    pure_reference.load_state_dict(state, strict=True)
    pure_native.load_state_dict(state, strict=True)
    if reference_checkpoint:
        record_check(checks, "FR-5 reference checkpoint strict restore", True,
                     {"path": os.path.abspath(reference_checkpoint), "sha256": file_sha256(reference_checkpoint)})
    else:
        embedding = torch.randn(8, channels, 64, 64, generator=torch.Generator().manual_seed(42))
        expected_stats = pure_reference.gaussian.fit(embedding)
        actual_stats = pure_native.gaussian.fit(embedding)
        for name, left, right in zip(("mean", "inv_covariance"), expected_stats, actual_stats, strict=True):
            compare_tensor(checks, f"FR-5 synthetic {name}", left, right, rtol=PADIM_RTOL, atol=PADIM_ATOL)
    for data in ("mvtec", "visa"):
        try:
            compare_padim_forwards(checks, pure_reference, pure_native, data, samples)
        except Exception as error:
            record_check(checks, f"FR-5 {data}", False, f"{type(error).__name__}: {error}")


def main():
    """Run every comparison group and write a machine-readable PASS or FAIL manifest."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=("stfpm", "efficient_ad", "padim"), required=True)
    parser.add_argument("--reference-checkpoint")
    parser.add_argument("--samples", type=int, default=8)
    args = parser.parse_args()
    if args.samples < 1:
        parser.error("--samples must be positive")
    checks = []
    rtol, atol = (EFFICIENT_AD_RTOL, EFFICIENT_AD_ATOL) if args.model == "efficient_ad" else (RTOL, ATOL)
    if args.model == "padim":
        rtol, atol = PADIM_RTOL, PADIM_ATOL
    print(f"Fixed tolerances: rtol={rtol}, atol={atol}, metric_atol={METRIC_ATOL}")
    if args.model == "padim":
        try:
            compare_padim(checks, args.samples, args.reference_checkpoint)
        except Exception as error:
            record_check(checks, "FR-23 / FR-5 model construction", False, f"{type(error).__name__}: {error}")
    elif args.model == "efficient_ad":
        try:
            compare_efficient_ad(checks, args.samples)
        except Exception as error:
            record_check(checks, "FR-23 / FR-5 model construction", False, f"{type(error).__name__}: {error}")
    else:
        compare_stfpm(checks, args.samples)
    try:
        compare_processing(checks)
    except Exception as error:
        record_check(checks, "metrics / post_processor", False, f"{type(error).__name__}: {error}")
    report = {"model": args.model, "rtol": rtol, "atol": atol, "metric_atol": METRIC_ATOL,
              "passed": all(check["passed"] for check in checks), "checks": checks}
    path = os.path.join(PROJECT_ROOT, "outputs", "compare", f"{args.model}_outputs.json")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(f"Manifest: {path}")
    return 0 if report["passed"] else 1


def compare_stfpm(checks, samples):
    """Keep the original STFPM teacher and forward comparison unchanged."""
    record_check(checks, "FR-6 constructor signature", signature_fields(Stfpm) == signature_fields(StfpmLitModel),
                 {"reference": signature_fields(Stfpm), "native": signature_fields(StfpmLitModel)})
    try:
        reference = Stfpm(pre_processor=False, post_processor=False, evaluator=False, visualizer=False)
        native = StfpmLitModel(pre_processor=False, post_processor=False, evaluator=False, visualizer=False)
        expected = reference.model.teacher_model.state_dict()
        actual = native.model.teacher_model.state_dict()
        mismatches = sorted(name for name in expected.keys() & actual.keys() if not torch.equal(expected[name], actual[name]))
        record_check(checks, "FR-23 local teacher", expected.keys() == actual.keys() and not mismatches,
                     {"missing": sorted(expected.keys() - actual.keys()), "unexpected": sorted(actual.keys() - expected.keys()),
                      "mismatches": mismatches, "weights": native.teacher_weights_path,
                      "sha256": file_sha256(native.teacher_weights_path)})
        pure = STFPMModel(backbone="resnet18", layers=("layer1", "layer2", "layer3"))
        pure.load_state_dict(reference.model.state_dict(), strict=True)
        for data in ("mvtec", "visa"):
            try:
                compare_forwards(checks, reference.model, pure, data, samples)
            except Exception as error:
                record_check(checks, f"FR-5 {data}", False, f"{type(error).__name__}: {error}")
    except Exception as error:
        record_check(checks, "FR-23 / FR-5 model construction", False, f"{type(error).__name__}: {error}")


if __name__ == "__main__":
    sys.exit(main())
