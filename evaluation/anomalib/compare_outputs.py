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
from enum import Enum
from types import SimpleNamespace

import torch
import yaml
from jsonargparse import ArgumentParser
from anomalib.data import ImageBatch
from anomalib.metrics import AUROC as ReferenceAUROC
from anomalib.metrics import F1AdaptiveThreshold, F1Score
from anomalib.models import Cfa, Dfkde, Dfm, EfficientAd, Fre, Padim, Patchcore, ReverseDistillation, Stfpm, Supersimplenet
from anomalib.data.utils.generators import generate_perlin_noise as reference_perlin_noise
from anomalib.models.image.supersimplenet.torch_model import SupersimplenetModel as ReferenceSupersimplenetModel
from anomalib.models.image.supersimplenet.anomaly_generator import AnomalyGenerator as ReferenceAnomalyGenerator
from anomalib.models.image.supersimplenet.loss import SSNLoss as ReferenceSSNLoss
from anomalib.models.image.cfa import torch_model as reference_cfa_module
from anomalib.models.image.fre.torch_model import FREModel as ReferenceFREModel
from anomalib.models.image.dfkde.torch_model import DfkdeModel as ReferenceDfkdeModel
from anomalib.models.image.dfm.torch_model import DFMModel as ReferenceDFMModel
from anomalib.models.image.padim.torch_model import PadimModel as ReferencePadimModel
from anomalib.models.image.patchcore.torch_model import PatchcoreModel as ReferencePatchcoreModel
from anomalib.models.image.reverse_distillation.torch_model import ReverseDistillationModel as ReferenceReverseDistillationModel
from anomalib.models.image.reverse_distillation.anomaly_map import AnomalyMapGenerationMode as ReferenceReverseMode
from anomalib.post_processing import PostProcessor as ReferencePostProcessor
from anomalib.models import Draem
from anomalib.models.image.draem.torch_model import DraemModel as ReferenceDraemModel
from anomalib.models.image.draem.loss import DraemLoss as ReferenceDraemLoss
from anomalib.data.utils.generators.perlin import PerlinAnomalyGenerator as ReferencePerlinAnomalyGenerator

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
from visionlab.anomaly_detection.models.dfm.lightning_model import DfmLitModel
from visionlab.anomaly_detection.models.dfm.model import DFMModel
from visionlab.anomaly_detection.models.dfkde.lightning_model import DfkdeLitModel
from visionlab.anomaly_detection.models.dfkde.model import DfkdeModel
from visionlab.anomaly_detection.models.patchcore.lightning_model import PatchcoreLitModel
from visionlab.anomaly_detection.models.patchcore.model import PatchcoreModel
from visionlab.anomaly_detection.models.reverse_distillation.lightning_model import ReverseDistillationLitModel
from visionlab.anomaly_detection.models.reverse_distillation.model import ReverseDistillationModel
from visionlab.anomaly_detection.models.reverse_distillation.anomaly_map import AnomalyMapGenerationMode
from visionlab.anomaly_detection.models.cfa.lightning_model import CfaLitModel
from visionlab.anomaly_detection.models.cfa.model import CfaModel
from visionlab.anomaly_detection.models.fre.lightning_model import FreLitModel
from visionlab.anomaly_detection.models.fre.model import FREModel
from visionlab.anomaly_detection.common.perlin import generate_perlin_noise
from visionlab.anomaly_detection.models.supersimplenet.lightning_model import SupersimplenetLitModel
from visionlab.anomaly_detection.models.supersimplenet.model import SupersimplenetModel
from visionlab.anomaly_detection.models.supersimplenet.anomaly_generator import AnomalyGenerator
from visionlab.anomaly_detection.models.supersimplenet.loss import SSNLoss
from visionlab.utils.io import file_sha256
from visionlab.anomaly_detection.common.perlin import PerlinAnomalyGenerator
from visionlab.anomaly_detection.models.draem.lightning_model import DraemLitModel
from visionlab.anomaly_detection.models.draem.model import DraemModel
from visionlab.anomaly_detection.models.draem.loss import DraemLoss
from local_models import LocalCfa, LocalEfficientAd

RTOL = 1e-4
ATOL = 1e-5
METRIC_ATOL = 1e-6
EFFICIENT_AD_RTOL = 1e-5
EFFICIENT_AD_ATOL = 1e-6
PADIM_RTOL = 1e-5
PADIM_ATOL = 1e-6
DFM_RTOL = 1e-5
DFM_ATOL = 1e-6
DFKDE_RTOL = 1e-5
DFKDE_ATOL = 1e-6
PATCHCORE_RTOL = 1e-5
PATCHCORE_ATOL = 1e-6
REVERSE_DISTILLATION_RTOL = 1e-5
REVERSE_DISTILLATION_ATOL = 1e-6
CFA_RTOL = 1e-5
CFA_ATOL = 1e-6
FRE_RTOL = 1e-5
FRE_ATOL = 1e-6
SUPERSIMPLENET_RTOL = 1e-5
SUPERSIMPLENET_ATOL = 1e-6
DRAEM_RTOL = 1e-5
DRAEM_ATOL = 1e-6


def signature_fields(model_class, enum_values=False):
    """Compare constructor names, order, kinds and default representations without annotations."""
    return [{"name": name, "kind": parameter.kind.name,
             "default": repr(parameter.default.value if enum_values and isinstance(parameter.default, Enum)
                             else parameter.default)}
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


def compare_dfm_forwards(checks, reference, native, data, samples):
    """Compare original FRE outputs on separately cloned real normalized image batches."""
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
        expected, actual = reference(images.clone()), native(images.clone())
    compare_tensor(checks, f"FR-5 {data} pred_score", expected.pred_score, actual.pred_score,
                   rtol=DFM_RTOL, atol=DFM_ATOL)
    compare_tensor(checks, f"FR-5 {data} anomaly_map", expected.anomaly_map, actual.anomaly_map,
                   rtol=DFM_RTOL, atol=DFM_ATOL)


def compare_dfm(checks, samples, reference_checkpoint):
    """Compare DFM constructors, local backbone weights and fitted PCA reconstruction outputs."""
    record_check(checks, "FR-6 constructor signature", signature_fields(Dfm) == signature_fields(DfmLitModel),
                 {"reference": signature_fields(Dfm), "native": signature_fields(DfmLitModel)})
    reference = Dfm(backbone="resnet50", pre_trained=True, pre_processor=False,
                    post_processor=False, evaluator=False, visualizer=False)
    native = DfmLitModel(backbone="resnet50", pre_trained=True, pre_processor=False,
                         post_processor=False, evaluator=False, visualizer=False)
    expected = reference.model.feature_extractor.state_dict()
    actual = native.model.feature_extractor.state_dict()
    mismatches = sorted(name for name in expected.keys() & actual.keys() if not torch.equal(expected[name], actual[name]))
    record_check(checks, "FR-23 local backbone", expected.keys() == actual.keys() and not mismatches,
                 {"missing": sorted(expected.keys() - actual.keys()), "unexpected": sorted(actual.keys() - expected.keys()),
                  "mismatches": mismatches, "weights": native.teacher_weights_path,
                  "sha256": file_sha256(native.teacher_weights_path)})
    pure_reference = ReferenceDFMModel(backbone="resnet50", layer="layer3", pre_trained=False)
    pure_native = DFMModel(backbone="resnet50", layer="layer3", pre_trained=False)
    if reference_checkpoint:
        checkpoint = torch.load(reference_checkpoint, map_location="cpu", weights_only=False)
        state = {name.removeprefix("model."): value for name, value in checkpoint["state_dict"].items()
                 if name.startswith("model.")}
        if not state:
            raise ValueError("Reference checkpoint has no model.* state")
    else:
        state = reference.model.state_dict()
    pure_reference.load_state_dict(state, strict=True)
    pure_native.load_state_dict(state, strict=True)
    if reference_checkpoint:
        record_check(checks, "FR-5 reference checkpoint strict restore", True,
                     {"path": os.path.abspath(reference_checkpoint), "sha256": file_sha256(reference_checkpoint)})
    else:
        pure_reference.train()
        with torch.no_grad():
            dimensions = pure_reference.get_features(torch.zeros(1, 3, 256, 256)).shape[1]
        features = torch.randn(12, dimensions, generator=torch.Generator().manual_seed(42))
        pure_reference.fit(features.clone())
        pure_native.fit(features.clone())
        left, right = pure_reference.pca_model, pure_native.pca_model
        for name in ("mean", "num_components"):
            compare_tensor(checks, f"FR-5 synthetic PCA {name}", getattr(left, name), getattr(right, name),
                           rtol=DFM_RTOL, atol=DFM_ATOL)
        same_shape = left.singular_vectors.shape == right.singular_vectors.shape
        k = left.singular_vectors.shape[1]
        overlap = (left.singular_vectors.double().T @ right.singular_vectors.double()).square().sum().item() if same_shape else None
        record_check(checks, "FR-5 synthetic PCA subspace", same_shape and math.isclose(overlap, k, rel_tol=DFM_RTOL, abs_tol=DFM_ATOL),
                     {"reference_shape": list(left.singular_vectors.shape), "native_shape": list(right.singular_vectors.shape),
                      "components": k, "cross_gram_squared_frobenius": overlap})
    for data in ("mvtec", "visa"):
        try:
            compare_dfm_forwards(checks, pure_reference, pure_native, data, samples)
        except Exception as error:
            record_check(checks, f"FR-5 {data}", False, f"{type(error).__name__}: {error}")


def compare_dfkde_forwards(checks, reference, native, data, samples):
    """Compare image-only KDE scores on separately cloned normalized real samples."""
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
        expected, actual = reference(images.clone()), native(images.clone())
    compare_tensor(checks, f"FR-5 {data} pred_score", expected.pred_score, actual.pred_score,
                   rtol=DFKDE_RTOL, atol=DFKDE_ATOL)
    record_check(checks, f"FR-5 {data} image-only output", expected.anomaly_map is None and actual.anomaly_map is None,
                 {"reference_anomaly_map_none": expected.anomaly_map is None, "native_anomaly_map_none": actual.anomaly_map is None})


def compare_dfkde(checks, samples, reference_checkpoint):
    """Compare DFKDE signatures, injected backbone and restored or synthetic KDE scores."""
    expected_signature = signature_fields(Dfkde, enum_values=True)
    actual_signature = signature_fields(DfkdeLitModel, enum_values=True)
    record_check(checks, "FR-6 constructor signature", expected_signature == actual_signature,
                 {"reference": expected_signature, "native": actual_signature})
    reference = Dfkde(backbone="resnet18", pre_trained=True, pre_processor=False,
                      post_processor=False, evaluator=False, visualizer=False)
    native = DfkdeLitModel(backbone="resnet18", pre_trained=True, pre_processor=False,
                           post_processor=False, evaluator=False, visualizer=False)
    expected = reference.model.feature_extractor.state_dict()
    actual = native.model.feature_extractor.state_dict()
    mismatches = sorted(name for name in expected.keys() & actual.keys() if not torch.equal(expected[name], actual[name]))
    record_check(checks, "FR-23 local backbone", expected.keys() == actual.keys() and not mismatches,
                 {"missing": sorted(expected.keys() - actual.keys()), "unexpected": sorted(actual.keys() - expected.keys()),
                  "mismatches": mismatches, "weights": native.teacher_weights_path,
                  "sha256": file_sha256(native.teacher_weights_path)})
    pure_reference = ReferenceDfkdeModel(backbone="resnet18", layers=["layer4"], pre_trained=False)
    pure_native = DfkdeModel(backbone="resnet18", layers=["layer4"], pre_trained=False)
    if reference_checkpoint:
        checkpoint = torch.load(reference_checkpoint, map_location="cpu", weights_only=False)
        state = {name.removeprefix("model."): value for name, value in checkpoint["state_dict"].items()
                 if name.startswith("model.")}
        if not state:
            raise ValueError("Reference checkpoint has no model.* state")
    else:
        state = reference.model.state_dict()
    pure_reference.load_state_dict(state, strict=True)
    pure_native.load_state_dict(state, strict=True)
    if reference_checkpoint:
        record_check(checks, "FR-5 reference checkpoint strict restore", True,
                     {"path": os.path.abspath(reference_checkpoint), "sha256": file_sha256(reference_checkpoint)})
    else:
        generator = torch.Generator().manual_seed(42)
        embeddings = torch.randn(64, 512, generator=generator)
        left, right = pure_reference.classifier, pure_native.classifier
        fitted_left, fitted_right = left.fit(embeddings.clone()), right.fit(embeddings.clone())
        record_check(checks, "FR-5 synthetic KDE fitting", fitted_left and fitted_right,
                     {"reference": fitted_left, "native": fitted_right})
        features = torch.randn(8, 512, generator=generator)
        compare_tensor(checks, "FR-5 synthetic KDE scores", left(features.clone()), right(features.clone()),
                       rtol=DFKDE_RTOL, atol=DFKDE_ATOL)
        for name, expected_value, actual_value in (
            ("max_length", left.max_length, right.max_length),
            ("kde_model.norm", left.kde_model.norm, right.kde_model.norm),
        ):
            compare_tensor(checks, f"FR-5 synthetic {name}", expected_value, actual_value,
                           rtol=DFKDE_RTOL, atol=DFKDE_ATOL)
    for data in ("mvtec", "visa"):
        try:
            compare_dfkde_forwards(checks, pure_reference, pure_native, data, samples)
        except Exception as error:
            record_check(checks, f"FR-5 {data}", False, f"{type(error).__name__}: {error}")


def compare_patchcore_forwards(checks, reference, native, data, samples):
    """Compare PatchCore scores and maps on shared central crops of real resized samples."""
    config_path = os.path.join(PROJECT_ROOT, "src", "visionlab", "anomaly_detection", "configs", "data", f"{data}.yaml")
    with open(config_path, encoding="utf-8") as stream:
        config = yaml.safe_load(stream)["data"]["init_args"]
    dataset_class = MVTecDataset if data == "mvtec" else VisADataset
    dataset = dataset_class(root=config["root"], category=config["category"], split="test")
    if not len(dataset):
        raise ValueError(f"Empty {data} test dataset")
    count = min(samples, len(dataset))
    images = PreProcessor(mean=IMAGENET_MEAN, std=IMAGENET_STD, center_crop=(224, 224))(
        torch.stack([dataset[index][0] for index in range(count)]))
    assert images.shape[1:] == (3, 224, 224)
    record_check(checks, f"FR-5 {data} inputs", True,
                 {"root": config["root"], "category": config["category"],
                  "image_paths": [sample["image_path"] for sample in dataset.samples[:count]]})
    reference.eval()
    native.eval()
    with torch.no_grad():
        expected, actual = reference(images.clone()), native(images.clone())
    compare_tensor(checks, f"FR-5 {data} pred_score", expected.pred_score, actual.pred_score,
                   rtol=PATCHCORE_RTOL, atol=PATCHCORE_ATOL)
    compare_tensor(checks, f"FR-5 {data} anomaly_map", expected.anomaly_map, actual.anomaly_map,
                   rtol=PATCHCORE_RTOL, atol=PATCHCORE_ATOL)


def compare_patchcore(checks, samples, reference_checkpoint):
    """Compare PatchCore signatures, local weights and shared checkpoint or synthetic memory banks."""
    record_check(checks, "FR-6 constructor signature",
                 signature_fields(Patchcore) == signature_fields(PatchcoreLitModel),
                 {"reference": signature_fields(Patchcore), "native": signature_fields(PatchcoreLitModel)})
    reference = Patchcore(backbone="wide_resnet50_2", pre_trained=True, pre_processor=False,
                          post_processor=False, evaluator=False, visualizer=False)
    native = PatchcoreLitModel(backbone="wide_resnet50_2", pre_trained=True, pre_processor=False,
                               post_processor=False, evaluator=False, visualizer=False)
    expected = reference.model.feature_extractor.state_dict()
    actual = native.model.feature_extractor.state_dict()
    mismatches = sorted(name for name in expected.keys() & actual.keys() if not torch.equal(expected[name], actual[name]))
    record_check(checks, "FR-23 local backbone", expected.keys() == actual.keys() and not mismatches,
                 {"missing": sorted(expected.keys() - actual.keys()), "unexpected": sorted(actual.keys() - expected.keys()),
                  "mismatches": mismatches, "weights": native.teacher_weights_path,
                  "sha256": file_sha256(native.teacher_weights_path)})
    pure_reference = ReferencePatchcoreModel(layers=("layer2", "layer3"), backbone="wide_resnet50_2", pre_trained=False)
    pure_native = PatchcoreModel(layers=("layer2", "layer3"), backbone="wide_resnet50_2", pre_trained=False)
    if reference_checkpoint:
        checkpoint = torch.load(reference_checkpoint, map_location="cpu", weights_only=False)
        state = {name.removeprefix("model."): value for name, value in checkpoint["state_dict"].items()
                 if name.startswith("model.")}
        if not state:
            raise ValueError("Reference checkpoint has no model.* state")
    else:
        state = reference.model.state_dict()
        # A shared synthetic bank isolates forward equivalence from stochastic coreset selection.
        state["memory_bank"] = torch.randn(64, 1536, generator=torch.Generator().manual_seed(42))
    pure_reference.load_state_dict(state, strict=True)
    pure_native.load_state_dict(state, strict=True)
    record_check(checks, "FR-5 shared memory bank strict restore", True,
                 {"source": os.path.abspath(reference_checkpoint) if reference_checkpoint else "synthetic",
                  "memory_bank_shape": list(state["memory_bank"].shape)})
    del reference, native, state, expected, actual
    if reference_checkpoint:
        del checkpoint
    for data in ("mvtec", "visa"):
        try:
            compare_patchcore_forwards(checks, pure_reference, pure_native, data, samples)
        except Exception as error:
            record_check(checks, f"FR-5 {data}", False, f"{type(error).__name__}: {error}")


def compare_reverse_distillation_forwards(checks, reference, native, data, samples):
    """Compare Reverse Distillation raw outputs on shared normalized 256-pixel real samples."""
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
        expected, actual = reference(images.clone()), native(images.clone())
    compare_tensor(checks, f"FR-5 {data} pred_score", expected.pred_score, actual.pred_score,
                   rtol=REVERSE_DISTILLATION_RTOL, atol=REVERSE_DISTILLATION_ATOL)
    compare_tensor(checks, f"FR-5 {data} anomaly_map", expected.anomaly_map, actual.anomaly_map,
                   rtol=REVERSE_DISTILLATION_RTOL, atol=REVERSE_DISTILLATION_ATOL)


def compare_reverse_distillation(checks, information, samples, reference_checkpoint):
    """Compare signatures, local encoder, shared raw forwards and informational seeded reconstruction weights."""
    expected_signature = signature_fields(ReverseDistillation, enum_values=True)
    actual_signature = signature_fields(ReverseDistillationLitModel, enum_values=True)
    record_check(checks, "FR-6 constructor signature", expected_signature == actual_signature,
                 {"reference": expected_signature, "native": actual_signature})
    # The reference needs its default pre-processor to discover input_size.
    reference = ReverseDistillation(backbone="wide_resnet50_2", pre_trained=True,
                                    post_processor=False, evaluator=False, visualizer=False)
    native = ReverseDistillationLitModel(backbone="wide_resnet50_2", pre_trained=True,
                                         post_processor=False, evaluator=False, visualizer=False)
    expected, actual = reference.model.encoder.state_dict(), native.model.encoder.state_dict()
    mismatches = sorted(name for name in expected.keys() & actual.keys() if not torch.equal(expected[name], actual[name]))
    record_check(checks, "FR-23 local encoder", expected.keys() == actual.keys() and not mismatches,
                 {"missing": sorted(expected.keys() - actual.keys()), "unexpected": sorted(actual.keys() - expected.keys()),
                  "mismatches": mismatches, "weights": native.teacher_weights_path,
                  "sha256": file_sha256(native.teacher_weights_path)})
    if reference_checkpoint:
        checkpoint = torch.load(reference_checkpoint, map_location="cpu", weights_only=False)
        state = {name.removeprefix("model."): value for name, value in checkpoint["state_dict"].items()
                 if name.startswith("model.")}
        if not state:
            raise ValueError("Reference checkpoint has no model.* state")
        del checkpoint
    else:
        state = reference.model.state_dict()
    del reference, native, expected, actual
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(42)
        pure_reference = ReferenceReverseDistillationModel(
            backbone="wide_resnet50_2", input_size=(256, 256), layers=("layer1", "layer2", "layer3"),
            anomaly_map_mode=ReferenceReverseMode.ADD, pre_trained=False)
        torch.random.default_generator.manual_seed(42)
        pure_native = ReverseDistillationModel(
            backbone="wide_resnet50_2", input_size=(256, 256), layers=("layer1", "layer2", "layer3"),
            anomaly_map_mode=AnomalyMapGenerationMode.ADD, pre_trained=False)
    for branch in ("bottleneck", "decoder"):
        left, right = getattr(pure_reference, branch).state_dict(), getattr(pure_native, branch).state_dict()
        mismatches = sorted(name for name in left.keys() & right.keys() if not torch.equal(left[name], right[name]))
        entry = {"name": f"seeded initial {branch}", "informational_only": True, "seed": 42,
                 "equal": left.keys() == right.keys() and not mismatches,
                 "missing": sorted(left.keys() - right.keys()), "unexpected": sorted(right.keys() - left.keys()),
                 "mismatches": mismatches}
        information.append(entry)
        print(f"INFO {entry['name']}: {entry}")
    del left, right
    pure_reference.load_state_dict(state, strict=True)
    pure_native.load_state_dict(state, strict=True)
    record_check(checks, "FR-5 shared model strict restore", True,
                 {"source": os.path.abspath(reference_checkpoint) if reference_checkpoint else "reference random initial state"})
    del state
    for data in ("mvtec", "visa"):
        try:
            compare_reverse_distillation_forwards(checks, pure_reference, pure_native, data, samples)
        except Exception as error:
            record_check(checks, f"FR-5 {data}", False, f"{type(error).__name__}: {error}")


def compare_cfa_forwards(checks, reference, native, data, samples):
    """Compare CFA raw outputs on shared normalized 256-pixel real samples."""
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
        expected, actual = reference(images.clone()), native(images.clone())
    compare_tensor(checks, f"FR-5 {data} pred_score", expected.pred_score, actual.pred_score,
                   rtol=CFA_RTOL, atol=CFA_ATOL)
    compare_tensor(checks, f"FR-5 {data} anomaly_map", expected.anomaly_map, actual.anomaly_map,
                   rtol=CFA_RTOL, atol=CFA_ATOL)


def compare_cfa(checks, information, samples, reference_checkpoint):
    """Compare CFA signatures, local weights, shared centroids and strict-restored real forwards."""
    record_check(checks, "FR-6 constructor signature", signature_fields(Cfa) == signature_fields(CfaLitModel),
                 {"reference": signature_fields(Cfa), "native": signature_fields(CfaLitModel)})
    native = CfaLitModel(backbone="wide_resnet50_2", pre_processor=False, post_processor=False,
                          evaluator=False, visualizer=False)
    original_factory = reference_cfa_module.get_feature_extractor
    try:
        reference = LocalCfa(backbone="wide_resnet50_2", weights_file=native.teacher_weights_path,
                             pre_processor=False, post_processor=False, evaluator=False, visualizer=False)
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(42)
            seeded_reference = reference_cfa_module.CfaModel(
                backbone="wide_resnet50_2", gamma_c=1, gamma_d=1, num_nearest_neighbors=3,
                num_hard_negative_features=3, radius=1e-5)
    finally:
        reference_cfa_module.get_feature_extractor = original_factory
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(42)
        pure = CfaModel(backbone="wide_resnet50_2", gamma_c=1, gamma_d=1, num_nearest_neighbors=3,
                        num_hard_negative_features=3, radius=1e-5)
    left, right = seeded_reference.descriptor.state_dict(), pure.descriptor.state_dict()
    mismatches = sorted(name for name in left.keys() & right.keys() if not torch.equal(left[name], right[name]))
    entry = {"name": "seeded initial descriptor", "informational_only": True, "seed": 42,
             "equal": left.keys() == right.keys() and not mismatches, "mismatches": mismatches}
    information.append(entry)
    print(f"INFO {entry['name']}: {entry}")
    del seeded_reference, left, right
    expected, actual = reference.model.feature_extractor.state_dict(), native.model.feature_extractor.state_dict()
    mismatches = sorted(name for name in expected.keys() & actual.keys() if not torch.equal(expected[name], actual[name]))
    record_check(checks, "FR-23 local backbone", expected.keys() == actual.keys() and not mismatches,
                 {"missing": sorted(expected.keys() - actual.keys()), "unexpected": sorted(actual.keys() - expected.keys()),
                  "mismatches": mismatches, "weights": native.teacher_weights_path,
                  "sha256": file_sha256(native.teacher_weights_path)})
    if reference_checkpoint:
        checkpoint = torch.load(reference_checkpoint, map_location="cpu", weights_only=False)
        state = {name.removeprefix("model."): value for name, value in checkpoint["state_dict"].items()
                 if name.startswith("model.")}
        if not state:
            raise ValueError("Reference checkpoint has no model.* state")
        del checkpoint
    else:
        state = reference.model.state_dict()
        state["memory_bank"] = torch.randn(1792, 64, generator=torch.Generator().manual_seed(42))
    reference.model.load_state_dict(state, strict=True)
    pure.load_state_dict(state, strict=True)
    record_check(checks, "FR-5 shared model strict restore", True,
                 {"source": os.path.abspath(reference_checkpoint) if reference_checkpoint else "shared synthetic memory bank",
                  "memory_bank_shape": list(state["memory_bank"].shape)})
    del state, native, expected, actual
    for data in ("mvtec", "visa"):
        try:
            compare_cfa_forwards(checks, reference.model, pure, data, samples)
        except Exception as error:
            record_check(checks, f"FR-5 {data}", False, f"{type(error).__name__}: {error}")
    # Fresh scalar banks reproduce the initial centroid pass without overwriting the checkpoint comparison.
    reference.model.memory_bank = reference.model.memory_bank.new_zeros(())
    pure.memory_bank = pure.memory_bank.new_zeros(())
    reference.model.eval()
    pure.eval()
    generator = torch.Generator().manual_seed(42)
    images = [torch.rand(1, 3, 256, 256, generator=generator) for _ in range(2)]
    reference.model.initialize_centroid([SimpleNamespace(image=image.clone()) for image in images])
    pure.initialize_centroid([(image.clone(), {}) for image in images])
    compare_tensor(checks, "FR-5 raw centroid initialization", reference.model.memory_bank, pure.memory_bank,
                   rtol=CFA_RTOL, atol=CFA_ATOL)


def compare_fre_forwards(checks, reference, native, data, samples):
    """Compare FRE raw outputs on shared normalized 256-pixel real samples."""
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
        expected, actual = reference(images.clone()), native(images.clone())
    compare_tensor(checks, f"FR-5 {data} pred_score", expected.pred_score, actual.pred_score,
                   rtol=FRE_RTOL, atol=FRE_ATOL)
    compare_tensor(checks, f"FR-5 {data} anomaly_map", expected.anomaly_map, actual.anomaly_map,
                   rtol=FRE_RTOL, atol=FRE_ATOL)


def compare_fre(checks, information, samples, reference_checkpoint):
    """Compare FRE signatures, injected backbone, shared forwards and informational tied initialization."""
    record_check(checks, "FR-6 constructor signature", signature_fields(Fre) == signature_fields(FreLitModel),
                 {"reference": signature_fields(Fre), "native": signature_fields(FreLitModel)})
    reference = Fre(backbone="resnet50", pre_trained=True, pre_processor=False,
                    post_processor=False, evaluator=False, visualizer=False)
    native = FreLitModel(backbone="resnet50", pre_trained=True, pre_processor=False,
                         post_processor=False, evaluator=False, visualizer=False)
    expected, actual = reference.model.feature_extractor.state_dict(), native.model.feature_extractor.state_dict()
    mismatches = sorted(name for name in expected.keys() & actual.keys() if not torch.equal(expected[name], actual[name]))
    record_check(checks, "FR-23 local backbone", expected.keys() == actual.keys() and not mismatches,
                 {"missing": sorted(expected.keys() - actual.keys()), "unexpected": sorted(actual.keys() - expected.keys()),
                  "mismatches": mismatches, "weights": native.teacher_weights_path,
                  "sha256": file_sha256(native.teacher_weights_path)})
    if reference_checkpoint:
        checkpoint = torch.load(reference_checkpoint, map_location="cpu", weights_only=False)
        state = {name.removeprefix("model."): value for name, value in checkpoint["state_dict"].items()
                 if name.startswith("model.")}
        if not state:
            raise ValueError("Reference checkpoint has no model.* state")
        del checkpoint
    else:
        state = reference.model.state_dict()
    del reference, native, expected, actual
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(42)
        pure_reference = ReferenceFREModel(backbone="resnet50", layer="layer3", input_dim=65536,
                                           latent_dim=220, pre_trained=False, pooling_kernel_size=2)
        torch.random.default_generator.manual_seed(42)
        pure_native = FREModel(backbone="resnet50", layer="layer3", input_dim=65536,
                               latent_dim=220, pre_trained=False, pooling_kernel_size=2)
    for name in ("weight", "encoder_bias", "decoder_bias"):
        entry = {"name": f"seeded initial TiedAE {name}", "informational_only": True, "seed": 42,
                 "equal": torch.equal(getattr(pure_reference.fre_model, name), getattr(pure_native.fre_model, name))}
        information.append(entry)
        print(f"INFO {entry['name']}: {entry}")
    pure_reference.load_state_dict(state, strict=True)
    pure_native.load_state_dict(state, strict=True)
    record_check(checks, "FR-5 shared model strict restore", True,
                 {"source": os.path.abspath(reference_checkpoint) if reference_checkpoint else "reference random initial state"})
    del state
    for data in ("mvtec", "visa"):
        try:
            compare_fre_forwards(checks, pure_reference, pure_native, data, samples)
        except Exception as error:
            record_check(checks, f"FR-5 {data}", False, f"{type(error).__name__}: {error}")


def compare_supersimplenet_randomness(checks):
    """Compare CPU and available CUDA Perlin and default-device synthetic features under matched seeds."""
    devices = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])
    with torch.random.fork_rng(devices=list(range(torch.cuda.device_count()))):
        for name in devices:
            device = torch.device(name)
            torch.manual_seed(42)
            expected = reference_perlin_noise(height=64, width=64, device=device)
            torch.manual_seed(42)
            actual = generate_perlin_noise(height=64, width=64, device=device)
            record_check(checks, f"Perlin exact RNG {name}", torch.equal(expected, actual),
                         {"device": name, "seed": 42, "shape": list(actual.shape)})
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        features = torch.rand(2, 8, 32, 40, device=device)
        masks, labels = torch.zeros(2, 1, 32, 40, device=device), torch.zeros(2, device=device)
        reference = ReferenceAnomalyGenerator(noise_mean=0, noise_std=0.015, threshold=0.2)
        native = AnomalyGenerator(noise_mean=0, noise_std=0.015, threshold=0.2)
        torch.manual_seed(42)
        expected = reference(features.clone(), masks.clone(), labels.clone())
        torch.manual_seed(42)
        actual = native(features.clone(), masks.clone(), labels.clone())
    for name, left, right in zip(("perturbed", "masks", "labels"), expected, actual, strict=True):
        compare_tensor(checks, f"synthetic anomalies {device} {name}", left, right,
                       rtol=SUPERSIMPLENET_RTOL, atol=SUPERSIMPLENET_ATOL)


def compare_supersimplenet_forwards(checks, reference, native, state, data, samples, device):
    """Reload independent BN states for matched training, loss and checkpoint evaluation comparisons."""
    config_path = os.path.join(PROJECT_ROOT, "src", "visionlab", "anomaly_detection", "configs", "data", f"{data}.yaml")
    with open(config_path, encoding="utf-8") as stream:
        config = yaml.safe_load(stream)["data"]["init_args"]
    dataset_class = MVTecDataset if data == "mvtec" else VisADataset
    dataset = dataset_class(root=config["root"], category=config["category"], split="test")
    if not len(dataset):
        raise ValueError(f"Empty {data} test dataset")
    count = min(samples, len(dataset))
    images = PreProcessor(mean=IMAGENET_MEAN, std=IMAGENET_STD)(
        torch.stack([dataset[index][0] for index in range(count)])).to(device)
    assert images.shape[1:] == (3, 256, 256)
    masks = torch.zeros(count, 256, 256, dtype=torch.int64, device=device)
    labels = torch.zeros(count, dtype=torch.int64, device=device)
    record_check(checks, f"FR-5 {data} inputs", True,
                 {"root": config["root"], "category": config["category"], "device": str(device),
                  "image_paths": [sample["image_path"] for sample in dataset.samples[:count]]})
    reference.load_state_dict(state, strict=True)
    native.load_state_dict(state, strict=True)
    reference.train()
    native.train()
    with torch.no_grad(), torch.random.fork_rng(devices=list(range(torch.cuda.device_count()))):
        torch.manual_seed(42)
        expected = reference(images=images.clone(), masks=masks.clone(), labels=labels.clone())
        torch.manual_seed(42)
        actual = native(images=images.clone(), masks=masks.clone(), labels=labels.clone())
        for name, left, right in zip(("anomaly_map", "anomaly_score", "masks", "labels"), expected, actual, strict=True):
            compare_tensor(checks, f"FR-5 {data} train {name}", left, right,
                           rtol=SUPERSIMPLENET_RTOL, atol=SUPERSIMPLENET_ATOL)
        expected_loss = ReferenceSSNLoss()(pred_map=expected[0], pred_score=expected[1],
                                           target_mask=expected[2], target_label=expected[3])
        actual_loss = SSNLoss()(pred_map=actual[0], pred_score=actual[1], target_mask=actual[2], target_label=actual[3])
        compare_tensor(checks, f"FR-5 {data} SSNLoss", expected_loss, actual_loss,
                       rtol=SUPERSIMPLENET_RTOL, atol=SUPERSIMPLENET_ATOL)
    # Discard training-forward BN updates before checking the supplied evaluation state.
    reference.load_state_dict(state, strict=True)
    native.load_state_dict(state, strict=True)
    reference.eval()
    native.eval()
    with torch.no_grad():
        expected, actual = reference(images.clone()), native(images.clone())
    for name in ("pred_score", "anomaly_map"):
        compare_tensor(checks, f"FR-5 {data} eval {name}", getattr(expected, name), getattr(actual, name),
                       rtol=SUPERSIMPLENET_RTOL, atol=SUPERSIMPLENET_ATOL)
    record_check(checks, f"FR-5 {data} eval shapes",
                 actual.anomaly_map.shape == (count, 1, 256, 256) and actual.pred_score.shape == (() if count == 1 else (count,)),
                 {"anomaly_map": list(actual.anomaly_map.shape), "pred_score": list(actual.pred_score.shape)})


def compare_supersimplenet(checks, information, samples, reference_checkpoint):
    """Compare signatures, local backbone, initialized heads, synthesis and shared real forwards."""
    record_check(checks, "FR-6 constructor signature",
                 signature_fields(Supersimplenet) == signature_fields(SupersimplenetLitModel),
                 {"reference": signature_fields(Supersimplenet), "native": signature_fields(SupersimplenetLitModel)})
    compare_supersimplenet_randomness(checks)
    reference = Supersimplenet(backbone="wide_resnet50_2", pre_processor=False, post_processor=False,
                               evaluator=False, visualizer=False)
    native = SupersimplenetLitModel(backbone="wide_resnet50_2", pre_processor=False, post_processor=False,
                                    evaluator=False, visualizer=False)
    expected = reference.model.feature_extractor.feature_extractor.state_dict()
    actual = native.model.feature_extractor.feature_extractor.state_dict()
    mismatches = sorted(name for name in expected.keys() & actual.keys() if not torch.equal(expected[name], actual[name]))
    record_check(checks, "FR-23 local backbone", expected.keys() == actual.keys() and not mismatches,
                 {"missing": sorted(expected.keys() - actual.keys()), "unexpected": sorted(actual.keys() - expected.keys()),
                  "mismatches": mismatches, "weights": native.teacher_weights_path,
                  "sha256": file_sha256(native.teacher_weights_path)})
    if reference_checkpoint:
        checkpoint = torch.load(reference_checkpoint, map_location="cpu", weights_only=False)
        state = {name.removeprefix("model."): value for name, value in checkpoint["state_dict"].items()
                 if name.startswith("model.")}
        if not state:
            raise ValueError("Reference checkpoint has no model.* state")
        del checkpoint
    else:
        state = reference.model.state_dict()
    del reference, native, expected, actual
    with torch.random.fork_rng(devices=list(range(torch.cuda.device_count()))):
        torch.manual_seed(42)
        pure_reference = ReferenceSupersimplenetModel(perlin_threshold=0.2, backbone="wide_resnet50_2",
                                                      layers=["layer2", "layer3"], stop_grad=True)
        torch.manual_seed(42)
        pure_native = SupersimplenetModel(perlin_threshold=0.2, backbone="wide_resnet50_2",
                                          layers=["layer2", "layer3"], stop_grad=True)
    for branch in ("adaptor", "segdec"):
        left, right = getattr(pure_reference, branch).state_dict(), getattr(pure_native, branch).state_dict()
        mismatches = sorted(name for name in left.keys() & right.keys() if not torch.equal(left[name], right[name]))
        entry = {"name": f"seeded initial {branch}", "informational_only": True, "seed": 42,
                 "equal": left.keys() == right.keys() and not mismatches,
                 "missing": sorted(left.keys() - right.keys()), "unexpected": sorted(right.keys() - left.keys()),
                 "mismatches": mismatches}
        information.append(entry)
        print(f"INFO {entry['name']}: {entry}")
    del left, right
    pure_reference.load_state_dict(state, strict=True)
    pure_native.load_state_dict(state, strict=True)
    record_check(checks, "FR-5 shared model strict restore", True,
                 {"source": os.path.abspath(reference_checkpoint) if reference_checkpoint else "reference random initial state"})
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pure_reference.to(device)
    pure_native.to(device)
    for data in ("mvtec", "visa"):
        try:
            compare_supersimplenet_forwards(checks, pure_reference, pure_native, state, data, samples, device)
        except Exception as error:
            record_check(checks, f"FR-5 {data}", False, f"{type(error).__name__}: {error}")


def compare_draem_forwards(checks, reference, native, state, images, augmented, mask, prefix):
    """Restore BatchNorm state separately for training and evaluation comparisons."""
    for model in (reference, native):
        model.load_state_dict(state, strict=True)
        model.train()
    with torch.no_grad():
        expected_reconstruction, expected_prediction = reference(augmented)
        actual_reconstruction, actual_prediction = native(augmented)
        expected_loss = ReferenceDraemLoss().to(images.device)(
            images, expected_reconstruction, mask, expected_prediction,
        )
        actual_loss = DraemLoss().to(images.device)(images, actual_reconstruction, mask, actual_prediction)
    for name, expected, actual in (
        ("train reconstruction", expected_reconstruction, actual_reconstruction),
        ("train prediction", expected_prediction, actual_prediction), ("loss", expected_loss, actual_loss),
    ):
        compare_tensor(checks, f"{prefix} {name}", expected, actual, rtol=DRAEM_RTOL, atol=DRAEM_ATOL)
    for model in (reference, native):
        model.load_state_dict(state, strict=True)
        model.eval()
    with torch.no_grad():
        expected, actual = reference(images), native(images)
    for name in ("pred_score", "anomaly_map"):
        compare_tensor(checks, f"{prefix} eval {name}", getattr(expected, name), getattr(actual, name),
                       rtol=DRAEM_RTOL, atol=DRAEM_ATOL)
    count, _, height, width = images.shape
    record_check(checks, f"{prefix} eval shapes",
                 actual.pred_score.shape == expected.pred_score.shape == (count,) and
                 actual.anomaly_map.shape == expected.anomaly_map.shape == (count, height, width),
                 {"pred_score": list(actual.pred_score.shape), "anomaly_map": list(actual.anomaly_map.shape)})


def compare_draem(checks, information, samples, reference_checkpoint):
    """Compare DRAEM YAML beta, seeded synthesis and shared-state forwards on MVTec only."""
    record_check(checks, "FR-6 constructor signature", signature_fields(Draem) == signature_fields(DraemLitModel),
                 {"reference": signature_fields(Draem), "native": signature_fields(DraemLitModel)})
    config_path = os.path.join(PROJECT_ROOT, "evaluation", "anomalib", "models", "11_draem.yaml")
    with open(config_path, encoding="utf-8") as stream:
        options = yaml.safe_load(stream)["model"]["init_args"]
    source_path = options["anomaly_source_path"]
    parser = ArgumentParser()
    parser.add_class_arguments(Draem, "model")
    parsed = parser.parse_object({"model": {
        "anomaly_source_path": source_path, "beta": options["beta"], "pre_processor": False,
        "post_processor": False, "evaluator": False, "visualizer": False,
    }})
    with torch.random.fork_rng(devices=list(range(torch.cuda.device_count()))):
        torch.manual_seed(42)
        reference = parser.instantiate_classes(parsed).model
        torch.manual_seed(42)
        native = DraemLitModel(anomaly_source_path=source_path, beta=options["beta"],
                               pre_processor=False, post_processor=False, evaluator=False, visualizer=False)
    record_check(checks, "beta YAML tuple",
                 isinstance(reference.augmenter.blend_factor, tuple) and
                 isinstance(native.augmenter.blend_factor, tuple) and
                 reference.augmenter.blend_factor == native.augmenter.blend_factor,
                 {"reference": reference.augmenter.blend_factor, "native": native.augmenter.blend_factor})
    left, right = reference.model.state_dict(), native.model.state_dict()
    mismatches = sorted(name for name in left.keys() & right.keys() if not torch.equal(left[name], right[name]))
    entry = {"name": "seeded DRAEM initialization", "informational_only": True, "seed": 42,
             "equal": left.keys() == right.keys() and not mismatches, "mismatches": mismatches,
             "reference_solarize_threshold": reference.augmenter.augmenters.transforms[4].threshold,
             "native_solarize_threshold": native.augmenter.augmenters.transforms[4].threshold,
             "solarize_equal": reference.augmenter.augmenters.transforms[4].threshold ==
                               native.augmenter.augmenters.transforms[4].threshold}
    information.append(entry)
    print(f"INFO {entry['name']}: {entry}")
    if reference_checkpoint:
        checkpoint = torch.load(reference_checkpoint, map_location="cpu", weights_only=False)
        state = {name.removeprefix("model."): value for name, value in checkpoint["state_dict"].items()
                 if name.startswith("model.")}
        if not state:
            raise ValueError("Reference checkpoint has no model.* state")
        del checkpoint
    else:
        state = {name: value.clone() for name, value in left.items()}
    del reference, native, left, right
    with open(os.path.join(PROJECT_ROOT, "src", "visionlab", "anomaly_detection", "configs", "data", "mvtec.yaml"),
              encoding="utf-8") as stream:
        config = yaml.safe_load(stream)["data"]["init_args"]
    dataset = MVTecDataset(root=config["root"], category=config["category"], split="test")
    if len(dataset) < samples:
        raise ValueError(f"Expected {samples} MVTec samples, found {len(dataset)}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    images = torch.stack([dataset[index][0] for index in range(samples)]).to(device)
    record_check(checks, "FR-5 mvtec inputs",
                 images.shape == (samples, 3, 256, 256) and
                 images.min() >= -DRAEM_ATOL and images.max() <= 1 + DRAEM_ATOL,
                 {"device": str(device), "category": config["category"],
                  "shape": list(images.shape), "min": images.min().item(), "max": images.max().item(),
                  "image_paths": [sample["image_path"] for sample in dataset.samples[:samples]]})
    pure_reference, pure_native = ReferenceDraemModel().to(device), DraemModel().to(device)
    record_check(checks, "FR-5 shared state source", True,
                 {"source": os.path.abspath(reference_checkpoint) if reference_checkpoint else "reference initial state"})
    for path in (None, source_path):
        label = "noise" if path is None else "DTD"
        with torch.random.fork_rng(devices=list(range(torch.cuda.device_count()))):
            torch.manual_seed(42)
            reference_generator = ReferencePerlinAnomalyGenerator(anomaly_source_path=path, blend_factor=(0.1, 1.0))
            torch.manual_seed(42)
            native_generator = PerlinAnomalyGenerator(anomaly_source_path=path, blend_factor=(0.1, 1.0))
            torch.manual_seed(43)
            expected_augmented, expected_mask = reference_generator(images)
            torch.manual_seed(43)
            actual_augmented, actual_mask = native_generator(images)
        compare_tensor(checks, f"augmentation {label} image", expected_augmented, actual_augmented,
                       rtol=DRAEM_RTOL, atol=DRAEM_ATOL)
        compare_tensor(checks, f"augmentation {label} mask", expected_mask, actual_mask,
                       rtol=DRAEM_RTOL, atol=DRAEM_ATOL)
        compare_draem_forwards(checks, pure_reference, pure_native, state, images, expected_augmented, expected_mask,
                               f"FR-5 mvtec {label}")
    del pure_reference, pure_native, state
    torch.manual_seed(42)
    pure_reference = ReferenceDraemModel(sspcab=True).to(device)
    pure_native = DraemModel(sspcab=True).to(device)
    state = {name: value.detach().cpu().clone() for name, value in pure_reference.state_dict().items()}
    compare_draem_forwards(checks, pure_reference, pure_native, state, images, expected_augmented, expected_mask,
                           "SSPCAB mvtec")


def main():
    """Run every comparison group and write a machine-readable PASS or FAIL manifest."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=("stfpm", "efficient_ad", "padim", "dfm", "dfkde", "patchcore",
                                          "reverse_distillation", "cfa", "fre", "supersimplenet", "draem"), required=True)
    parser.add_argument("--reference-checkpoint")
    parser.add_argument("--samples", type=int, default=8)
    args = parser.parse_args()
    if args.samples < 1:
        parser.error("--samples must be positive")
    checks = []
    information = []
    rtol, atol = (EFFICIENT_AD_RTOL, EFFICIENT_AD_ATOL) if args.model == "efficient_ad" else (RTOL, ATOL)
    if args.model == "padim":
        rtol, atol = PADIM_RTOL, PADIM_ATOL
    elif args.model == "dfm":
        rtol, atol = DFM_RTOL, DFM_ATOL
    elif args.model == "dfkde":
        rtol, atol = DFKDE_RTOL, DFKDE_ATOL
    elif args.model == "patchcore":
        rtol, atol = PATCHCORE_RTOL, PATCHCORE_ATOL
    elif args.model == "reverse_distillation":
        rtol, atol = REVERSE_DISTILLATION_RTOL, REVERSE_DISTILLATION_ATOL
    elif args.model == "cfa":
        rtol, atol = CFA_RTOL, CFA_ATOL
    elif args.model == "fre":
        rtol, atol = FRE_RTOL, FRE_ATOL
    elif args.model == "supersimplenet":
        rtol, atol = SUPERSIMPLENET_RTOL, SUPERSIMPLENET_ATOL
    elif args.model == "draem":
        rtol, atol = DRAEM_RTOL, DRAEM_ATOL
    print(f"Fixed tolerances: rtol={rtol}, atol={atol}, metric_atol={METRIC_ATOL}")
    if args.model == "draem":
        try:
            compare_draem(checks, information, args.samples, args.reference_checkpoint)
        except Exception as error:
            record_check(checks, "DRAEM construction / augmentation / FR-5", False, f"{type(error).__name__}: {error}")
    elif args.model == "supersimplenet":
        try:
            compare_supersimplenet(checks, information, args.samples, args.reference_checkpoint)
        except Exception as error:
            record_check(checks, "FR-23 / FR-5 model construction", False, f"{type(error).__name__}: {error}")
    elif args.model == "fre":
        try:
            compare_fre(checks, information, args.samples, args.reference_checkpoint)
        except Exception as error:
            record_check(checks, "FR-23 / FR-5 model construction", False, f"{type(error).__name__}: {error}")
    elif args.model == "cfa":
        try:
            compare_cfa(checks, information, args.samples, args.reference_checkpoint)
        except Exception as error:
            record_check(checks, "FR-23 / FR-5 model construction", False, f"{type(error).__name__}: {error}")
    elif args.model == "reverse_distillation":
        try:
            compare_reverse_distillation(checks, information, args.samples, args.reference_checkpoint)
        except Exception as error:
            record_check(checks, "FR-23 / FR-5 model construction", False, f"{type(error).__name__}: {error}")
    elif args.model == "patchcore":
        try:
            compare_patchcore(checks, args.samples, args.reference_checkpoint)
        except Exception as error:
            record_check(checks, "FR-23 / FR-5 model construction", False, f"{type(error).__name__}: {error}")
    elif args.model == "dfkde":
        try:
            compare_dfkde(checks, args.samples, args.reference_checkpoint)
        except Exception as error:
            record_check(checks, "FR-23 / FR-5 model construction", False, f"{type(error).__name__}: {error}")
    elif args.model == "dfm":
        try:
            compare_dfm(checks, args.samples, args.reference_checkpoint)
        except Exception as error:
            record_check(checks, "FR-23 / FR-5 model construction", False, f"{type(error).__name__}: {error}")
    elif args.model == "padim":
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
    if args.model in ("reverse_distillation", "cfa", "fre", "supersimplenet", "draem"):
        report["information"] = information
    if args.model == "draem":
        report["not_applicable"] = {"FR-23": "DRAEM has no pretrained backbone or teacher weights"}
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
