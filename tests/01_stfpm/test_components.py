# tests/01_stfpm/test_components.py: pytest contracts for torchmetrics and pure processing components

import math
import os

import numpy as np
import pytest
import torch
from torchmetrics.classification import BinaryPrecisionRecallCurve

from visionlab.anomaly_detection.common.evaluator import Evaluator
from visionlab.anomaly_detection.common.learning_type import LearningType
from visionlab.anomaly_detection.common.metric import AUROC, F1
from visionlab.anomaly_detection.common.post_processor import PostProcessor
from visionlab.anomaly_detection.common.pre_processor import IMAGENET_MEAN, IMAGENET_STD, PreProcessor
from visionlab.anomaly_detection.common.visualizer import ImageVisualizer


@pytest.fixture
def calibration_outputs():
    """Provide raw scores with a hand-computed optimal threshold of two."""
    scores, labels = torch.arange(4, dtype=torch.float32).reshape(4, 1), torch.tensor([0, 0, 1, 1])
    return {"pred_score": scores, "gt_label": labels,
            "anomaly_map": scores.reshape(4, 1, 1, 1).clone(), "gt_mask": labels.reshape(4, 1, 1).clone()}


@pytest.fixture
def evaluation_outputs():
    """Provide raw and binary image and pixel fields with known metric values."""
    scores = torch.tensor([0.1, 0.9, 0.8, 0.4, 0.3, 0.7]).reshape(6, 1)
    labels = torch.tensor([0, 1, 1, 0, 1, 0])
    maps = scores.reshape(6, 1, 1, 1).expand(6, 1, 2, 2).clone()
    return {"pred_score": scores, "gt_label": labels, "anomaly_map": maps,
            "gt_mask": labels.reshape(6, 1, 1).expand(6, 2, 2).clone(),
            "pred_label": scores > 0.5, "pred_mask": maps > 0.5}


def assert_state_equal(left, right):
    """Compare calibration tensor snapshots including uncalibrated NaNs."""
    assert set(left) == set(right)
    for name in left:
        torch.testing.assert_close(left[name], right[name], equal_nan=True)


def test_normalization_and_identity():
    """Normalize each channel exactly once without changing the source tensor."""
    image = torch.tensor([0.0, 0.5, 1.0], dtype=torch.float64).reshape(1, 3, 1, 1).expand(2, 3, 4, 5).clone()
    original = image.clone()
    result = PreProcessor(mean=IMAGENET_MEAN, std=IMAGENET_STD)(image)
    for channel in range(3):
        torch.testing.assert_close(result[:, channel], (original[:, channel] - IMAGENET_MEAN[channel]) / IMAGENET_STD[channel])
    assert result.dtype == image.dtype and result.data_ptr() != image.data_ptr()
    assert torch.equal(image, original) and torch.equal(PreProcessor()(image), original)


def test_invalid_inputs_and_statistics():
    """Reject malformed images and inconsistent normalization statistics."""
    for image in (torch.zeros(3, 4, 5), torch.zeros(1, 1, 4, 5), torch.zeros(1, 3, 4, 5, dtype=torch.uint8),
                  torch.zeros(0, 3, 4, 5), torch.full((1, 3, 1, 1), float("nan")), [0.0]):
        with pytest.raises(ValueError):
            PreProcessor()(image)
    for mean, std in ((IMAGENET_MEAN, None), (None, IMAGENET_STD), ((0, 0), (1, 1)), ((0, 0, 0), (1, 0, 1))):
        with pytest.raises(ValueError):
            PreProcessor(mean=mean, std=std)


def test_hand_computed_and_random_auroc():
    """Match small ROC examples, exact ties and independently calculated pairwise expectations."""
    metric = AUROC()
    metric.update(torch.tensor([0.1, 0.4, 0.35, 0.8]), torch.tensor([0, 0, 1, 1]))
    assert metric.compute() == pytest.approx(0.75, abs=1e-6)
    metric.reset()
    metric.update(torch.ones(4), torch.tensor([0, 1, 0, 1]))
    assert metric.compute() == 0.5
    generator = torch.Generator().manual_seed(42)
    for _ in range(8):
        scores = torch.randint(-3, 4, (30,), generator=generator).double()
        labels = torch.arange(30) % 2
        positive, negative = scores[labels == 1].unsqueeze(1), scores[labels == 0].unsqueeze(0)
        expected = ((positive > negative).double() + 0.5 * (positive == negative).double()).mean().item()
        metric.reset()
        metric.update(scores[:7], labels[:7])
        metric.update(scores[7:], labels[7:])
        assert metric.compute() == pytest.approx(expected, abs=1e-6)
        assert metric.reason is None


def test_pixel_scale_and_f1():
    """Retain million-pixel ROC coverage and known binary F1 including a defined zero."""
    target = torch.arange(1 << 20) % 2
    metric = AUROC()
    metric.update(target.float(), target)
    assert metric.compute() == 1.0
    metric = F1()
    metric.update(torch.tensor([True, False, True, False]), torch.tensor([1, 1, 0, 0]))
    assert metric.compute() == 0.5
    metric.reset()
    metric.update(torch.zeros(2, dtype=torch.bool), torch.tensor([0, 1]))
    assert metric.compute() == 0 and metric.reason is None


def test_metric_batching_reset_and_snapshot():
    """Keep one-shot and split updates equal and snapshot incoming CPU tensors."""
    scores, labels = torch.tensor([0.1, 0.4, 0.4, 0.8, 0.2, 0.8]), torch.tensor([0, 1, 0, 1, 1, 0])
    for metric_class, prediction in ((AUROC, scores), (F1, scores > 0.5)):
        whole, split = metric_class(), metric_class()
        whole.update(prediction, labels)
        for start in range(0, 6, 2):
            split.update(prediction[start:start + 2], labels[start:start + 2])
        expected = whole.compute()
        assert split.compute() == expected
        changed = prediction.clone()
        whole.reset()
        whole.update(changed, labels)
        changed.zero_()
        assert whole.compute() == expected
        assert whole.metric.device.type == "cpu"
        whole.reset()
        assert whole.reason is None and math.isnan(whole.compute()) and whole.reason == "no_data"


def test_undefined_and_invalid_data():
    """Return NaN reasons before metric computation and reject invalid binary inputs."""
    for metric_class in (AUROC, F1):
        metric = metric_class()
        assert math.isnan(metric.compute()) and metric.reason == "no_data"
        for label, reason in ((0, "no_positive_targets"), (1, "no_negative_targets")):
            metric.reset()
            metric.update(torch.tensor([False, True]), torch.full((2,), label))
            assert math.isnan(metric.compute()) and metric.reason == reason
        metric.reset()
        assert metric.reason is None
        for prediction, target in ((torch.ones(2), torch.ones(3)), (torch.ones(2), torch.tensor([0, 2])),
                                   ([0, 1], torch.tensor([0, 1])), (torch.tensor([float("nan")]), torch.tensor([0])),
                                   (torch.ones(1), torch.tensor([float("inf")])),
                                   (torch.ones(1, dtype=torch.complex64), torch.tensor([0]))):
            with pytest.raises(ValueError):
                metric.update(prediction, target)
    with pytest.raises(ValueError):
        F1().update(torch.tensor([0.1, 0.8]), torch.tensor([0, 1]))


def test_calibration_and_immutable_inference(calibration_outputs):
    """Commit calibration only on finalize and preserve raw tensors through inference and restore."""
    processor, outputs = PostProcessor(), calibration_outputs
    state = processor.state_dict()
    assert not processor.is_calibrated
    for start in (0, 2):
        processor.update({name: tensor[start:start + 2] for name, tensor in outputs.items()})
    assert_state_equal(state, processor.state_dict())
    processor.finalize()
    assert processor.is_calibrated
    for level in ("image", "pixel"):
        assert processor.state_dict()[f"{level}_threshold"] == 2
        assert processor.state_dict()[f"{level}_min"] == 0 and processor.state_dict()[f"{level}_max"] == 3
    before = processor.state_dict()
    raw_score, raw_map = outputs["pred_score"].clone(), outputs["anomaly_map"].clone()
    prediction = processor(raw_score, raw_map)
    expected = torch.tensor([0, 1 / 6, 0.5, 5 / 6]).reshape(4, 1)
    torch.testing.assert_close(prediction["pred_score"], expected)
    torch.testing.assert_close(prediction["anomaly_map"].flatten(), expected.flatten())
    assert prediction["pred_label"].flatten().tolist() == prediction["pred_mask"].flatten().tolist() == [False, False, False, True]
    assert torch.equal(raw_score, outputs["pred_score"]) and torch.equal(raw_map, outputs["anomaly_map"])
    assert_state_equal(before, processor.state_dict())
    processor.finalize()
    assert_state_equal(before, processor.state_dict())
    restored = PostProcessor()
    restored.load_state_dict(before)
    for name, value in restored(raw_score, raw_map).items():
        torch.testing.assert_close(value, prediction[name])
    before["image_min"].fill_(-100)
    assert restored.image_min == 0


def test_snapshot_and_tied_f1_threshold(calibration_outputs):
    """Use the reference PR formula and preserve the first threshold for tied maximal F1."""
    processor, outputs = PostProcessor(), calibration_outputs
    outputs["gt_label"] = torch.tensor([1, 0, 0, 1])
    outputs["gt_mask"] = outputs["gt_label"].reshape(4, 1, 1)
    curve = BinaryPrecisionRecallCurve(normalization=None)
    curve.update(outputs["pred_score"].flatten(), outputs["gt_label"])
    precision, recall, thresholds = curve.compute()
    f1 = 2 * precision * recall / (precision + recall + 1e-10)
    expected = thresholds if thresholds.ndim == 0 else thresholds[torch.argmax(f1)]
    assert expected == 0
    processor.update(outputs)
    outputs["pred_score"].add_(100)
    outputs["anomaly_map"].add_(100)
    processor.finalize()
    assert processor.image_threshold == processor.pixel_threshold == expected
    assert processor.image_max == 3
    processor.update(outputs)
    processor.finalize()
    assert processor.image_min == 100 and processor.image_max == 103


def test_uncalibrated_midpoint_and_cross_level_fallback():
    """Retain raw-copy, midpoint and opposite-level threshold fallbacks."""
    processor = PostProcessor()
    score = torch.tensor([[0.1], [0.9]])
    maps, before = score.reshape(2, 1, 1, 1), processor.state_dict()
    result = processor(score, maps)
    assert torch.equal(result["pred_score"], score) and result["pred_score"].data_ptr() != score.data_ptr()
    assert result["pred_label"].flatten().tolist() == [False, True]
    assert_state_equal(before, processor.state_dict())
    state = processor.state_dict()
    for level in ("image", "pixel"):
        state[f"{level}_min"], state[f"{level}_max"] = torch.tensor(0.0), torch.tensor(10.0)
    processor.load_state_dict(state)
    torch.testing.assert_close(processor(torch.tensor([5.0]), None)["pred_score"], torch.tensor([0.5]))
    state["pixel_threshold"] = torch.tensor(7.0)
    processor.load_state_dict(state)
    torch.testing.assert_close(processor(torch.tensor([5.0]), None)["pred_score"], torch.tensor([0.3]))
    torch.testing.assert_close(PostProcessor()(None, maps)["pred_score"], score)
    assert processor(score, None)["pred_mask"] is None
    with pytest.raises(ValueError):
        processor(None, None)


def test_single_class_constant_range_and_sensitivity(calibration_outputs):
    """Continue no-positive calibration with a warning and preserve constant-range behavior."""
    outputs, processor = calibration_outputs, PostProcessor(image_sensitivity=0.6, pixel_sensitivity=0.6)
    processor.update(outputs)
    processor.finalize()
    assert processor(torch.tensor([2.0]), None)["pred_label"].item()
    outputs["gt_label"].zero_()
    with pytest.warns(UserWarning):
        processor.update(outputs)
        processor.finalize()
    assert processor.image_threshold == 0
    constant = PostProcessor()
    constant.update({"pred_score": torch.ones(2), "gt_label": torch.tensor([0, 1])})
    constant.finalize()
    assert torch.isnan(constant(torch.ones(2), None)["pred_score"]).all()
    state = processor.state_dict()
    with pytest.raises(ValueError):
        processor.load_state_dict({})
    assert_state_equal(state, processor.state_dict())


def test_default_contract_epochs_and_batching(evaluation_outputs):
    """Preserve metric names, fields, batch invariance and isolated epoch and stage state."""
    outputs, evaluator = evaluation_outputs, Evaluator.default()
    assert [(name, prediction, target) for name, _, prediction, target in evaluator.val_metrics] == [
        ("image_AUROC", "pred_score", "gt_label"), ("pixel_AUROC", "anomaly_map", "gt_mask")]
    assert [(name, prediction, target) for name, _, prediction, target in evaluator.test_metrics] == [
        ("image_AUROC", "pred_score", "gt_label"), ("pixel_AUROC", "anomaly_map", "gt_mask"),
        ("image_F1Score", "pred_label", "gt_label"), ("pixel_F1Score", "pred_mask", "gt_mask")]
    baseline = None
    for batch_size in (6, 3):
        for _ in range(2):
            evaluator.reset("test")
            for start in range(0, 6, batch_size):
                evaluator.update("test", {name: value[start:start + batch_size] for name, value in outputs.items()})
            values, reasons = evaluator.compute("test")
            assert all(reason is None for reason in reasons.values())
            assert values == pytest.approx({"image_AUROC": 7 / 9, "pixel_AUROC": 7 / 9,
                                            "image_F1Score": 2 / 3, "pixel_F1Score": 2 / 3}, abs=1e-6)
            baseline = values if baseline is None else baseline
            assert values == baseline
    assert all(math.isnan(value) for value in evaluator.compute("val")[0].values())
    evaluator.update("val", outputs)
    evaluator.reset("test")
    assert evaluator.compute("val")[0]["image_AUROC"] == pytest.approx(7 / 9, abs=1e-6)


def test_shared_instances_missing_pixels_and_invalid_stage():
    """Separate reused metric instances and report missing or single-class data honestly."""
    entry = ("image_AUROC", AUROC(), "pred_score", "gt_label")
    evaluator = Evaluator(val_metrics=[entry], test_metrics=[entry])
    assert evaluator.val_metrics[0][1] is not evaluator.test_metrics[0][1]
    evaluator.update("val", {"pred_score": torch.tensor([0.1, 0.9]), "gt_label": torch.tensor([0, 1])})
    assert math.isnan(evaluator.compute("test")[0]["image_AUROC"])
    evaluator = Evaluator.default()
    evaluator.update("val", {"pred_score": torch.tensor([0.1, 0.9]), "gt_label": torch.zeros(2)})
    values, reasons = evaluator.compute("val")
    assert all(math.isnan(value) for value in values.values())
    assert reasons == {"image_AUROC": "no_positive_targets", "pixel_AUROC": "no_data"}
    for operation in (evaluator.compute, evaluator.reset):
        with pytest.raises(ValueError):
            operation("train")


def test_visualization_and_safe_paths():
    """Keep image panel shape, immutable inputs, category identity and unsafe path rejection."""
    visualizer, image = ImageVisualizer(), torch.zeros(3, 4, 5)
    anomaly_map = torch.linspace(0, 1, 20).reshape(1, 4, 5)
    mask = anomaly_map > 0.5
    result = visualizer.visualize(image, anomaly_map, mask, mask, 0.8, True, 1)
    assert result.shape == (4, 20, 3) and result.dtype == np.uint8
    assert visualizer.visualize(image).shape == (4, 5, 3)
    assert visualizer.visualize(None, pred_mask=torch.ones(1, 1, 1)).shape == (1, 1, 3)
    assert torch.equal(image, torch.zeros_like(image))
    assert visualizer.file_name("bottle\\test\\broken_large\\000.jpg") == os.path.join("bottle", "test", "broken_large", "000.png")
    assert visualizer.file_name("bottle/test/bad/000.png") != visualizer.file_name("candle/test/bad/000.png")
    for path in ("../000.png", "bottle/../000.png", "bottle\\..\\000.png", "/tmp/000.png", "C:\\tmp\\000.png",
                 "C:000.png", "\\\\server\\share\\000.png", "", ".", "folder/"):
        with pytest.raises(ValueError):
            visualizer.file_name(path)


def test_learning_types_and_imports():
    """Preserve learning type names while subprocess dependency tests check imports."""
    assert {member.name: member.value for member in LearningType} == {
        "ONE_CLASS": "one_class", "ZERO_SHOT": "zero_shot", "FEW_SHOT": "few_shot"}
