# tests/common/test_auroc.py: pytest rank-preserving CPU AUROC across score scales and batches

import math

import pytest
import torch
from torchmetrics.functional.classification import binary_auroc

from visionlab.anomaly_detection.common.metric import AUROC


@pytest.mark.parametrize("dtype", (torch.float32, torch.float64))
def test_auroc_probability_scores_match_torchmetrics(dtype):
    """Match torchmetrics on bounded probabilities including duplicate values and endpoints."""
    generator = torch.Generator().manual_seed(42)
    scores = torch.randint(0, 17, (128,), generator=generator).to(dtype) / 16
    targets = torch.arange(128) % 2
    metric = AUROC()
    metric.update(scores, targets)
    assert metric.compute() == pytest.approx(binary_auroc(scores, targets).item(), abs=1e-6)


@pytest.mark.parametrize("scores, expected", (
    ([20.0, 22.0, 25.0, 30.0], 1.0),
    ([20.0, 25.0, 22.0, 30.0], 0.75),
    ([20.0, 25.0, 25.0, 30.0], 0.875),
    ([30.0, 25.0, 22.0, 20.0], 0.0),
    ([25.0, 25.0, 25.0, 25.0], 0.5),
))
def test_auroc_large_scores_match_hand_counted_pairs(scores, expected):
    """Count four positive-negative pairs with one point per win and half per tie."""
    metric = AUROC()
    metric.update(torch.tensor(scores), torch.tensor([0, 0, 1, 1]))
    assert metric.compute() == pytest.approx(expected, abs=1e-6)
    assert metric.positives == metric.negatives == 2 and metric.reason is None


def test_auroc_global_ranks_batching_and_repeated_compute():
    """Compute global ranks across interleaved batches and keep repeated computation idempotent."""
    scores = torch.tensor([20.0, 30.0, 25.0, 22.0, 25.0, 18.0])
    targets = torch.tensor([0, 1, 1, 0, 0, 1])
    whole, split = AUROC(), AUROC()
    whole.update(scores, targets)
    for start in range(0, scores.numel(), 2):
        split.update(scores[start:start + 2], targets[start:start + 2])
    expected = whole.compute()
    assert expected == pytest.approx(11 / 18, abs=1e-6)
    assert split.compute() == expected and split.compute() == expected
    split.update(torch.tensor([40.0]), torch.tensor([1]))
    whole.update(torch.tensor([40.0]), torch.tensor([1]))
    assert split.compute() == whole.compute()


@pytest.mark.parametrize("scores", (
    torch.tensor([20.0, 20.0 + 1e-12, 20.0 + 2e-12, 20.0 + 2e-12], dtype=torch.float64),
    torch.tensor([2**60, 2**60 + 1, 2**60 + 2, 2**60 + 2], dtype=torch.int64),
))
def test_auroc_rank_transform_preserves_close_values_and_ties(scores):
    """Retain distinct scores and exact ties in one float64 CPU ROC update."""
    metric = AUROC()
    metric.update(scores, torch.tensor([0, 0, 1, 1]))
    assert metric.compute() == 1.0
    assert len(metric.metric.preds) == 1
    transformed = metric.metric.preds[0]
    assert transformed.device.type == "cpu" and transformed.dtype == torch.float64
    torch.testing.assert_close(transformed, torch.tensor([0.0, 0.5, 1.0, 1.0], dtype=torch.float64), rtol=0, atol=0)


def test_auroc_snapshots_reset_and_undefined_reasons():
    """Snapshot CPU inputs and reset all accumulated scores, counters and undefined reasons."""
    metric = AUROC()
    scores = torch.tensor([20.0, 22.0, 25.0, 30.0])
    targets = torch.tensor([0, 0, 1, 1])
    metric.update(scores, targets)
    scores.zero_()
    targets.zero_()
    assert metric.compute() == 1.0
    assert all(value.device.type == "cpu" for value in metric.predictions + metric.targets)
    metric.reset()
    assert metric.reason is None and metric.positives == metric.negatives == 0
    assert not metric.predictions and not metric.targets
    assert math.isnan(metric.compute()) and metric.reason == "no_data"
    for label, reason in ((0, "no_positive_targets"), (1, "no_negative_targets")):
        metric.reset()
        metric.update(torch.tensor([20.0, 30.0]), torch.full((2,), label))
        assert math.isnan(metric.compute()) and metric.reason == reason
