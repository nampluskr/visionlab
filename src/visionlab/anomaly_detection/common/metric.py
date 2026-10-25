# src/visionlab/anomaly_detection/common/metric.py: CPU torchmetrics AUROC and F1 with explicit undefined reasons

import torch
from torchmetrics.classification import BinaryF1Score, BinaryROC
from torchmetrics.utilities.compute import auc


def _binary_inputs(prediction, target, binary_prediction=False):
    """Validate aligned finite predictions and binary targets and snapshot them on CPU."""
    if not isinstance(prediction, torch.Tensor) or not isinstance(target, torch.Tensor):
        raise ValueError("prediction and target must be tensors")
    if prediction.is_complex() or target.is_complex():
        raise ValueError("prediction and target must be real")
    if prediction.numel() != target.numel():
        raise ValueError("prediction and target must have equal element counts")
    if not torch.isfinite(prediction).all() or not torch.isfinite(target).all():
        raise ValueError("prediction and target must contain finite values")
    if not ((target == 0) | (target == 1)).all():
        raise ValueError("target must contain only 0 or 1")
    if binary_prediction and not ((prediction == 0) | (prediction == 1)).all():
        raise ValueError("F1 predictions must contain only 0 or 1")
    return prediction.detach().reshape(-1).cpu().clone(), target.detach().reshape(-1).to(device="cpu", dtype=torch.bool).clone()


def _undefined_reason(positives, negatives):
    """Describe missing target classes instead of silently returning a numeric score."""
    if positives + negatives == 0:
        return "no_data"
    if positives == 0:
        return "no_positive_targets"
    if negatives == 0:
        return "no_negative_targets"
    return None


class AUROC:
    """Compute rank-preserving ROC area on CPU and report missing target classes."""

    def __init__(self):
        self.metric = BinaryROC().cpu()
        self.reset()

    def update(self, prediction, target):
        prediction, target = _binary_inputs(prediction, target)
        if prediction.numel():
            self.predictions.append(prediction)
            self.targets.append(target)
        self.positives += int(target.sum())
        self.negatives += target.numel() - int(target.sum())
        self.reason = None

    def compute(self):
        self.reason = _undefined_reason(self.positives, self.negatives)
        if self.reason is not None:
            return float("nan")
        self.metric = BinaryROC().cpu()
        # Preserve global ordering and exact ties without BinaryROC's saturating sigmoid.
        unique, ranks = torch.unique(torch.cat(self.predictions), sorted=True, return_inverse=True)
        ranks = ranks.to(dtype=torch.float64).div_(max(unique.numel() - 1, 1))
        self.metric.update(ranks, torch.cat(self.targets).long())
        fpr, tpr, _ = self.metric.compute()
        return auc(fpr, tpr, reorder=True).item()

    def reset(self):
        self.metric.reset()
        self.predictions = []
        self.targets = []
        self.positives = 0
        self.negatives = 0
        self.reason = None


class F1:
    """Compute CPU BinaryF1Score while preserving undefined target reasons."""

    def __init__(self):
        self.metric = BinaryF1Score().cpu()
        self.reset()

    def update(self, prediction, target):
        prediction, target = _binary_inputs(prediction, target, binary_prediction=True)
        if prediction.numel():
            self.metric.update(prediction.long(), target.long())
        self.positives += int(target.sum())
        self.negatives += target.numel() - int(target.sum())
        self.reason = None

    def compute(self):
        self.reason = _undefined_reason(self.positives, self.negatives)
        if self.reason is not None:
            return float("nan")
        return self.metric.compute().item()

    def reset(self):
        self.metric.reset()
        self.positives = 0
        self.negatives = 0
        self.reason = None
