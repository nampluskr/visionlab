# src/visionlab/anomaly_detection/common/evaluator.py: independent validation and test metric collections

import copy

from visionlab.anomaly_detection.common.metric import AUROC, F1


class Evaluator:
    """Evaluate named tensor fields with independent metrics for validation and test."""

    def __init__(self, val_metrics=None, test_metrics=None):
        self.val_metrics = [(name, copy.deepcopy(metric), prediction, target)
                            for name, metric, prediction, target in (val_metrics or [])]
        self.test_metrics = [(name, copy.deepcopy(metric), prediction, target)
                             for name, metric, prediction, target in (test_metrics or [])]
        for metrics in (self.val_metrics, self.test_metrics):
            names = [name for name, _, _, _ in metrics]
            if len(names) != len(set(names)):
                raise ValueError("Metric names must be unique within a stage")

    @classmethod
    def default(cls):
        return cls(
            val_metrics=[("image_AUROC", AUROC(), "pred_score", "gt_label"),
                         ("pixel_AUROC", AUROC(), "anomaly_map", "gt_mask")],
            test_metrics=[("image_AUROC", AUROC(), "pred_score", "gt_label"),
                          ("pixel_AUROC", AUROC(), "anomaly_map", "gt_mask"),
                          ("image_F1Score", F1(), "pred_label", "gt_label"),
                          ("pixel_F1Score", F1(), "pred_mask", "gt_mask")],
        )

    def _metrics(self, stage):
        if stage not in ("val", "test"):
            raise ValueError("stage must be 'val' or 'test'")
        return self.val_metrics if stage == "val" else self.test_metrics

    def update(self, stage, outputs):
        for _, metric, prediction, target in self._metrics(stage):
            if target == "gt_mask" and outputs.get(target) is None:
                continue
            if outputs.get(prediction) is None or outputs.get(target) is None:
                raise ValueError(f"Missing metric fields: {prediction}, {target}")
            metric.update(outputs[prediction], outputs[target])

    def compute(self, stage):
        values = {}
        reasons = {}
        for name, metric, _, _ in self._metrics(stage):
            values[name] = metric.compute()
            reasons[name] = metric.reason
        return values, reasons

    def reset(self, stage):
        for _, metric, _, _ in self._metrics(stage):
            metric.reset()
