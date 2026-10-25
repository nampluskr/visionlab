# src/visionlab/anomaly_detection/common/post_processor.py: CPU torchmetrics calibration and immutable inference post-processing

import torch
from torchmetrics.classification import BinaryPrecisionRecallCurve

from visionlab.anomaly_detection.common.metric import _binary_inputs


def _f1_threshold(prediction, target):
    """Select the first ascending raw threshold maximizing the reference PR-curve F1."""
    # The reference custom PR curve retains raw scores instead of applying sigmoid.
    curve = BinaryPrecisionRecallCurve(normalization=None).cpu()
    curve.update(prediction.detach().flatten().cpu(), target.detach().flatten().to(device="cpu", dtype=torch.int64))
    precision, recall, thresholds = curve.compute()
    f1 = 2 * precision * recall / (precision + recall + 1e-10)
    return (thresholds if thresholds.ndim == 0 else thresholds[torch.argmax(f1)]).clone()


class PostProcessor:
    """Calibrate at validation finalization and return new normalized prediction tensors."""

    state_names = ("image_threshold", "pixel_threshold", "image_min", "image_max", "pixel_min", "pixel_max")

    def __init__(self, image_sensitivity=0.5, pixel_sensitivity=0.5):
        if not (0 <= image_sensitivity <= 1 and 0 <= pixel_sensitivity <= 1):
            raise ValueError("Sensitivity must be between 0 and 1")
        self.image_sensitivity = image_sensitivity
        self.pixel_sensitivity = pixel_sensitivity
        for name in self.state_names:
            setattr(self, name, torch.tensor(float("nan")))
        self._pending = []

    @property
    def is_calibrated(self):
        return all(torch.isfinite(getattr(self, name)).item() for name in self.state_names)

    def update(self, outputs):
        batch = {}
        for prediction_name, target_name in (("pred_score", "gt_label"), ("anomaly_map", "gt_mask")):
            prediction = outputs.get(prediction_name)
            if prediction is None:
                continue
            if not isinstance(prediction, torch.Tensor) or not prediction.is_floating_point():
                raise ValueError("Raw predictions must be floating point tensors")
            if not torch.isfinite(prediction).all():
                raise ValueError("Raw predictions must be finite")
            if not prediction.numel():
                continue
            target = outputs.get(target_name)
            if target is not None:
                prediction, target = _binary_inputs(prediction, target)
                batch[target_name] = target
            else:
                prediction = prediction.detach().reshape(-1).cpu().clone()
            batch[prediction_name] = prediction
        if batch:
            self._pending.append(batch)

    def finalize(self):
        state = self.state_dict()
        for level, prediction_name, target_name in (
            ("image", "pred_score", "gt_label"), ("pixel", "anomaly_map", "gt_mask"),
        ):
            predictions = [batch[prediction_name] for batch in self._pending if prediction_name in batch]
            if predictions:
                values = torch.cat(predictions)
                state[f"{level}_min"] = values.min()
                state[f"{level}_max"] = values.max()
            paired = [batch for batch in self._pending if prediction_name in batch and target_name in batch]
            if paired:
                state[f"{level}_threshold"] = _f1_threshold(
                    torch.cat([batch[prediction_name] for batch in paired]),
                    torch.cat([batch[target_name] for batch in paired]),
                )
        for name, value in state.items():
            setattr(self, name, value)
        self._pending.clear()

    @staticmethod
    def _normalize(prediction, minimum, maximum, threshold):
        if prediction is None:
            return None
        if torch.isnan(minimum) or torch.isnan(maximum):
            return prediction.clone()
        if torch.isnan(threshold):
            threshold = (maximum + minimum) / 2
        return ((prediction - threshold.to(prediction)) / (maximum - minimum).to(prediction) + 0.5).clamp(0, 1)

    def __call__(self, pred_score, anomaly_map):
        if pred_score is None and anomaly_map is None:
            raise ValueError("At least one of pred_score or anomaly_map must be provided")
        for prediction in (pred_score, anomaly_map):
            if prediction is not None and (
                not isinstance(prediction, torch.Tensor) or not prediction.is_floating_point()
                or not torch.isfinite(prediction).all()
            ):
                raise ValueError("Raw predictions must be finite floating point tensors")
        if pred_score is None:
            pred_score = anomaly_map.amax(dim=(-2, -1))
        image_threshold = self.pixel_threshold if torch.isnan(self.image_threshold) else self.image_threshold
        pixel_threshold = self.image_threshold if torch.isnan(self.pixel_threshold) else self.pixel_threshold
        normalized_score = self._normalize(pred_score, self.image_min, self.image_max, image_threshold)
        normalized_map = self._normalize(anomaly_map, self.pixel_min, self.pixel_max, pixel_threshold)
        return {
            "pred_score": normalized_score,
            "anomaly_map": normalized_map,
            "pred_label": normalized_score > (1 - self.image_sensitivity),
            "pred_mask": None if normalized_map is None else normalized_map > (1 - self.pixel_sensitivity),
        }

    def state_dict(self):
        return {name: getattr(self, name).detach().cpu().clone() for name in self.state_names}

    def load_state_dict(self, state):
        if set(state) != set(self.state_names):
            raise ValueError("State must contain exactly the six calibration tensors")
        restored = {}
        for name in self.state_names:
            value = state[name]
            if not isinstance(value, torch.Tensor) or value.numel() != 1 or not value.is_floating_point():
                raise ValueError(f"Calibration state {name} must be a floating point scalar tensor")
            if torch.isinf(value).any():
                raise ValueError("Calibration state cannot contain infinity")
            restored[name] = value.detach().reshape(()).cpu().clone()
        for level in ("image", "pixel"):
            if restored[f"{level}_min"] > restored[f"{level}_max"]:
                raise ValueError("Calibration minimum cannot exceed maximum")
        for name, value in restored.items():
            setattr(self, name, value)
        self._pending.clear()
