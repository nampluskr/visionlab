# src/visionlab/anomaly_detection/common/visualizer.py: RGB anomaly panels and safe sample-relative output names

import os

import numpy as np
from PIL import Image


def _array(value):
    """Convert an array or detached CPU tensor to a NumPy view without modifying it."""
    if hasattr(value, "detach"):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def _plane(value):
    """Read a finite single-channel map without removing singleton spatial axes."""
    array = _array(value)
    while array.ndim > 2 and array.shape[0] == 1:
        array = array[0]
    if array.ndim != 2 or not all(array.shape) or not np.isfinite(array).all():
        raise ValueError("Maps must be finite nonempty single-channel images")
    return array


def _heatmap(values):
    """Interpolate a 256-entry RGB jet palette from the reference color anchors."""
    anchors = np.array([(0, 0, 143), (0, 0, 255), (0, 127, 255), (0, 255, 255),
                        (127, 255, 127), (255, 255, 0), (255, 127, 0), (255, 0, 0),
                        (127, 0, 0)], dtype=np.float32)
    positions = np.clip((np.arange(256, dtype=np.float32) + 0.5) * (9 / 256) - 0.5, 0, 8)
    lower = positions.astype(np.int64)
    upper = np.minimum(lower + 1, 8)
    weight = (positions - lower).astype(np.float32)[:, None]
    palette = (anchors[lower] * (1 - weight) + anchors[upper] * weight).astype(np.uint8)
    indices = (np.clip(values, 0, 1) * 255).astype(np.uint8)
    return palette[indices]


class ImageVisualizer:
    """Render available original, heat overlay and binary masks as adjacent RGB panels."""

    def visualize(self, image, anomaly_map=None, pred_mask=None, gt_mask=None,
                  pred_score=None, pred_label=None, gt_label=None):
        del pred_score, pred_label, gt_label
        panels = []
        original = None
        if image is not None:
            array = _array(image)
            if hasattr(image, "detach"):
                if array.ndim != 3 or array.shape[0] != 3:
                    raise ValueError("Tensor images must have shape [3, H, W]")
                array = array.transpose(1, 2, 0)
            if array.ndim != 3 or array.shape[2] != 3 or not all(array.shape) or not np.isfinite(array).all():
                raise ValueError("Images must be finite nonempty RGB arrays")
            if array.dtype != np.uint8:
                if array.min() < 0 or array.max() > 1:
                    raise ValueError("Float images must be in [0, 1]")
                array = (array * 255).astype(np.uint8)
            original = Image.fromarray(array)
            panels.append(np.array(original))
            size = original.size
        else:
            available = next((value for value in (anomaly_map, pred_mask, gt_mask) if value is not None), None)
            if available is None:
                raise ValueError("At least one image or map is required")
            height, width = _plane(available).shape
            size = (width, height)
        if anomaly_map is not None:
            heat = Image.fromarray(_heatmap(_plane(anomaly_map))).resize(size, resample=Image.Resampling.BILINEAR)
            panels.append(np.array(Image.blend(original, heat, alpha=0.5) if original is not None else heat))
        for mask in (pred_mask, gt_mask):
            if mask is not None:
                binary = Image.fromarray((_plane(mask) > 0).astype(np.uint8) * 255)
                panels.append(np.array(binary.resize(size, resample=Image.Resampling.NEAREST).convert("RGB")))
        return np.concatenate(panels, axis=1)

    @staticmethod
    def file_name(sample_id):
        if not isinstance(sample_id, str) or not sample_id or "\x00" in sample_id:
            raise ValueError("sample_id must be a nonempty relative file path")
        relative = sample_id.replace("\\", "/")
        parts = relative.split("/")
        if (os.path.isabs(sample_id) or relative.startswith("/") or ":" in relative
                or ".." in parts or parts[-1] in ("", ".")):
            raise ValueError("Absolute paths and parent traversal are forbidden in sample_id")
        return os.path.splitext(os.path.normpath(relative))[0] + ".png"
