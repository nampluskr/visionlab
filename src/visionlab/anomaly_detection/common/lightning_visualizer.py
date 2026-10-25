# src/visionlab/anomaly_detection/common/lightning_visualizer.py: native Lightning PNG saving callback

import os

from lightning.pytorch import Callback
from PIL import Image


class LitVisualizer(Callback):
    """Save pure visualizer panels under sample-relative paths on rank zero."""

    def __init__(self, visualizer):
        self.visualizer = visualizer

    def _save(self, trainer, outputs):
        if not trainer.is_global_zero or outputs is None:
            return
        sample_ids = outputs.get("sample_id")
        if sample_ids is None:
            raise ValueError("Visualization requires sample_id")
        if isinstance(sample_ids, str):
            sample_ids = [sample_ids]
        for index, sample_id in enumerate(sample_ids):
            image = outputs.get("image")
            anomaly_map = outputs.get("normalized_anomaly_map", outputs.get("anomaly_map"))
            pred_score = outputs.get("normalized_pred_score", outputs.get("pred_score"))
            pred_mask = outputs.get("pred_mask")
            gt_mask = outputs.get("gt_mask")
            pred_label = outputs.get("pred_label")
            gt_label = outputs.get("gt_label")
            array = self.visualizer.visualize(
                image=None if image is None else image[index],
                anomaly_map=None if anomaly_map is None else anomaly_map[index],
                pred_mask=None if pred_mask is None else pred_mask[index],
                gt_mask=None if gt_mask is None else gt_mask[index],
                pred_score=None if pred_score is None else pred_score[index],
                pred_label=None if pred_label is None else pred_label[index],
                gt_label=None if gt_label is None else gt_label[index],
            )
            path = os.path.join(trainer.default_root_dir, "images", self.visualizer.file_name(sample_id))
            os.makedirs(os.path.dirname(path), exist_ok=True)
            Image.fromarray(array).save(path, format="PNG")

    def on_test_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0):
        self._save(trainer, outputs)

    def on_predict_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0):
        self._save(trainer, outputs)
