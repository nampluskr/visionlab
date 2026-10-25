# src/visionlab/inference/predictor.py: export-only pure PyTorch model inference

import torch

from visionlab.anomaly_detection.common.post_processor import PostProcessor
from visionlab.anomaly_detection.common.pre_processor import PreProcessor
from visionlab.anomaly_detection.models.stfpm.model import STFPMModel
from visionlab.anomaly_detection.models.efficient_ad.model import EfficientAdModel
from visionlab.anomaly_detection.models.padim.model import PadimModel
from visionlab.utils.io import load_export


class Predictor:
    """Run a restored pure model while keeping raw and normalized outputs separate."""

    def __init__(self, model, pre_processor, post_processor, device="cpu"):
        self.device = torch.device(device)
        self.model = model.to(self.device).eval()
        self.pre_processor = pre_processor
        self.post_processor = post_processor

    @classmethod
    def from_export(cls, path, device="cpu"):
        exported = load_export(path)
        models = {"stfpm": STFPMModel, "efficient_ad": EfficientAdModel, "padim": PadimModel}
        model = models[exported["model_name"]](**exported["model_args"])
        model.load_state_dict(exported["model_state"], strict=True)
        pre_processor = PreProcessor(mean=exported["pre_processor"]["mean"], std=exported["pre_processor"]["std"])
        post_processor = None
        if exported["post_processor"] is not None:
            # shortcut: Exports omit sensitivities; extend the format before supporting custom values.
            post_processor = PostProcessor()
            post_processor.load_state_dict(exported["post_processor"])
        return cls(model=model, pre_processor=pre_processor, post_processor=post_processor, device=device)

    @torch.no_grad()
    def predict(self, images):
        if not isinstance(images, torch.Tensor) or images.dtype != torch.float32:
            raise ValueError("images must be a float32 RGB batch")
        prediction = self.model(self.pre_processor(images.to(self.device)))
        outputs = {"pred_score": prediction.pred_score, "anomaly_map": prediction.anomaly_map,
                   "normalized_pred_score": None, "normalized_anomaly_map": None,
                   "pred_label": None, "pred_mask": None}
        if self.post_processor is not None:
            processed = self.post_processor(prediction.pred_score, prediction.anomaly_map)
            outputs["normalized_pred_score"] = processed["pred_score"]
            outputs["normalized_anomaly_map"] = processed["anomaly_map"]
            outputs["pred_label"] = processed["pred_label"]
            outputs["pred_mask"] = processed["pred_mask"]
        return outputs
