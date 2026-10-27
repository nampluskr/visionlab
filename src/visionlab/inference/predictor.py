# src/visionlab/inference/predictor.py: export-only pure PyTorch model inference

import torch

from visionlab.anomaly_detection.common.post_processor import PostProcessor
from visionlab.anomaly_detection.common.pre_processor import PreProcessor
from visionlab.anomaly_detection.models.stfpm.model import STFPMModel
from visionlab.anomaly_detection.models.efficient_ad.model import EfficientAdModel
from visionlab.anomaly_detection.models.padim.model import PadimModel
from visionlab.anomaly_detection.models.dfm.model import DFMModel
from visionlab.anomaly_detection.models.dfkde.model import DfkdeModel
from visionlab.anomaly_detection.models.patchcore.model import PatchcoreModel
from visionlab.anomaly_detection.models.reverse_distillation.model import ReverseDistillationModel
from visionlab.anomaly_detection.models.cfa.model import CfaModel
from visionlab.anomaly_detection.models.fre.model import FREModel
from visionlab.anomaly_detection.models.supersimplenet.model import SupersimplenetModel
from visionlab.anomaly_detection.models.draem.model import DraemModel
from visionlab.anomaly_detection.models.reverse_distillation.anomaly_map import AnomalyMapGenerationMode
from visionlab.anomaly_detection.models.components.classification import FeatureScalingMethod
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
        models = {"stfpm": STFPMModel, "efficient_ad": EfficientAdModel, "padim": PadimModel,
                  "dfm": DFMModel, "dfkde": DfkdeModel, "patchcore": PatchcoreModel,
                  "reverse_distillation": ReverseDistillationModel, "cfa": CfaModel, "fre": FREModel,
                  "supersimplenet": SupersimplenetModel, "draem": DraemModel}
        if exported["model_name"] == "fastflow":
            from visionlab.anomaly_detection.models.fastflow.model import FastflowModel

            models["fastflow"] = FastflowModel
        model_args = dict(exported["model_args"])
        if exported["model_name"] == "dfkde":
            model_args["feature_scaling_method"] = FeatureScalingMethod(model_args["feature_scaling_method"])
        elif exported["model_name"] == "reverse_distillation":
            model_args["anomaly_map_mode"] = AnomalyMapGenerationMode(model_args["anomaly_map_mode"])
        model = models[exported["model_name"]](**model_args)
        model.load_state_dict(exported["model_state"], strict=True)
        pre_processor = PreProcessor(mean=exported["pre_processor"]["mean"], std=exported["pre_processor"]["std"],
                                     center_crop=exported["pre_processor"]["center_crop"])
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
