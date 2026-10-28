# src/visionlab/anomaly_detection/models/winclip/lightning_model.py: Native zero-shot WinCLIP calibration and export
# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import math
import os

import torch
from lightning.pytorch import LightningModule
from lightning.pytorch.trainer.states import TrainerFn

from visionlab.anomaly_detection.common.evaluator import Evaluator
from visionlab.anomaly_detection.common.learning_type import LearningType
from visionlab.anomaly_detection.common.lightning_visualizer import LitVisualizer
from visionlab.anomaly_detection.common.post_processor import PostProcessor
from visionlab.anomaly_detection.common.pre_processor import PreProcessor
from visionlab.anomaly_detection.common.visualizer import ImageVisualizer
from visionlab.anomaly_detection.models.winclip.model import WinClipModel
from visionlab.utils.io import load_pretrained_teacher, save_export

CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD = (0.26862954, 0.26130258, 0.27577711)


class WinClipLitModel(LightningModule):
    """Calibrate zero-shot WinCLIP and export all CLIP weights using native Lightning."""

    def __init__(self, class_name=None, k_shot=0, scales=(2, 3), few_shot_source=None,
                 pre_processor=True, post_processor=True, evaluator=True, visualizer=True):
        super().__init__()
        self.pre_processor = self._resolve(
            pre_processor, PreProcessor, lambda: PreProcessor(mean=CLIP_MEAN, std=CLIP_STD),
        )
        self.post_processor = self._resolve(post_processor, PostProcessor, PostProcessor)
        self.evaluator = self._resolve(evaluator, Evaluator, Evaluator.default)
        self.visualizer = self._resolve(visualizer, ImageVisualizer, ImageVisualizer)
        self.model = WinClipModel(scales=scales, apply_transform=False)
        self.teacher_weights_path = load_pretrained_teacher(self.model.clip, "winclip_clip")
        self.class_name = class_name
        self.k_shot = k_shot
        self.few_shot_source = few_shot_source
        self.is_setup = False
        # PredictDataset is not ported; only reference tensors from the native data module are supported.
        if few_shot_source is not None:
            raise NotImplementedError("few_shot_source is not supported; use the training dataloader")
        self.save_hyperparameters({"class_name": class_name, "k_shot": k_shot, "scales": scales,
                                  "few_shot_source": few_shot_source,
                                  "_class_path": f"{type(self).__module__}.{type(self).__name__}"})
        self.metric_reasons = {}
        self.last_metrics = {}

    @staticmethod
    def _resolve(value, component_type, factory):
        if value is True:
            return factory()
        if value is False:
            return None
        if isinstance(value, component_type):
            return value
        raise TypeError(f"Expected bool or {component_type.__name__}, received {type(value).__name__}")

    def setup(self, stage):
        if self.is_setup:
            return
        self.class_name = self._get_class_name()
        ref_images = None
        if self.k_shot:
            ref_images = self.collect_reference_images(self.trainer.datamodule.train_dataloader())
        self.model.setup(self.class_name, ref_images)
        self.is_setup = True

    def _get_class_name(self):
        if self.class_name is not None:
            return self.class_name
        if self._trainer is not None and hasattr(self.trainer.datamodule, "category"):
            return self.trainer.datamodule.category
        return "object"

    def collect_reference_images(self, dataloader):
        ref_images = torch.Tensor()
        for image, _ in dataloader:
            images = image[:self.k_shot - ref_images.shape[0]]
            if self.pre_processor is not None:
                images = self.pre_processor(images)
            ref_images = torch.cat((ref_images, images))
            if self.k_shot == ref_images.shape[0]:
                break
        return ref_images

    def forward(self, image):
        if self.pre_processor is not None:
            image = self.pre_processor(image)
        return self.model(image)

    def _outputs(self, batch):
        image, target = batch
        prediction = self(image)
        return {"image": image, "gt_label": target.get("label"), "gt_mask": target.get("mask"),
                "pred_score": prediction.pred_score, "anomaly_map": prediction.anomaly_map,
                "image_path": target.get("image_path"), "sample_id": target.get("sample_id")}

    def _post_process(self, outputs):
        if self.post_processor is not None:
            processed = self.post_processor(outputs["pred_score"], outputs["anomaly_map"])
            outputs = dict(outputs)
            outputs["pred_label"] = processed["pred_label"]
            outputs["pred_mask"] = processed["pred_mask"]
            outputs["normalized_pred_score"] = processed["pred_score"]
            outputs["normalized_anomaly_map"] = processed["anomaly_map"]
        return outputs

    def _calibrating_validation(self):
        if self._trainer is None or self.trainer.sanity_checking:
            return False
        return (self.trainer.state.fn == TrainerFn.FITTING or
                (self.trainer.state.fn == TrainerFn.VALIDATING and
                 self.learning_type in (LearningType.ZERO_SHOT, LearningType.FEW_SHOT)))

    def validation_step(self, batch, batch_idx):
        outputs = self._outputs(batch)
        if self.post_processor is not None and self._calibrating_validation():
            self.post_processor.update(outputs)
        if self.evaluator is not None:
            self.evaluator.update("val", outputs)
        return outputs

    def test_step(self, batch, batch_idx):
        outputs = self._post_process(self._outputs(batch))
        if self.evaluator is not None:
            self.evaluator.update("test", outputs)
        return outputs

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        return self._post_process(self._outputs(batch))

    def on_validation_epoch_start(self):
        if self.evaluator is not None:
            self.evaluator.reset("val")

    def on_test_epoch_start(self):
        if self.evaluator is not None:
            self.evaluator.reset("test")

    def _log_metrics(self, stage, prefix):
        if self.evaluator is None:
            return
        values, reasons = self.evaluator.compute(stage)
        self.last_metrics[stage] = dict(values)
        for name, value in values.items():
            key = prefix + name
            self.log(key, value, on_epoch=True, logger=True)
            if math.isnan(value):
                self.metric_reasons[key] = reasons[name]
            else:
                self.metric_reasons.pop(key, None)
        self.evaluator.reset(stage)

    def on_validation_epoch_end(self):
        if self.post_processor is not None and self._calibrating_validation():
            self.post_processor.finalize()
        self._log_metrics("val", "val_")

    def on_test_epoch_end(self):
        self._log_metrics("test", "")

    def on_save_checkpoint(self, checkpoint):
        checkpoint["post_processor"] = None if self.post_processor is None else self.post_processor.state_dict()

    def on_load_checkpoint(self, checkpoint):
        state = checkpoint.get("post_processor")
        if self.post_processor is not None and state is not None:
            self.post_processor.load_state_dict(state)

    def on_validation_end(self):
        if self.trainer.state.fn == TrainerFn.VALIDATING and self.trainer.is_global_zero:
            save_export(
                path=os.path.join(self.trainer.default_root_dir, "export", "model.pt"),
                model=self.model, model_name="winclip", model_args={"scales": list(self.model.scales)},
                pre_processor=self.pre_processor, post_processor=self.post_processor,
            )

    @property
    def trainer_arguments(self):
        return {}

    @property
    def learning_type(self):
        return LearningType.FEW_SHOT if self.k_shot else LearningType.ZERO_SHOT

    def configure_optimizers(self):
        return None

    def configure_callbacks(self):
        return [] if self.visualizer is None else [LitVisualizer(self.visualizer)]
