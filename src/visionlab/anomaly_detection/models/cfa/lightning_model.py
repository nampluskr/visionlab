# src/visionlab/anomaly_detection/models/cfa/lightning_model.py: native Lightning training and evaluation for CFA

# Copyright (C) 2022-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import math
import os

from lightning.pytorch import LightningModule
from lightning.pytorch.trainer.states import TrainerFn
from torch import optim

from visionlab.anomaly_detection.common.evaluator import Evaluator
from visionlab.anomaly_detection.common.learning_type import LearningType
from visionlab.anomaly_detection.common.lightning_visualizer import LitVisualizer
from visionlab.anomaly_detection.common.post_processor import PostProcessor
from visionlab.anomaly_detection.common.pre_processor import IMAGENET_MEAN, IMAGENET_STD, PreProcessor
from visionlab.anomaly_detection.common.visualizer import ImageVisualizer
from visionlab.anomaly_detection.models.cfa.loss import CfaLoss
from visionlab.anomaly_detection.models.cfa.model import CfaModel
from visionlab.utils.io import load_teacher_weights, save_export


class _CfaLitVisualizer(LitVisualizer):
    """Restore squeezed batch-one axes only for the shared image-saving callback."""

    def _save(self, trainer, outputs):
        if outputs is not None and outputs["image"].shape[0] == 1:
            outputs = dict(outputs)
            for name in ("pred_score", "normalized_pred_score", "pred_label"):
                if outputs.get(name) is not None:
                    outputs[name] = outputs[name].reshape(1)
            for name in ("anomaly_map", "normalized_anomaly_map", "pred_mask"):
                if outputs.get(name) is not None:
                    outputs[name] = outputs[name].reshape(1, *outputs[name].shape[-2:])
        super()._save(trainer, outputs)


class CfaLitModel(LightningModule):
    """Train descriptors and evaluate pure CFA outputs with native Lightning hooks."""

    def __init__(self, backbone="wide_resnet50_2", gamma_c=1, gamma_d=1, num_nearest_neighbors=3,
                 num_hard_negative_features=3, radius=1e-5,
                 pre_processor=True, post_processor=True, evaluator=True, visualizer=True):
        super().__init__()
        self.pre_processor = self._resolve(
            pre_processor, PreProcessor, lambda: PreProcessor(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        )
        self.post_processor = self._resolve(post_processor, PostProcessor, PostProcessor)
        self.evaluator = self._resolve(evaluator, Evaluator, Evaluator.default)
        self.visualizer = self._resolve(visualizer, ImageVisualizer, ImageVisualizer)
        self.backbone = backbone
        self.model_args = {"backbone": backbone, "gamma_c": gamma_c, "gamma_d": gamma_d,
                           "num_nearest_neighbors": num_nearest_neighbors,
                           "num_hard_negative_features": num_hard_negative_features, "radius": radius}
        self.save_hyperparameters(dict(self.model_args, _class_path=f"{type(self).__module__}.{type(self).__name__}"))
        self.model = CfaModel(backbone=backbone, gamma_c=gamma_c, gamma_d=gamma_d,
                              num_nearest_neighbors=num_nearest_neighbors,
                              num_hard_negative_features=num_hard_negative_features, radius=radius)
        self.loss = CfaLoss(num_nearest_neighbors=num_nearest_neighbors,
                            num_hard_negative_features=num_hard_negative_features, radius=radius)
        # D-19: Only the Lightning layer injects required local torchvision weights.
        self.teacher_weights_path = load_teacher_weights(self.model, backbone)
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

    def forward(self, image):
        if self.pre_processor is not None:
            image = self.pre_processor(image)
        return self.model(image)

    def on_train_start(self):
        # Reset the fitted bank shape before rebuilding centroids on every fit, including resume.
        self.model.memory_bank = self.model.memory_bank.new_zeros(())
        # Centroid initialization uses raw resized Dataset images without PreProcessor normalization.
        self.model.initialize_centroid(data_loader=self.trainer.datamodule.train_dataloader())

    def training_step(self, batch, batch_idx):
        image, _ = batch
        if self.pre_processor is not None:
            image = self.pre_processor(image)
        distance = self.model(image)
        loss = self.loss(distance)
        return {"loss": loss}

    def backward(self, loss, *args, **kwargs):
        del args, kwargs
        loss.backward(retain_graph=True)

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

    def _fit_validation(self):
        return (self._trainer is not None and self.trainer.state.fn == TrainerFn.FITTING
                and not self.trainer.sanity_checking)

    def validation_step(self, batch, batch_idx):
        outputs = self._outputs(batch)
        if self.post_processor is not None and self._fit_validation():
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
        if self.post_processor is not None and self._fit_validation():
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

    def on_fit_end(self):
        if self.trainer.is_global_zero:
            save_export(
                path=os.path.join(self.trainer.default_root_dir, "export", "model.pt"),
                model=self.model,
                model_name="cfa",
                model_args=dict(self.model_args),
                pre_processor=self.pre_processor,
                post_processor=self.post_processor,
            )

    @property
    def trainer_arguments(self):
        return {"gradient_clip_val": 0, "num_sanity_val_steps": 0}

    @property
    def learning_type(self):
        return LearningType.ONE_CLASS

    def configure_optimizers(self):
        return optim.AdamW(params=self.model.parameters(), lr=1e-3, weight_decay=5e-4, amsgrad=True)

    def configure_callbacks(self):
        return [] if self.visualizer is None else [_CfaLitVisualizer(self.visualizer)]
