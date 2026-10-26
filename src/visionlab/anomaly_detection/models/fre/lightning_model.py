# src/visionlab/anomaly_detection/models/fre/lightning_model.py: native Lightning training and evaluation for FRE

# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import math
import os

from lightning.pytorch import LightningModule
from lightning.pytorch.trainer.states import TrainerFn
from torch import nn, optim

from visionlab.anomaly_detection.common.evaluator import Evaluator
from visionlab.anomaly_detection.common.learning_type import LearningType
from visionlab.anomaly_detection.common.lightning_visualizer import LitVisualizer
from visionlab.anomaly_detection.common.post_processor import PostProcessor
from visionlab.anomaly_detection.common.pre_processor import IMAGENET_MEAN, IMAGENET_STD, PreProcessor
from visionlab.anomaly_detection.common.visualizer import ImageVisualizer
from visionlab.anomaly_detection.models.fre.model import FREModel
from visionlab.utils.io import load_teacher_weights, save_export


class FreLitModel(LightningModule):
    """Train the tied autoencoder and evaluate pure FRE outputs with native Lightning hooks."""

    def __init__(self, backbone="resnet50", layer="layer3", pre_trained=True,
                 pooling_kernel_size=2, input_dim=65536, latent_dim=220,
                 pre_processor=True, post_processor=True, evaluator=True, visualizer=True):
        super().__init__()
        self.pre_processor = self._resolve(
            pre_processor, PreProcessor, lambda: PreProcessor(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        )
        self.post_processor = self._resolve(post_processor, PostProcessor, PostProcessor)
        self.evaluator = self._resolve(evaluator, Evaluator, Evaluator.default)
        self.visualizer = self._resolve(visualizer, ImageVisualizer, ImageVisualizer)
        self.backbone = backbone
        self.layer = layer
        self.pooling_kernel_size = pooling_kernel_size
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.pre_trained = pre_trained
        self.save_hyperparameters({"backbone": backbone, "layer": layer, "pre_trained": pre_trained,
                                  "pooling_kernel_size": pooling_kernel_size, "input_dim": input_dim,
                                  "latent_dim": latent_dim,
                                  "_class_path": f"{type(self).__module__}.{type(self).__name__}"})
        self.model = FREModel(backbone=backbone, layer=layer, input_dim=input_dim, latent_dim=latent_dim,
                              pre_trained=pre_trained, pooling_kernel_size=pooling_kernel_size)
        # D-19: The loader unwraps TimmFeatureExtractor to inject the local backbone state.
        self.teacher_weights_path = load_teacher_weights(self.model.feature_extractor, backbone) if pre_trained else None
        self.loss_fn = nn.MSELoss()
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

    def training_step(self, batch, batch_idx):
        image, _ = batch
        if self.pre_processor is not None:
            image = self.pre_processor(image)
        features_in, features_out, _ = self.model.get_features(image)
        loss = self.loss_fn(features_in, features_out)
        # Match the reference epoch aggregation, where every step counts once.
        self.log("train_loss", loss.item(), on_epoch=True, prog_bar=True, logger=True, batch_size=1)
        return {"loss": loss}

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
                model_name="fre",
                model_args={"backbone": self.backbone, "layer": self.layer, "input_dim": self.input_dim,
                            "latent_dim": self.latent_dim, "pre_trained": False,
                            "pooling_kernel_size": self.pooling_kernel_size},
                pre_processor=self.pre_processor,
                post_processor=self.post_processor,
            )

    @property
    def trainer_arguments(self):
        # The original pins max_epochs=220; this phase uses 10 by user decision.
        return {"gradient_clip_val": 0, "max_epochs": 10, "num_sanity_val_steps": 0}

    @property
    def learning_type(self):
        return LearningType.ONE_CLASS

    def configure_optimizers(self):
        return optim.Adam(params=self.model.fre_model.parameters(), lr=1e-3)

    def configure_callbacks(self):
        return [] if self.visualizer is None else [LitVisualizer(self.visualizer)]
